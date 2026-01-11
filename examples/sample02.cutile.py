# Code from https://github.com/iori2333/kernels/blob/master/kernels/cutile/flash_sdpa.py
import math

import cuda.tile as ct
import numpy as np

INV_LOG2 = 1.0 / math.log(2)

Batch = 5
Sequence = 1024
Head = 16
HeadKV = 4
Dim = 128
Groups = Head // HeadKV


@ct.kernel
def flash_sdpa(
    q: ct.Array,  # [b, s, h, d]
    k: ct.Array,  # [b, s_kv, h_kv, d]
    v: ct.Array,  # [b, s_kv, h_kv, d]
    o: ct.Array,  # [b, s, h, d]
    qk_scale: float,
    groups: ct.Constant[int],
    br: ct.Constant[int],
    bc: ct.Constant[int],
    h: ct.Constant[int],
    d: ct.Constant[int],
):
    """
    <typecheck>
    MockTensor((Batch, Sequence, Head, Dim), dtype="bfloat16")
    MockTensor((Batch, Sequence, HeadKV, Dim), dtype="bfloat16")
    MockTensor((Batch, Sequence, HeadKV, Dim), dtype="bfloat16")
    MockTensor((Batch, Sequence, Head, Dim), dtype="bfloat16")
    1.0
    Groups
    32
    64
    Head
    Dim
    </typecheck>
    """
    bid_b_h = ct.bid(0)
    bid_b = bid_b_h // h
    bid_s = ct.bid(1)
    bid_h = bid_b_h % h
    bid_hkv = bid_h // groups

    # trick: use log2 instead of loge
    qk_scale = qk_scale * INV_LOG2

    # initialize buffers
    l_i = ct.zeros((br, 1), dtype=ct.float32)
    m_i = ct.full((br, 1), -np.inf, dtype=ct.float32)
    o_i = ct.zeros((br, d), dtype=ct.float32)

    # load q_i
    q_i = ct.load(
        q,
        index=(bid_b, bid_s, bid_h, 0),
        shape=(1, br, 1, d),
    ).reshape((br, d))

    t_c = ct.cdiv(k.shape[1], bc)
    for j in range(t_c):  # type: ignore
        # load (k_j)^T and v_j
        k_jt = (
            ct.load(
                k,
                index=(bid_b, j, bid_hkv, 0),
                shape=(1, bc, 1, d),
            )
            .reshape((bc, d))
            .transpose()
        )

        v_j = ct.load(
            v,
            index=(bid_b, j, bid_hkv, 0),
            shape=(1, bc, 1, d),
        ).reshape((bc, d))

        # calculate s_ij = q_i @ (k_j)^T
        s_ij = ct.zeros((br, bc), dtype=ct.float32)
        s_ij = ct.mma(q_i, k_jt, s_ij)

        # perform online softmax
        s_ij_rowmax = ct.max(s_ij, axis=1, keepdims=True)
        m_ij = max(m_i, s_ij_rowmax * qk_scale)
        p_ij = ct.exp2(s_ij * qk_scale - m_ij)  # [br, bc]

        alpha = ct.exp2(m_i - m_ij)  # [br, 1]
        l_i = l_i * alpha + ct.sum(p_ij, axis=-1, keepdims=True)

        # calculate o_i = alpha * o_i-1 + p_ij @ v_j
        o_i = o_i * alpha
        p_ij = p_ij.astype(v_j.dtype)  # type: ignore
        o_i = ct.mma(p_ij, v_j, o_i)

        # write back m_i
        m_i = m_ij

    # scale o_i
    o_i = o_i / l_i
    o_i = o_i.reshape((1, br, 1, d)).astype(o.dtype)
    ct.store(o, index=(bid_b, bid_s, bid_h, 0), tile=o_i)


# cutile-typeviz: end

import numpy as np
from cutile_typeviz.transpiler import launch_numpy
from pathlib import Path
import torch
from icecream import ic

q = np.random.uniform(low=0.0, high=4.0, size=(Batch, Sequence, Head, Dim)).astype(np.float32)
k = np.random.uniform(low=0.0, high=4.0, size=(Batch, Sequence, HeadKV, Dim)).astype(np.float32)
v = np.random.uniform(low=0.0, high=4.0, size=(Batch, Sequence, HeadKV, Dim)).astype(np.float32)
out = np.zeros_like(q)

br = 32
bc = 64

tmp_dir = Path("ir_artifacts") / "sdpa"

launch_numpy(
    flash_sdpa,
    [q, k, v, out, Dim ** (-0.5), Groups, br, bc, Head, Dim],
    grid=(Batch * Head, Sequence // br, 1),
    tmp_dir=tmp_dir,
)


# TODO

q_torch = torch.from_numpy(q).permute(0, 2, 1, 3)  # (b, h, s, d)
k_torch = torch.from_numpy(k).permute(0, 2, 1, 3)  # (b, h_kv, s_kv, d)
v_torch = torch.from_numpy(v).permute(0, 2, 1, 3)  # (b, h_kv, s_kv, d)

out_torch = torch.nn.functional.scaled_dot_product_attention(q_torch, k_torch, v_torch, enable_gqa=True)
expected = out_torch.permute(0, 2, 1, 3).numpy()

mae = np.abs(out - expected).mean()
print(f"MAE: {mae}")
ic(out[0, 0, :3, :3])
ic(expected[0, 0, :3, :3])
