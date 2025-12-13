# Code from https://github.com/iori2333/kernels/blob/master/kernels/cutile/flash_sdpa.py
import math

import cuda.tile as ct
import numpy as np

INV_LOG2 = 1.0 / math.log(2)


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
    MockTensor((5, 16384, 16, 128), dtype="bfloat16")
    MockTensor((5, 16384, 4, 128), dtype="bfloat16")
    MockTensor((5, 16384, 4, 128), dtype="bfloat16")
    MockTensor((5, 16384, 16, 128), dtype="bfloat16")
    1.0
    4
    32
    64
    16
    128
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
        k_jt = ct.load(
            k,
            index=(bid_b, 0, bid_hkv, j),
            shape=(1, d, 1, bc),
            order=(0, 3, 2, 1),  # transpose here
        ).reshape((d, bc))

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


@ct.kernel
def apply_rope(
    q: ct.Array,  # [b, s, h, 2, d // 2]
    k: ct.Array,  # [b, s, h_kv, 2, d // 2]
    cos: ct.Array,  # [b, s, 2, d // 2]
    sin: ct.Array,  # [b, s, 2, d // 2]
    out_q: ct.Array,
    out_k: ct.Array,
    thq: ct.Constant[int],
    thk: ct.Constant[int],
    td: ct.Constant[int],
):
    """
    <typecheck>
    MockTensor((5, 16384, 16, 2, 64), dtype="bfloat16")
    MockTensor((5, 16384, 4, 2, 64), dtype="bfloat16")
    MockTensor((5, 16384, 2, 64), dtype="float32")
    MockTensor((5, 16384, 2, 64), dtype="float32")
    MockTensor((5, 16384, 16, 2, 64), dtype="bfloat16")
    MockTensor((5, 16384, 4, 2, 64), dtype="bfloat16")
    16
    4
    64
    </typecheck>
    """
    bid_b = ct.bid(0)
    bid_s = ct.bid(1)

    cos_tile_1 = ct.load(
        cos,
        index=(bid_b, bid_s, 0, 0),
        shape=(1, 1, 1, td),
    ).reshape((1, td))

    cos_tile_2 = ct.load(
        cos,
        index=(bid_b, bid_s, 1, 0),
        shape=(1, 1, 1, td),
    ).reshape((1, td))

    sin_tile_1 = ct.load(
        sin,
        index=(bid_b, bid_s, 0, 0),
        shape=(1, 1, 1, td),
    ).reshape((1, td))

    sin_tile_2 = ct.load(
        sin,
        index=(bid_b, bid_s, 1, 0),
        shape=(1, 1, 1, td),
    ).reshape((1, td))

    q_tile_1 = ct.load(q, index=(bid_b, bid_s, 0, 0, 0), shape=(1, 1, thq, 1, td))
    q_tile_1 = q_tile_1.reshape((thq, td))
    q_tile_2 = ct.load(q, index=(bid_b, bid_s, 0, 1, 0), shape=(1, 1, thq, 1, td))
    q_tile_2 = q_tile_2.reshape((thq, td))

    # [q_1, q_2] * [cos_1, cos_2] + [-q_2, q_1] * [sin_1, sin_2]
    # [q_1cos_1 - q_2sin_1, q_2cos_2 + q_1sin_2]
    q_1 = q_tile_1 * cos_tile_1 - q_tile_2 * sin_tile_1
    q_1 = q_1.reshape((1, 1, thq, 1, td)).astype(out_q.dtype)
    q_2 = q_tile_2 * cos_tile_2 + q_tile_1 * sin_tile_2
    q_2 = q_2.reshape((1, 1, thq, 1, td)).astype(out_q.dtype)

    ct.store(out_q, index=(bid_b, bid_s, 0, 0, 0), tile=q_1)
    ct.store(out_q, index=(bid_b, bid_s, 0, 1, 0), tile=q_2)

    k_tile_1 = ct.load(k, index=(bid_b, bid_s, 0, 0, 0), shape=(1, 1, thk, 1, td))
    k_tile_1 = k_tile_1.reshape((thk, td))
    k_tile_2 = ct.load(k, index=(bid_b, bid_s, 0, 1, 0), shape=(1, 1, thk, 1, td))
    k_tile_2 = k_tile_2.reshape((thk, td))

    # [k_1, k_2] * [cos_1, cos_2] + [-k_2, k_1] * [sin_1, sin_2]
    # [k_1cos_1 - k_2sin_1, k_2cos_2 + k_1sin_2]
    k_1 = k_tile_1 * cos_tile_1 - k_tile_2 * sin_tile_1
    k_1 = k_1.reshape((1, 1, thk, 1, td)).astype(out_k.dtype)
    k_2 = k_tile_2 * cos_tile_2 + k_tile_1 * sin_tile_2
    k_2 = k_2.reshape((1, 1, thk, 1, td)).astype(out_k.dtype)

    ct.store(out_k, index=(bid_b, bid_s, 0, 0, 0), tile=k_1)
    ct.store(out_k, index=(bid_b, bid_s, 0, 1, 0), tile=k_2)
