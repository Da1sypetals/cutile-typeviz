import json
import cuda.tile as ct


@ct.kernel
def flash_attention_forward_v2(
    q,
    k,
    v,
    out,
    hidden_size: ct.Constant,
    br: ct.Constant,
    bc: ct.Constant,
):
    ib = ct.bid(0)
    ih = ct.bid(1)
    z = ih * ct.float64(244.1)
    Tc = k.shape[0] // bc
    qi = ct.load(q, index=(ib, ih, ct.bid(2), 0), shape=(1, 1, br, hidden_size))
    y = ct.full((2, 4), 2.8751, dtype=ct.float32)
    qw = ct.full((4,), 2.113, dtype=ct.float32)
    ma = ct.matmul(y, qw)

    qi = ct.reshape(qi, (br, hidden_size))

    oi = ct.full((br, hidden_size), 0.0, dtype=q.dtype)
    li = ct.full((br, 1), 0.0, dtype=q.dtype)
    mi = ct.full((br, 1), -1e10, dtype=q.dtype)

    for j in range(0, Tc):
        kjt = ct.load(k, index=(ib, ih, j, 0), shape=(1, 1, bc, hidden_size))
        vj = ct.load(v, index=(ib, ih, j, 0), shape=(1, 1, bc, hidden_size))
        kjt = ct.reshape(kjt, (bc, hidden_size))
        vj = ct.reshape(vj, (bc, hidden_size))
        kj = kjt.transpose()
        sij = ct.matmul(qi, kj) / hidden_size**0.5
        mij = ct.max(sij, axis=-1, keepdims=True)
        mi_mij = ct.cat((mi, mij), axis=-1)
        mi_new = ct.max(mi_mij, axis=-1, keepdims=True)
        pij = ct.exp(sij - mi_new)
        lij = ct.sum(pij, axis=-1, keepdims=True)
        exp_mi = ct.exp(mi - mi_new)
        li_new = li * exp_mi + lij
        oi = ct.mma(pij, vj, oi * exp_mi)
        li = li_new
        mi = mi_new

    oi = oi / li
    ct.store(out, index=(ib, ih, ct.bid(2), 0), tile=oi)


if __name__ == "__main__":
    from ir_dump.mock_tensor import MockTensor
    from ir_dump.shape_check import get_kernel_shapes_info

    # 定义参数
    batch_size = 8
    num_head = 16
    seq_len = 1024
    hidden_size = 64
    Br = 32  # block row size
    Bc = 32  # block col size

    q = MockTensor((batch_size, num_head, seq_len, hidden_size), dtype="float32")
    k = MockTensor((batch_size, num_head, seq_len, hidden_size), dtype="float32")
    v = MockTensor((batch_size, num_head, seq_len, hidden_size), dtype="float32")
    out = MockTensor((batch_size, num_head, seq_len, hidden_size), dtype="float32")

    assignment_ops = get_kernel_shapes_info(
        kernel_func=flash_attention_forward_v2,
        args=[q, k, v, out, hidden_size, Br, Bc],
    )
    ops_str = json.dumps(assignment_ops)
    print(ops_str)
