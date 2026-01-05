import cuda.tile as ct

batch = 2
seq_len = 1025
n_stream = 4  # consistent with deepseek paper


@ct.kernel
def sinkhorn_knopp(mat, out, num_iter, tilesize: ct.Constant[int]):
    """
    <typecheck>
    MockTensor((batch, seq_len, n_stream, n_stream), dtype="float32")
    MockTensor((batch, seq_len, n_stream, n_stream), dtype="float32")
    20
    32
    </typecheck>
    """
    i_batch = ct.bid(0)
    i_seq = ct.bid(1)

    tile = ct.load(
        mat,
        index=(i_batch, i_seq, 0, 0),
        shape=(1, tilesize, n_stream, n_stream),
    )

    tile = ct.exp(tile)

    for _ in range(num_iter):
        tile = tile / ct.sum(tile, axis=-2, keepdims=True)
        tile = tile / ct.sum(tile, axis=-1, keepdims=True)

    ct.store(
        out,
        index=(i_batch, i_seq, 0, 0),
        tile=tile,
    )


@ct.kernel
def sinkhorn_knopp_bwd_implicit(out, dout, res, num_iter, tilesize: ct.Constant[int]):
    """
    <typecheck>
    MockTensor((batch, seq_len, n_stream, n_stream), dtype="float32")
    MockTensor((batch, seq_len, n_stream, n_stream), dtype="float32")
    MockTensor((batch, seq_len, n_stream, n_stream), dtype="float32")
    20
    32
    </typecheck>
    """
    i_batch = ct.bid(0)
    i_seq = ct.bid(1)

    R = ct.load(
        out,
        index=(i_batch, i_seq, 0, 0),
        shape=(1, tilesize, n_stream, n_stream),
    )
    dR = ct.load(
        dout,
        index=(i_batch, i_seq, 0, 0),
        shape=(1, tilesize, n_stream, n_stream),
    )

    R = R.reshape((tilesize, n_stream, n_stream))
    dR = dR.reshape((tilesize, n_stream, n_stream))

    diffR = R - dR
    r = ct.sum(diffR, axis=-1).reshape((tilesize, n_stream, 1))
    c = ct.sum(diffR, axis=-2).reshape((tilesize, n_stream, 1))

    # TODO: sol = solve_linear(A, b)
    alpha = ct.zeros((tilesize, n_stream, 1), dtype=ct.float32)
    beta = ct.zeros((tilesize, n_stream, 1), dtype=ct.float32)

    R = R.reshape((tilesize, n_stream, n_stream))

    for _ in range(num_iter):
        alpha = r - ct.matmul(R, beta)
        beta = c - ct.matmul(R.transpose(-2, -1), alpha)

    alpha = alpha.reshape((tilesize, n_stream))
    beta = beta.reshape((tilesize, n_stream))

    res_tile = dR - ct.expand_dims(alpha, -1) - ct.expand_dims(beta, -2)

    ct.store(
        res,
        index=(i_batch, i_seq, 0, 0),
        tile=res_tile,
    )
