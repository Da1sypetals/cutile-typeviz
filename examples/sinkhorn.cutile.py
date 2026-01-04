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
        mat,
        index=(i_batch, i_seq, 0, 0),
        tile=tile,
    )


@ct.kernel
def sinkhorn_knopp_bwd(out, dout, grad, num_iter, tilesize: ct.Constant[int]):
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
    # TODO
