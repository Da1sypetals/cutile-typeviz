import cuda.tile as ct

batch = 2
seq_len = 1025
n_stream = 4  # consistent with deepseek paper
num_iter_cg = n_stream * 2


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


tilesize = 32


@ct.function(host=False, tile=True)
def matvec_A(R, x):
    """
    R: (tilesize, n_stream, n_stream)
    x: (tilesize, n_stream*2, 1)
    """
    x1 = ct.extract(x, index=(0, 0, 0), shape=(tilesize, n_stream, 1))
    x2 = ct.extract(x, index=(0, 1, 0), shape=(tilesize, n_stream, 1))
    ax1 = x1 + ct.matmul(R, x2)
    ax2 = ct.matmul(R.transpose(-2, -1), x1) + x2
    return ct.cat((ax1, ax2), axis=-2)  # (tilesize, n_stream*2, 1)


@ct.function(host=False, tile=True)
def dot(a, b):  # a/b: (..., dim, 1)
    return ct.matmul(a.transpose(-2, -1), b)


@ct.kernel
def sinkhorn_knopp_bwd_implicit_cg(out, dout, res):
    """
    <typecheck>
    MockTensor((batch, seq_len, n_stream, n_stream), dtype="float32")
    MockTensor((batch, seq_len, n_stream, n_stream), dtype="float32")
    MockTensor((batch, seq_len, n_stream, n_stream), dtype="float32")
    </typecheck>

    Side note:
    1. Number of CG iterations is typically num_streams*2.
        This is derived from the theoretical properties of CG method.
    2. Matrix R is typically near-ingular (not full-rank), so the solution of x_sol can be very different from the real solution x_real.
        However, the outer product of the first half and the second half of x_sol is same with the result of x_real, which **is what we need**.
        This means the solution set has some mathematical property that applies to every element in it.
        We shall make use of that property.
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
    # row sum
    b1 = ct.sum(diffR, axis=-1).reshape((tilesize, n_stream, 1))
    # col sum
    b2 = ct.sum(diffR, axis=-2).reshape((tilesize, n_stream, 1))

    b = ct.cat((b1, b2), axis=-2)

    # Solve: Ax=b
    R = R.reshape((tilesize, n_stream, n_stream))
    # Conjugate Gradients: init
    x = ct.zeros((tilesize, n_stream * 2, 1), dtype=ct.float32)
    r = b - matvec_A(R, x)
    p = r
    r_normsq = dot(r, r)

    # Conjugate Gradients: iter
    for _ in range(num_iter_cg):
        Ap = matvec_A(R, p)
        alpha = r_normsq / dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        r_new_normsq = dot(r, r)
        beta = r_new_normsq / r_normsq
        p = r + beta * p
        r_normsq = r_new_normsq

    x1 = ct.extract(x, index=(0, 0, 0), shape=(tilesize, n_stream, 1)).reshape((tilesize, n_stream))
    x2 = ct.extract(x, index=(0, 1, 0), shape=(tilesize, n_stream, 1)).reshape((tilesize, n_stream))

    res_tile = dR - ct.expand_dims(x1, -1) - ct.expand_dims(x2, -2)

    ct.store(
        res,
        index=(i_batch, i_seq, 0, 0),
        tile=res_tile,
    )
