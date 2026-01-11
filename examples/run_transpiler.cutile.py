"""
Example usage of the transpiler module.

This script demonstrates how to:
1. Get optimized IR from a kernel function
2. Eliminate tokens from the IR
3. Convert the IR to JSON format

Run with:
    cd /Users/daisy/develop/cutile-typeviz
    python src/cutile_typeviz/transpiler/example.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to find cutile_typeviz
sys.path.insert(0, str(Path(__file__).parent.parent))

from cutile_typeviz.cutile_utils.ir_dump.mock_tensor import MockTensor
from cutile_typeviz.transpiler import transpile
import cutile_typeviz.cutile_utils.cuda.tile as ct

batch = 2
seq_len = 1025
n_stream = 4  # consistent with deepseek paper
num_iter_cg = n_stream * 2

EPS = 1e-10
tilesize = 32


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

    # tile = -tile
    # tile = ct.sinh(tile)

    for _ in range(num_iter):
        tile = tile / ct.sum(tile, axis=-2, keepdims=True)
        tile = tile / ct.sum(tile, axis=-1, keepdims=True)

    ct.store(
        out,
        index=(i_batch, i_seq, 0, 0),
        tile=tile,
    )


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
    2. Matrix R is theoretically singular (not full-rank) and numerically near-singular,
        so the solution of x_sol can be very different from the real solution x_real.
        However, the tensor sum of the first half and the second half of x_sol is same with the result of x_real, which **is what we need**.
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

    RdR = R * dR
    # row sum
    b1 = ct.sum(RdR, axis=-1).reshape((tilesize, n_stream, 1))
    # col sum
    b2 = ct.sum(RdR, axis=-2).reshape((tilesize, n_stream, 1))

    b = ct.cat((b1, b2), axis=-2)

    # Solve: Ax=b =========================================
    R = R.reshape((tilesize, n_stream, n_stream))
    # Conjugate Gradients: init
    x = ct.zeros((tilesize, n_stream * 2, 1), dtype=ct.float32)
    r = b - matvec_A(R, x)
    p = r
    r_normsq = dot(r, r)

    # Conjugate Gradients: iter
    for it in range(num_iter_cg):
        Ap = matvec_A(R, p)
        pAp = dot(p, Ap)
        # VERY important to avoid divide by zero
        alpha = r_normsq / (pAp + EPS)
        x += alpha * p
        r -= alpha * Ap
        r_new_normsq = dot(r, r)
        # not very important to avoid divide by zero, but it's good to have it
        beta = r_new_normsq / (r_normsq + EPS)
        p = r + beta * p
        r_normsq = r_new_normsq
    # End solve: Ax=b =========================================

    x1 = ct.extract(x, index=(0, 0, 0), shape=(tilesize, n_stream, 1))
    x2 = ct.extract(x, index=(0, 1, 0), shape=(tilesize, n_stream, 1))

    x1_expanded = x1.reshape((tilesize, n_stream, 1))
    x2_expanded = x2.reshape((tilesize, 1, n_stream))

    res_tile = dR - x1_expanded - x2_expanded
    res_tile = res_tile * R
    res_tile = res_tile.reshape((1, tilesize, n_stream, n_stream))

    ct.store(
        res,
        index=(i_batch, i_seq, 0, 0),
        tile=res_tile,
    )


# cutile-typeviz: end


def main():
    print("Transpiler Example")
    print("-" * 30)
    print()

    out = MockTensor((batch, seq_len, n_stream, n_stream), dtype="float32")
    dout = MockTensor((batch, seq_len, n_stream, n_stream), dtype="float32")
    res = MockTensor((batch, seq_len, n_stream, n_stream), dtype="float32")
    num_iter = 24

    out_dir = Path.cwd() / "ir_artifacts"
    out_dir.mkdir(exist_ok=True)

    transpile(sinkhorn_knopp, [out, res, num_iter, tilesize], out_dir / "sinkhorn_knopp")
    transpile(sinkhorn_knopp_bwd_implicit_cg, [out, dout, res], out_dir / "sinkhorn_knopp_bwd_implicit_cg")


if __name__ == "__main__":
    main()
