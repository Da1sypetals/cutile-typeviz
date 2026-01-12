from pathlib import Path
import cutile_typeviz.cutile_utils.cuda.tile as ct


batch = 2
seq_len = 1025
n_stream = 4  # consistent with deepseek paper
num_iter_cg = n_stream * 2

EPS = 1e-10


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
    a = a.astype(ct.float16)
    b = b.astype(ct.float16)
    return ct.matmul(a.transpose(-2, -1), b).astype(ct.float32)


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

import torch
import einops as ein
from icecream import ic
import numpy as np
from cutile_typeviz.transpiler import launch_numpy
from pathlib import Path

dtype = torch.float32

batch = 160
n = 4
iters = 200
print(f"{n = }")
print(f"{iters = }")

# Fix torch seed
# torch.manual_seed(0)


def sinkhorn_forward(M, iters=20):
    P = torch.exp(M)
    R = P

    for _ in range(iters):
        R = R / R.sum(-2, keepdim=True)
        R = R / R.sum(-1, keepdim=True)

    return R, P


def batch_cg_solve(R, b):
    """
    Solve the system Ax = b using the Conjugate Gradient (CG) method.
    The matrix A is structured as:
    A = [[I,   R ],
         [R^T, I ]]
    """
    batch_size, n, _ = R.shape
    device = R.device
    dtype = R.dtype

    # 1. Construct the complete 2n x 2n matrix A
    # Create identity matrix I
    eye = torch.eye(n, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)

    # Concatenate blocks to form A
    # top: [I, R]
    top = torch.cat([eye, R], dim=-1)
    # bottom: [R^T, I]
    # Use einsum 'bij->bji' for transpose
    R_T = torch.einsum("bij->bji", R)
    bottom = torch.cat([R_T, eye], dim=-1)
    # A shape: (batch, 2n, 2n)
    A = torch.cat([top, bottom], dim=-2)

    # 2. CG Initialization
    # Initial guess x0 = 0, shape (batch, 2n)
    x = torch.zeros_like(b)

    # Initial residual r0 = b - A@x0 = b
    r = b.clone()

    # Initial search direction p0 = r0
    p = r.clone()

    # rs_old = r^T * r (dot product per batch)
    rs_old = torch.einsum("bi,bi->b", r, r)

    max_iter = 2 * n

    # 3. CG Iteration Loop
    for i in range(max_iter):
        # Calculate Ap = A @ p
        # 'bij,bj->bi' performs batch matrix-vector multiplication
        Ap = torch.einsum("bij,bj->bi", A, p)

        # Calculate step size alpha = (r^T * r) / (p^T * A * p)
        # pAp is the dot product of p and Ap per batch
        pAp = torch.einsum("bi,bi->b", p, Ap)
        # alpha = rs_old / pAp
        # Avoid division by zero here is very important
        alpha = rs_old / (pAp + 1e-12)

        # Update solution x = x + alpha * p
        # 'b,bi->bi' scales each vector in the batch by its corresponding alpha
        x += torch.einsum("b,bi->bi", alpha, p)

        # Update residual r = r - alpha * Ap
        r -= torch.einsum("b,bi->bi", alpha, Ap)

        # Calculate new residual inner product
        rs_new = torch.einsum("bi,bi->b", r, r)

        # Calculate beta = (r_new^T * r_new) / (r_old^T * r_old)
        # Avoid division by zero here is not so important experimentally
        # but it's good to have it
        beta = rs_new / (rs_old + 1e-12)

        # Update search direction p = r + beta * p
        p = r + torch.einsum("b,bi->bi", beta, p)

        rs_old = rs_new

    return x


def sinkhorn_backward_implicit(grad_R, R):
    R = R.detach()

    r = (R * grad_R).sum(dim=-1)  # shape (n,)
    c = (R * grad_R).sum(dim=-2)  # shape (n,)

    # Build 2n x 2n system
    A = torch.zeros((batch, 2 * n, 2 * n), dtype=dtype)

    A[:, :n, :n] = torch.eye(n, dtype=dtype).unsqueeze(0)
    A[:, :n, n:] = R
    A[:, n:, :n] = R.transpose(-2, -1)
    A[:, n:, n:] = torch.eye(n, dtype=dtype).unsqueeze(0)

    ic(torch.linalg.svdvals(A))

    b = torch.cat([r, c], dim=-1)

    ic(A.shape)
    ic(b.shape)

    # sol = torch.linalg.solve(A, b)
    sol = batch_cg_solve(R, b)

    alpha = sol[:, :n]
    beta = sol[:, n:]

    Gproj = grad_R - alpha.unsqueeze(-1) - beta.unsqueeze(-2)
    return Gproj * R


######################################################################
# Variable
######################################################################
dist = torch.distributions.uniform.Uniform(0.0, 4.0)
M = dist.sample((batch, n, n))
M.requires_grad_()


######################################################################
# Shared forward + one shared loss weight
######################################################################
R, P = sinkhorn_forward(M, iters)

M_np = M.detach().numpy()
M_np = np.expand_dims(M_np, axis=0)
R_np = R.detach().numpy()
np_sinkhorn = np.zeros_like(M_np)

fwd_dir = Path("ir_artifacts") / "sinkhorn_knopp"

launch_numpy(
    sinkhorn_knopp,
    [M_np, np_sinkhorn, iters, 32],
    grid=(1, batch // 32, 1),
    tmp_dir=fwd_dir,
)
np_sinkhorn = np_sinkhorn.squeeze(0)

# Compare
mae = torch.from_numpy(np_sinkhorn).sub(R).abs().mean().item()
print(f"{mae = }")
# set print precision to 0.001
np.set_printoptions(precision=3)
ic(R_np[0, :3, :3])
ic(np_sinkhorn[0, :3, :3])
print("\n\n")


loss_weight = torch.randn_like(R)

######################################################################
# Method A: Autograd
######################################################################
loss_a = (R * loss_weight).sum()
loss_a.backward()
grad_M_autograd = M.grad.detach().clone()

######################################################################
# Method B: NumPy Implicit differentiation
######################################################################
out = R.detach().numpy()
# insert a new dim for out at 0th
out = np.expand_dims(out, axis=0)
dout = np.expand_dims(loss_weight.detach().numpy(), axis=0)
res = np.zeros_like(out)
bwd_dir = Path("ir_artifacts") / "sinkhorn_knopp_bwd_implicit_cg"
launch_numpy(
    sinkhorn_knopp_bwd_implicit_cg,
    [out, dout, res],
    grid=(1, batch // 32, 1),
    tmp_dir=bwd_dir,
)
grad_M_implicit = torch.from_numpy(res).squeeze(0)


######################################################################
# Compare
######################################################################
g1 = grad_M_autograd
g2 = grad_M_implicit

abs_diff = (g1 - g2).abs()
rel_diff = abs_diff / (g1.abs() + 1e-12)

print("Comparison of gradients dL/dM")
print("--------------------------------")


def format_list(ls):
    return [f"{x:.2e}" for x in ls]


MAE = abs_diff.mean(dim=(-1, -2)).tolist()
max_abs_diff = abs_diff.reshape(batch, -1).max(-1).values.tolist()
mean_rel_diff = rel_diff.mean(dim=(-1, -2)).tolist()
max_rel_diff = rel_diff.reshape(batch, -1).max(-1).values.tolist()

# print(f"MAE: {format_list(MAE)}")
# print(f"max_abs_diff: {format_list(max_abs_diff)}")
# print(f"mean_rel_diff: {format_list(mean_rel_diff)}")
# print(f"max_rel_diff: {format_list(max_rel_diff)}")

print(f"Max MAE = {max(MAE)}")
print(f"Max max_abs_diff = {max(max_abs_diff)}")
print(f"Max mean_rel_diff = {max(mean_rel_diff)}")
print(f"Max max_rel_diff = {max(max_rel_diff)}")

print("\nGrad (autograd) sample:\n", g1[0, :3, :3])
print("\nGrad (implicit) sample:\n", g2[0, :3, :3])
