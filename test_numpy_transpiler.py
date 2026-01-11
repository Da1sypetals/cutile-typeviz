import os
from pathlib import Path
import numpy as np
import torch
import einops as ein
from ir_artifacts.sinkhorn_knopp_bwd_implicit_cg.numpy_code import sinkhorn_knopp_bwd_implicit_cg
from ir_artifacts.sinkhorn_knopp.numpy_code import sinkhorn_knopp
from icecream import ic


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

sinkhorn_knopp(M_np, np_sinkhorn, iters, 32, grid=(1, batch // 32, 1))
mae = torch.from_numpy(np_sinkhorn).sub(R).abs().mean()
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
print("Launch")
grad_M_implicit = sinkhorn_knopp_bwd_implicit_cg(out, dout, res, grid=(1, batch // 32, 1))
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
