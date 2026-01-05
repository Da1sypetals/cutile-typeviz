from icecream import ic
import torch

dtype = torch.float32

n = 4
iters = 10
print(f"{n = }")
print(f"{iters = }")

# Fix torch seed
torch.manual_seed(0)


######################################################################
# Sinkhorn forward
######################################################################
def sinkhorn_forward(M, iters=20):
    P = torch.exp(M)
    u = torch.ones(n, dtype=dtype)
    v = torch.ones(n, dtype=dtype)

    for _ in range(iters):
        u = 1.0 / (P @ v)
        v = 1.0 / (P.t() @ u)

    R = torch.diag(u) @ P @ torch.diag(v)
    return R, P, u, v


######################################################################
# Correct implicit backward for Sinkhorn (KL geometry)
######################################################################
def sinkhorn_backward_implicit(grad_R, R):
    # We solve:
    #
    # α + R β = r
    # R^T α + β = c
    #
    # where:
    # r_i = Σ_j R_ij * G_ij
    # c_j = Σ_i R_ij * G_ij
    #
    # Then projected gradient:
    # Gproj_ij = G_ij - α_i - β_j
    #
    # Finally:
    # grad_M = Gproj ⊙ R

    R = R.detach()

    r = (R * grad_R).sum(dim=1)  # shape (n,)
    c = (R * grad_R).sum(dim=0)  # shape (n,)

    # Build 2n x 2n system
    A = torch.zeros((2 * n, 2 * n), dtype=dtype)

    A[:n, :n] = torch.eye(n, dtype=dtype)
    A[:n, n:] = R
    A[n:, :n] = R.t()
    A[n:, n:] = torch.eye(n, dtype=dtype)

    ic(torch.linalg.svdvals(A))

    b = torch.cat([r, c])

    sol = torch.linalg.solve(A, b)

    alpha = sol[:n]
    beta = sol[n:]

    Gproj = grad_R - alpha[:, None] - beta[None, :]
    return Gproj


######################################################################
# Variable
######################################################################
M = torch.normal(0.0, 0.01, size=(n, n), dtype=dtype, requires_grad=True)

######################################################################
# Shared forward + one shared loss weight
######################################################################
R, P, u, v = sinkhorn_forward(M, iters)
loss_weight = torch.randn_like(R)

######################################################################
# Method A: Autograd
######################################################################
loss_a = (R * loss_weight).sum()
loss_a.backward()
grad_M_autograd = M.grad.detach().clone()

######################################################################
# Method B: Implicit differentiation
######################################################################
grad_R = loss_weight
Gproj = sinkhorn_backward_implicit(grad_R, R)

# KL pullback:
grad_M_implicit = Gproj * R


######################################################################
# Compare
######################################################################
g1 = grad_M_autograd
g2 = grad_M_implicit

abs_diff = (g1 - g2).abs()
rel_diff = abs_diff / (g1.abs() + 1e-12)

print("Comparison of gradients dL/dM")
print("--------------------------------")
print("MAE           :", abs_diff.mean().item())
print("Max abs diff  :", abs_diff.max().item())
print("Mean rel diff :", rel_diff.mean().item())
print("Max rel diff  :", rel_diff.max().item())

print("\nGrad (autograd) sample:\n", g1[:3, :3])
print("\nGrad (implicit) sample:\n", g2[:3, :3])
