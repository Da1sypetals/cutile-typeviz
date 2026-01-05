from icecream import ic

import numpy as np


def sinkhorn_knopp(M, max_iter=1000, tol=1e-12):
    """
    Convert a positive square matrix M into a bistochastic matrix using Sinkhorn-Knopp algorithm.
    [[18]], [[17]]
    """
    M = M.copy()
    for it in range(max_iter):
        # Normalize rows
        row_sums = M.sum(axis=1, keepdims=True)
        M /= row_sums
        # Normalize columns
        col_sums = M.sum(axis=0, keepdims=True)
        M /= col_sums
        # Check convergence
        if np.max(np.abs(M.sum(axis=1) - 1)) < tol and np.max(np.abs(M.sum(axis=0) - 1)) < tol:
            print("break", it)
            break
    return M


def matvec_A(R, x):
    """Efficient matvec for A = [I, R; R^T, I]"""
    x1 = x[:4]
    x2 = x[4:]
    Ax1 = x1 + R @ x2
    Ax2 = R.T @ x1 + x2
    return np.concatenate([Ax1, Ax2])


def conjugate_gradient(R, b, max_iter=200, tol=1e-12):
    """CG solver for Ax = b with A = [I, R; R^T, I]"""
    x = np.zeros(8)
    r = b - matvec_A(R, x)
    p = r.copy()
    r_norm_sq = np.dot(r, r)

    for k in range(max_iter):
        if r_norm_sq < tol:
            print(f"break at {k = }")
            break
        Ap = matvec_A(R, p)
        alpha = r_norm_sq / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        r_new_norm_sq = np.dot(r, r)
        beta = r_new_norm_sq / r_norm_sq
        p = r + beta * p
        r_norm_sq = r_new_norm_sq
    return x


# np.random.seed(42)  # reproducibility

# Step 1: Generate random positive 4x4 matrix and make it bistochastic
M = np.random.randn(4, 4) + 0.1  # ensure positivity
R = sinkhorn_knopp(np.exp(M))

# Verify R is bistochastic
assert np.allclose(R.sum(axis=1), 1, atol=1e-10), "Rows must sum to 1"
assert np.allclose(R.sum(axis=0), 1, atol=1e-10), "Columns must sum to 1"
print("✅ R is bistochastic")

# Step 2: Construct ground truth solution and compute b = A @ x_true
x_true = np.random.randn(8)
b = matvec_A(R, x_true)

# Step 3: Solve using CG
x_computed = conjugate_gradient(R, b)

# Step 4: Verify accuracy
print(x_computed)
print(x_true)
residual = np.linalg.norm(matvec_A(R, x_computed) - b)
res = x_computed[:4][:, None] + x_computed[4:][None, :]
res1 = x_true[:4][:, None] + x_true[4:][None, :]

ic(res)
ic(res1)
expected_diff = np.linalg.norm(res - res1)
ic(expected_diff)

print(f"Residual (||Ax - b||): {residual:.2e}")
