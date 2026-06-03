import numpy as np
from tqdm import tqdm

# ── Pauli matrices ─────────────────────────────────────────────────────────────
I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = [SX, SY, SZ]


def _kron_op(ops):
    r = ops[0]
    for p in ops[1:]:
        r = np.kron(r, p)
    return r


def infer_N(M):
    assert M % 3 == 0, f"M={M} not divisible by 3."
    return M // 3 + 1


def precompute_basis_ops(N):
    """
    Pre-build all 3*(N-1) interaction operators once.
    Avoids redundant kron products inside the time loop.
    """
    ops_list = []
    for j in range(N - 1):
        for pauli in PAULIS:
            ops = [I2] * N
            ops[j] = pauli
            ops[j + 1] = pauli
            ops_list.append(_kron_op(ops))
    return ops_list


def build_H(t, coeffs, basis_ops, norm_N):
    """Build H(t) from pre-computed basis operators."""
    A, omega, phi = coeffs[0], coeffs[1], coeffs[2]
    c = (A * np.sin(omega * t + phi)) / norm_N
    return sum(c[i] * basis_ops[i] for i in range(len(c)))


def check_at_T(coeffs, T, norm_N=2.0, n_points=300):
    M = coeffs.shape[1]
    N = infer_N(M)
    basis_ops = precompute_basis_ops(N)

    t_grid = np.linspace(0, T, n_points)
    norms = np.array([
        np.linalg.svd(build_H(t, coeffs, basis_ops, norm_N),
                      compute_uv=False)[0]
        for t in tqdm(t_grid, desc="Checking convergence")
    ])

    integral  = float(np.trapezoid(norms, t_grid))
    converges = integral < np.pi

    return converges