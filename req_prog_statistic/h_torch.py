import torch
from torch.linalg import matrix_exp

# ----------------------------------------------------------
#  Build fixed Ising operator basis (only computed once)
# ----------------------------------------------------------
def build_ops_ising(n, device="cpu"):
    """
    Build the fixed operator basis {X_i, Z_i Z_{i+1}} for the 1D Ising Hamiltonian.
    This function should be called once and reused, never rebuilt inside the training loop.

    Args:
        n (int): Number of qubits.
        device (str): Device to place the operators.

    Returns:
        ops (torch.Tensor): Tensor of shape (2n-1, 2^n, 2^n).
    """
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat)
    I = torch.eye(2, dtype=torch.cfloat)

    def kron_n(ops):
        out = ops[0]
        for op in ops[1:]:
            out = torch.kron(out, op)
        return out

    # Build local X_i terms
    X_ops = [kron_n([X if i == j else I for i in range(n)]) for j in range(n)]

    # Build nearest-neighbor Z_i Z_{i+1} terms
    ZZ_ops = [kron_n([Z if i == j or i == j+1 else I for i in range(n)]) for j in range(n-1)]

    ops = X_ops + ZZ_ops  # total 2n - 1 operators
    ops = torch.stack(ops).to(device)   # (2n - 1, d, d)
    return ops


# ----------------------------------------------------------
#  Build Hamiltonians H(t) efficiently (no VRAM explosion)
# ----------------------------------------------------------
def get_ising_H_batch(model, time_batch, ops):
    """
    Compute a batch of Ising Hamiltonians H(t) using a fixed operator basis ops.

    Args:
        model (nn.Module): Maps time t → Hamiltonian coefficients (B, K).
        time_batch (torch.Tensor): Time values (B, 1).
        ops (torch.Tensor): Fixed operator basis of shape (K, d, d).

    Returns:
        torch.Tensor: Hamiltonians of shape (B, d, d).
    """
    coeffs = model(time_batch)  # (B, K)
    coeffs = coeffs.to(dtype=ops.dtype, device=ops.device)
    
    H = torch.einsum("bk,kij->bij", coeffs, ops)
    return H


# ----------------------------------------------------------
#  Second-order Magnus expansion
# ----------------------------------------------------------
def unitary_magnus_torch(model, time, ops, steps=50):
    """
    Compute U(t) ≈ exp(-i Ω(t)) using the 2nd-order Magnus expansion.

    Args:
        model (nn.Module): Network mapping t → Hamiltonian coefficients.
        time (torch.Tensor): Shape (B, 1) target times.
        ops (torch.Tensor): Operator basis (K, d, d).
        steps (int): Number of discretization slices.

    Returns:
        torch.Tensor: Unitaries of shape (B, d, d).
    """
    device = time.device
    B = time.shape[0]
    K, d, _ = ops.shape
    dt = time / steps  # (B, 1)

    # Build time grid
    t_lin = torch.linspace(0, 1, steps, device=device).view(1, steps, 1)
    t_grid = time[:, None, :] * t_lin  # (B, steps, 1)
    t_flat = t_grid.reshape(-1, 1)

    # Evaluate all Hamiltonians
    H_all = get_ising_H_batch(model, t_flat, ops).view(B, steps, d, d)

    # First-order Magnus term
    Omega1 = H_all.sum(dim=1) * dt.view(B, 1, 1)

    # Second-order Magnus term
    Omega2 = torch.zeros(B, d, d, dtype=torch.cfloat, device=device)
    dt_sq = (dt ** 2).view(B, 1, 1)

    for i in range(steps):
        Hi = H_all[:, i]
        for j in range(i):
            Hj = H_all[:, j]
            comm = Hi @ Hj - Hj @ Hi
            Omega2 -= 0.5j * comm * dt_sq

    # Total Magnus operator
    Omega = Omega1 + Omega2

    # Exponential
    U = matrix_exp(-1j * Omega)
    return U


# ----------------------------------------------------------
#  First-order Lie–Trotter expansion
# ----------------------------------------------------------
def unitary_trotter_torch(model, time, ops, steps=50):
    """
    Compute U(t) using first-order Lie–Trotter time slicing.

    Args:
        model (nn.Module): Network mapping t → Hamiltonian coefficients.
        time (torch.Tensor): Shape (B,1) times.
        ops (torch.Tensor): Operator basis (K, d, d).
        steps (int): Number of Trotter slices.

    Returns:
        torch.Tensor: Unitaries of shape (B, d, d).
    """
    device = next(model.parameters()).device
    if time.dim() == 1:
        time = time.view(-1, 1)

    time = time.to(device)
    B = time.shape[0]
    K, d, _ = ops.shape

    dt = (time / steps).view(B, 1, 1)

    # Midpoint sampling
    tau = (torch.arange(steps, device=device, dtype=time.dtype) + 0.5) / steps
    t_grid = time.view(B, 1, 1) * tau.view(1, steps, 1)
    t_flat = t_grid.reshape(-1, 1)

    # Evaluate Hamiltonians
    H_all = get_ising_H_batch(model, t_flat, ops).view(B, steps, d, d)

    # Initialize U(t)
    U = torch.eye(d, dtype=torch.cfloat, device=device).expand(B, d, d).clone()

    for k in range(steps):
        Hk = H_all[:, k]
        Uk = matrix_exp(-1j * Hk * dt)
        U = Uk @ U

    return U