import torch
from torch.linalg import matrix_exp


# ----------------------------------------------------------
#  Build fixed XYZ operator basis (only computed once)
# ----------------------------------------------------------
def build_ops_xyz(n: int, device: str = "cpu") -> torch.Tensor:
    """
    Build the fixed operator basis for the XYZ Heisenberg Hamiltonian.

    Operators: [XX_01, YY_01, ZZ_01, XX_12, YY_12, ZZ_12, ...]
    Total: 3*(n-1) operators, each of shape (2^n, 2^n).

    Parameters
    ----------
    n      : int — number of qubits
    device : str

    Returns
    -------
    torch.Tensor, shape (3*(n-1), 2^n, 2^n)
    """
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cfloat)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat)
    I = torch.eye(2, dtype=torch.cfloat)

    def kron_n(op_list):
        out = op_list[0]
        for op in op_list[1:]:
            out = torch.kron(out, op)
        return out

    ops = []
    for i in range(n - 1):
        for P in (X, Y, Z):
            op_list = [I] * n
            op_list[i]     = P
            op_list[i + 1] = P
            ops.append(kron_n(op_list))

    return torch.stack(ops).to(device)   # (3*(n-1), 2^n, 2^n)


# ----------------------------------------------------------
#  Build XYZ Hamiltonians efficiently (batched)
# ----------------------------------------------------------
def get_xyz_H_batch(
    model: torch.nn.Module,
    time_batch: torch.Tensor,
    ops: torch.Tensor
) -> torch.Tensor:
    """
    Compute a batch of XYZ Hamiltonians H(t) from model coefficients.

    Parameters
    ----------
    model      : maps (B, 1) → (B, 3*(n-1)) coefficients
    time_batch : (B, 1)
    ops        : (3*(n-1), d, d)

    Returns
    -------
    torch.Tensor, shape (B, d, d)
    """
    coeffs = model(time_batch)                            # (B, K)
    coeffs = coeffs.to(dtype=ops.dtype, device=ops.device)
    H      = torch.einsum("bk,kij->bij", coeffs, ops)    # (B, d, d)
    return H


# ----------------------------------------------------------
#  Second-order Magnus expansion — XYZ
# ----------------------------------------------------------
def unitary_magnus_xyz(
    model: torch.nn.Module,
    time: torch.Tensor,
    ops: torch.Tensor,
    steps: int = 20
) -> torch.Tensor:
    """
    Compute U(t) using 2nd-order Magnus expansion for the XYZ Hamiltonian.

    Parameters
    ----------
    model : predicts XYZ coefficients (B, 3*(n-1))
    time  : (B, 1)
    ops   : (3*(n-1), d, d)
    steps : int — integration steps

    Returns
    -------
    torch.Tensor, shape (B, d, d)
    """
    device = time.device
    B      = time.shape[0]
    K, d, _ = ops.shape
    dt     = time / steps    # (B, 1)

    # Time grid
    t_lin  = torch.linspace(0, 1, steps, device=device).view(1, steps, 1)
    t_grid = time[:, None, :] * t_lin       # (B, steps, 1)
    t_flat = t_grid.reshape(-1, 1)          # (B*steps, 1)

    # Evaluate all H(t_k)
    H_all  = get_xyz_H_batch(model, t_flat, ops).view(B, steps, d, d)

    # First-order Magnus
    Omega1 = H_all.sum(dim=1) * dt.view(B, 1, 1)

    # Second-order Magnus
    Omega2 = torch.zeros(B, d, d, dtype=torch.cfloat, device=device)
    dt_sq  = (dt ** 2).view(B, 1, 1)

    for i in range(steps):
        Hi = H_all[:, i]
        for j in range(i):
            Hj   = H_all[:, j]
            comm = Hi @ Hj - Hj @ Hi
            Omega2 -= 0.5j * comm * dt_sq

    Omega = Omega1 + Omega2
    return matrix_exp(-1j * Omega)


# ----------------------------------------------------------
#  First-order Lie–Trotter — XYZ
# ----------------------------------------------------------
def unitary_trotter_xyz(
    model: torch.nn.Module,
    time: torch.Tensor,
    ops: torch.Tensor,
    steps: int = 20
) -> torch.Tensor:
    """
    Compute U(t) using first-order Trotter slicing for the XYZ Hamiltonian.

    Parameters
    ----------
    model : predicts XYZ coefficients (B, 3*(n-1))
    time  : (B, 1) or (B,)
    ops   : (3*(n-1), d, d)
    steps : int

    Returns
    -------
    torch.Tensor, shape (B, d, d)
    """
    device = next(model.parameters()).device
    if time.dim() == 1:
        time = time.view(-1, 1)
    time = time.to(device)

    B       = time.shape[0]
    K, d, _ = ops.shape
    dt      = (time / steps).view(B, 1, 1)

    # Midpoint sampling
    tau    = (torch.arange(steps, device=device, dtype=time.dtype) + 0.5) / steps
    t_grid = time.view(B, 1, 1) * tau.view(1, steps, 1)
    t_flat = t_grid.reshape(-1, 1)

    H_all = get_xyz_H_batch(model, t_flat, ops).view(B, steps, d, d)

    U = torch.eye(d, dtype=torch.cfloat, device=device).expand(B, d, d).clone()
    for k in range(steps):
        Hk = H_all[:, k]
        Uk = matrix_exp(-1j * Hk * dt)
        U  = Uk @ U

    # print(steps)
    return U
