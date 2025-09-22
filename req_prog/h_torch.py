import torch
from torch.linalg import matrix_exp

def get_ising_H_batch(model: torch.nn.Module, time_batch: torch.Tensor, n: int = 7) -> torch.Tensor:
    """
    Construct a batch of time-dependent Ising Hamiltonians H(t) for a 1D chain of n qubits,
    given a neural network model that outputs time-dependent coefficients.

    The model must map inputs of shape (B, 1) to outputs of shape (B, 2n - 1), corresponding to:
        - n local transverse field terms (X_i)
        - (n - 1) nearest-neighbor interaction terms (Z_i Z_{i+1})

    Args:
        model (torch.nn.Module): Neural network mapping time t → Hamiltonian coefficients.
        t_batch (torch.Tensor): Tensor of shape (B, 1) with time values.
        n (int): Number of qubits in the chain (default: 7).

    Returns:
        torch.Tensor: Tensor of shape (B, 2^n, 2^n) with one Hamiltonian per batch sample.
    """
    device = time_batch.device
    B = time_batch.shape[0]
    d = 2 ** n
    n_coeffs = 2 * n - 1

    # Predict coefficients (B, 2n - 1)
    params = model(time_batch)
    assert params.shape == (B, n_coeffs), f"Expected output shape (B, {n_coeffs}), got {params.shape}"

    # Define Pauli operators
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat, device=device)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat, device=device)
    I = torch.eye(2, dtype=torch.cfloat, device=device)

    # Build local X_i and ZZ_{i,i+1} terms
    def kron_n(ops):
        out = ops[0]
        for op in ops[1:]:
            out = torch.kron(out, op)
        return out

    X_ops = [kron_n([X if i == j else I for i in range(n)]) for j in range(n)]
    ZZ_ops = [kron_n([Z if i == j or i == j+1 else I for i in range(n)]) for j in range(n - 1)]

    all_ops = X_ops + ZZ_ops  # total of 2n - 1 operators

    # Stack operators and expand to batch
    ops_tensor = torch.stack(all_ops).unsqueeze(0).to(device)  # (1, 2n-1, d, d)
    ops_batch = ops_tensor.expand(B, -1, -1, -1)  # (B, 2n-1, d, d)
    coeffs = params.view(B, n_coeffs, 1, 1)  # (B, 2n-1, 1, 1)

    # Compute weighted sum of operators
    H_batch = (coeffs * ops_batch).sum(dim=1)  # (B, d, d)
    return H_batch

def unitary_magnus_torch(model: torch.nn.Module, time: torch.Tensor, steps: int = 50, n: int = 7) -> torch.Tensor:
    """
    Compute the time-evolution operator U(t) ≈ exp(-i * Ω(t)) using the second-order Magnus expansion.

    The evolution is governed by a time-dependent Hamiltonian:
        H(t) = Σ α_k(t) * P_k
    where α_k(t) are learned coefficients and P_k are local Pauli-based operators.

    Args:
        model (torch.nn.Module): Neural network that maps time t to (2n - 1) Hamiltonian parameters.
        t (torch.Tensor): Tensor of shape (B, 1), containing the target times.
        steps (int): Number of discretization steps for Magnus integration (default: 50).
        n (int): Number of qubits (default: 7).

    Returns:
        torch.Tensor: Tensor of shape (B, 2^n, 2^n) containing the time-evolution unitaries.
    """
    device = time.device
    B = time.shape[0]
    d = 2 ** n
    dt = time / steps  # (B, 1)

    # Create normalized time grid from 0 to t
    t_lin = torch.linspace(0, 1, steps, device=device).view(1, steps, 1)  # (1, steps, 1)
    t_grid = time[:, None, :] * t_lin  # (B, steps, 1)
    t_flat = t_grid.reshape(-1, 1)  # (B * steps, 1)

    # Compute H(t) at each time step
    H_all = get_ising_H_batch(model=model, time_batch=t_flat, n=n).view(B, steps, d, d)

    # First-order Magnus term (integral of H)
    Omega1 = H_all.sum(dim=1) * dt.view(B, 1, 1)

    # Second-order Magnus term (commutator contributions)
    dt_sq = (dt ** 2).view(B, 1, 1)
    Omega2 = torch.zeros(B, d, d, dtype=torch.cfloat, device=device)
    for i in range(steps):
        Hi = H_all[:, i]
        for j in range(i):
            Hj = H_all[:, j]
            comm = Hi @ Hj - Hj @ Hi
            Omega2 -= 0.5j * comm * dt_sq

    # Total Magnus operator
    Omega = Omega1 + Omega2

    # Compute the matrix exponential
    U = matrix_exp(-1j * Omega)
    return U

def unitary_trotter_torch(model: torch.nn.Module, time: torch.Tensor, steps: int = 50, n: int = 7) -> torch.Tensor:
    """
    Compute the time-evolution operator U(t) via first-order (Lie–Trotter) time slicing.

    We approximate the time-ordered exponential by freezing H(t) on r=steps subintervals:
        U(t) ≈ ∏_{k=0}^{r-1} exp(-i * H(t_k) * Δt)
    where Δt = t / r and t_k are sampling points inside each subinterval.
    Here we use mid-point sampling for improved accuracy over left/right endpoints.

    The Hamiltonian is provided as a full matrix H(t) through a batched helper:
        H_all[b, k] = H_b(t_k) ∈ ℂ^{2^n×2^n}

    Args:
        model (torch.nn.Module): Network that parametrizes H(t) (used by get_ising_H_batch).
        time (torch.Tensor): Shape (B, 1) or (B,), target evolution times for each batch item.
        steps (int): Number of time slices r for Trotter time discretization (default: 50).
        n (int): Number of qubits (dimension d = 2^n) (default: 7).

    Returns:
        torch.Tensor: Tensor of shape (B, 2^n, 2^n) with the approximate unitaries U(t).
    """
    # ---- Shapes & basic params ----
    device = time.device
    if time.dim() == 1:
        time = time.view(-1, 1)          # (B,1) for safer broadcasting
    B = time.shape[0]
    d = 2 ** n

    # Δt per sample (B,1,1) → we’ll broadcast to (B, d, d)
    dt = (time / steps).view(B, 1, 1)    # each batch item has its own Δt

    # ---- Time grid (midpoint sampling) ----
    # τ_k in (0,1): (k+0.5)/steps gives midpoints of each subinterval
    tau = (torch.arange(steps, device=device, dtype=time.dtype) + 0.5) / steps   # (steps,)
    # Absolute times t_k per batch: (B, steps, 1)
    t_grid = time.view(B, 1, 1) * tau.view(1, steps, 1)
    # Flatten for batched Hamiltonian evaluation: (B*steps, 1)
    t_flat = t_grid.reshape(-1, 1)

    # ---- Evaluate H(t_k) for all k and all batch items ----
    # Expected shape from helper: (B*steps, d, d)
    H_all = get_ising_H_batch(model=model, time_batch=t_flat, n=n)
    H_all = H_all.view(B, steps, d, d)   # (B, steps, d, d)

    # ---- First-order Lie–Trotter time slicing (ordered product) ----
    # Initialize U as identity per batch: (B, d, d)
    U = torch.eye(d, dtype=H_all.dtype, device=device).expand(B, d, d).clone()

    # Loop over time slices in chronological order
    for k in range(steps):
        # Current slice Hamiltonian for all batch items: (B, d, d)
        Hk = H_all[:, k]
        # Exponential factor for this slice: exp(-i H(t_k) Δt_b)
        # Broadcast dt (B,1,1) against (B,d,d) → (B,d,d)
        Uk = matrix_exp(-1j * Hk * dt)
        # Accumulate ordered product: U <- Uk · U
        U = Uk @ U

    return U
