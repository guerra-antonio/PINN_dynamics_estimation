import numpy as np
from scipy.linalg import expm
from itertools import product

# Load parameters for H(t) — MUST EXIST LOCALLY
amplitudes = np.load("req_prog/amplitude.npy")
omega = 4 * np.load("req_prog/frequencies.npy")
phase = np.load("req_prog/phases.npy")


def get_time_dependent_coeffs(time: float, N: int = 2):
    """
    Generate time-dependent coefficients for a Pauli basis Hamiltonian.

    Args:
        time (float): Time of evaluation.
        N (int): Number of qubits.

    Returns:
        np.ndarray: Real coefficients for each Pauli string (length 4^N).
    """
    n_terms = 4 ** N

    # Normalize based on N to match dataset scaling
    if N == 2:
        H = amplitudes * np.sin(omega * time + phase) / 2
    elif N == 3:
        H = amplitudes * np.sin(omega * time + phase) / 3
    elif N == 4:
        H = amplitudes * np.sin(omega * time + phase) / 4
    elif N == 5:
        H = amplitudes * np.sin(omega * time + phase) / 16
    elif N == 6:
        H = amplitudes * np.sin(omega * time + phase) / 24
    else:
        raise ValueError("N must be 2, 3, 4, 5 or 6")

    H[0] = 0  # Remove global identity term
    return H[:n_terms]


def get_pauli_dict():
    """Return dictionary of basic Pauli matrices."""
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    return {'I': I, 'X': X, 'Y': Y, 'Z': Z}


def generate_pauli_strings(N: int):
    """
    Generate the list of Pauli string labels (e.g., 'IXYZ') for N qubits.

    Args:
        N (int): Number of qubits.

    Returns:
        List[Tuple[str]]: List of Pauli label tuples.
    """
    return list(product(['I', 'X', 'Y', 'Z'], repeat=N))


def build_H_from_paulis(time: float, N: int = 4):
    """
    Construct H(t) using Pauli string basis for given number of qubits.

    Args:
        time (float): Evaluation time.
        N (int): Number of qubits.

    Returns:
        np.ndarray: Complex-valued Hamiltonian matrix of size 2^N x 2^N.
    """
    dim = 2 ** N
    coeffs = get_time_dependent_coeffs(time, N)
    paulis = get_pauli_dict()
    pauli_terms = generate_pauli_strings(N)

    H = np.zeros((dim, dim), dtype=np.complex128)

    for c, term in zip(coeffs, pauli_terms):
        P = paulis[term[0]]
        for p in term[1:]:
            P = np.kron(P, paulis[p])
        H += c * P

    return H


def unitary_magnus(time: float, N: int = 4, H_func=build_H_from_paulis, steps: int = 100):
    """
    Compute U(t) ≈ exp(-i Ω(t)) using Magnus expansion up to 3rd order.

    Args:
        time (float): Total evolution time.
        N (int): Number of qubits.
        H_func (function): Function that returns H(t) given (time, N).
        steps (int): Integration steps.

    Returns:
        np.ndarray: Unitary evolution matrix.
    """
    dt = time / steps
    times = np.linspace(0, time, steps)
    H_list = [H_func(ti, N) for ti in times]

    # Omega1
    Omega1 = sum(H_list) * dt

    # Omega2
    Omega2 = np.zeros_like(Omega1, dtype=np.complex64)
    for i in range(steps):
        for j in range(i):
            comm = H_list[i] @ H_list[j] - H_list[j] @ H_list[i]
            Omega2 -= 0.5j * comm * dt ** 2

    # Omega3
    Omega3 = np.zeros_like(Omega1, dtype=np.complex64)
    for i in range(steps):
        for j in range(i):
            for k in range(j):
                H1, H2, H3 = H_list[i], H_list[j], H_list[k]
                term1 = H1 @ (H2 @ H3 - H3 @ H2) - (H2 @ H3 - H3 @ H2) @ H1
                term2 = H3 @ (H2 @ H1 - H1 @ H2) - (H2 @ H1 - H1 @ H2) @ H3
                Omega3 -= (term1 + term2) * dt ** 3 / 6

    Omega = Omega1 + Omega2 + Omega3
    return expm(-1j * Omega)


def fid_pros(U1: np.ndarray, U2: np.ndarray) -> float:
    """
    Compute process fidelity between two unitary matrices.

    Args:
        U1 (np.ndarray): First unitary.
        U2 (np.ndarray): Second unitary.

    Returns:
        float: Fidelity value between 0 and 1.
    """
    d = U1.shape[0]
    Tr = np.trace(U1.conj().T @ U2)
    return (np.abs(Tr) / d) ** 2


# ------------------------------------------------------------------------
# Functions specific to 7-qubit Ising system with predefined structure
# ------------------------------------------------------------------------

def build_kron_op(pauli_list):
    """Kronecker product of a list of single-qubit operators."""
    result = pauli_list[0]
    for op in pauli_list[1:]:
        result = np.kron(result, op)
    return result


def pauli_single_site(pauli, pos, n):
    """Single Pauli operator at position `pos` in `n`-qubit system."""
    op = [np.eye(2)] * n
    op[pos] = pauli
    return build_kron_op(op)


def pauli_two_site(pauli1, pos1, pauli2, pos2, n):
    """Two-qubit operator: pauli1 at pos1 and pauli2 at pos2 in `n`-qubit system."""
    op = [np.eye(2)] * n
    op[pos1] = pauli1
    op[pos2] = pauli2
    return build_kron_op(op)

def ising_hamiltonian(params: np.ndarray, n: int) -> np.ndarray:
    """
    Builds an n-qubit Ising Hamiltonian with local X fields and ZZ interactions.
    Args:
        params (np.ndarray): 2n - 1 parameters [h0..hn-1, J0..Jn-2]
        n (int): Number of qubits
    Returns:
        np.ndarray: Hermitian matrix of shape (2^n, 2^n)
    """
    assert len(params) == 2 * n - 1, f"Expected {2 * n - 1} parameters for {n} qubits."

    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    H = np.zeros((2**n, 2**n), dtype=complex)

    for i in range(n):
        H += params[i] * pauli_single_site(X, i, n)
    for i in range(n - 1):
        H += params[n + i] * pauli_two_site(Z, i, Z, i + 1, n)

    return H

def unitary_magnus_78(time: float, steps: int = 100, n_qubits: int = 7) -> np.ndarray:
    """
    Compute U(t) ≈ exp(-i Ω(t)) for an n-qubit Ising model using 2nd-order Magnus expansion.

    Args:
        time (float): Total evolution time
        steps (int): Integration steps
        n_qubits (int): Number of qubits (supports 2 ≤ n ≤ 8)

    Returns:
        np.ndarray: Unitary matrix of shape (2^n, 2^n)
    """
    dt = time / steps
    times = np.linspace(0, time, steps)

    # Get time-dependent coefficients (amplitudes, phases and omega must be preloaded)
    n_params = 2 * n_qubits - 1
    H_list = []

    for t in times:
        coeffs = 2 * get_time_dependent_coeffs(t, N=2).astype(np.float32)
        coeffs = coeffs[1 : 1 + n_params]  # Remove first term, take correct slice
        H = ising_hamiltonian(coeffs, n_qubits)
        H_list.append(H)

    # Magnus expansion: Omega = Omega1 + Omega2
    Omega1 = sum(H_list) * dt

    Omega2 = np.zeros_like(Omega1, dtype=np.complex128)
    for i in range(steps):
        for j in range(i):
            comm = H_list[i] @ H_list[j] - H_list[j] @ H_list[i]
            Omega2 -= 0.5j * comm * dt**2

    Omega = Omega1 + Omega2
    return expm(-1j * Omega)