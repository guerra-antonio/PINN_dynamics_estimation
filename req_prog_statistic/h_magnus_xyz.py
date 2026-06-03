import numpy as np
from scipy.linalg import expm


# ----------------------------------------------------------
#  Pauli matrices
# ----------------------------------------------------------
def get_pauli_dict():
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return {'I': I, 'X': X, 'Y': Y, 'Z': Z}


def build_kron_op(pauli_list):
    """Kronecker product of a list of single-qubit operators."""
    result = pauli_list[0]
    for op in pauli_list[1:]:
        result = np.kron(result, op)
    return result


def pauli_two_site(pauli1, pos1, pauli2, pos2, n):
    """Two-qubit operator: pauli1 at pos1, pauli2 at pos2 in n-qubit system."""
    paulis = get_pauli_dict()
    I = paulis['I']
    op = [I] * n
    op[pos1] = pauli1
    op[pos2] = pauli2
    return build_kron_op(op)


# ----------------------------------------------------------
#  XYZ Hamiltonian — numpy version (data generation)
# ----------------------------------------------------------
def get_xyz_coeffs(amplitudes, omega, phase, time: float, N: int) -> np.ndarray:
    n_terms = 3 * (N - 1)
    return amplitudes * np.sin(omega * time + phase)


def build_H_xyz(amplitudes, omega, phase, time: float, N: int) -> np.ndarray:
    """
    Build the XYZ Heisenberg Hamiltonian matrix at a given time.

    H(t) = sum_{i=0}^{N-2} [ Jx_i(t) X_i X_{i+1}
                             + Jy_i(t) Y_i Y_{i+1}
                             + Jz_i(t) Z_i Z_{i+1} ]

    Parameters
    ----------
    N : int — number of qubits

    Returns
    -------
    np.ndarray, shape (2^N, 2^N)
    """
    paulis = get_pauli_dict()
    X, Y, Z = paulis['X'], paulis['Y'], paulis['Z']

    coeffs = get_xyz_coeffs(amplitudes, omega, phase, time, N)
    d      = 2 ** N
    H      = np.zeros((d, d), dtype=complex)

    for i in range(N - 1):
        idx_x = 3 * i
        idx_y = 3 * i + 1
        idx_z = 3 * i + 2

        H += coeffs[idx_x] * pauli_two_site(X, i, X, i + 1, N)
        H += coeffs[idx_y] * pauli_two_site(Y, i, Y, i + 1, N)
        H += coeffs[idx_z] * pauli_two_site(Z, i, Z, i + 1, N)

    return H


def xyz_hamiltonian_from_params(params: np.ndarray, N: int) -> np.ndarray:
    """
    Build XYZ Hamiltonian directly from a flat parameter vector.
    Used for 7-8 qubit systems where the model outputs H_eff coefficients.

    Parameters
    ----------
    params : np.ndarray, shape (3*(N-1),)
        Flat vector [Jx_01, Jy_01, Jz_01, Jx_12, ...]
    N : int

    Returns
    -------
    np.ndarray, shape (2^N, 2^N)
    """
    paulis = get_pauli_dict()
    X, Y, Z = paulis['X'], paulis['Y'], paulis['Z']

    assert len(params) == 3 * (N - 1), \
        f"Expected {3*(N-1)} params for {N} qubits, got {len(params)}"

    d = 2 ** N
    H = np.zeros((d, d), dtype=complex)

    for i in range(N - 1):
        H += params[3 * i]     * pauli_two_site(X, i, X, i + 1, N)
        H += params[3 * i + 1] * pauli_two_site(Y, i, Y, i + 1, N)
        H += params[3 * i + 2] * pauli_two_site(Z, i, Z, i + 1, N)

    return H


# ----------------------------------------------------------
#  Unitary evolution — numpy (for data generation)
# ----------------------------------------------------------
def unitary_evolution_xyz(
    amplitudes, omega, phase,
    time: float,
    N: int,
    method: str = "magnus",
    steps: int = 50
) -> np.ndarray:
    """
    Compute U(t) for the XYZ Hamiltonian using Magnus or Trotter.

    Parameters
    ----------
    method : str — 'magnus' or 'trotter'
    steps  : int — number of integration steps

    Returns
    -------
    np.ndarray, shape (2^N, 2^N)
    """

    if time == 0.0:
        return np.eye(2 ** N, dtype=complex)

    dt    = time / steps if time > 0 else 1.0
    times = np.linspace(0, time, steps)
    H_list = [
        build_H_xyz(amplitudes, omega, phase, t, N)
        for t in times
    ]

    if method.lower() == "magnus":
        Omega1 = sum(H_list) * dt
        Omega2 = np.zeros_like(Omega1, dtype=complex)
        for i in range(steps):
            for j in range(i):
                comm    = H_list[i] @ H_list[j] - H_list[j] @ H_list[i]
                Omega2 -= 0.5j * comm * (dt ** 2)
        Omega = Omega1 + Omega2
        return expm(-1j * Omega)

    elif method.lower() == "trotter":
        U = np.eye(2 ** N, dtype=complex)
        for H in H_list:
            U = expm(-1j * H * dt) @ U
        return U

    else:
        raise ValueError("method must be 'magnus' or 'trotter'")


def fid_pros(U1: np.ndarray, U2: np.ndarray) -> float:
    """Gate fidelity between two unitaries."""
    d  = U1.shape[0]
    Tr = np.trace(U1.conj().T @ U2)
    return (np.abs(Tr) / d) ** 2
