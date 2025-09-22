# In order to reduce computational costs, before to run everything we compute the required Unitary matrices for the time-evolution operator for all known times
import numpy as np

from tqdm import tqdm
from req_prog.h_magnus import unitary_magnus, unitary_magnus_78

# Define the function that compute the unitaries for a set of times
def gen_unitary(N = 2, times = None, test = False):
    """
    Generates and saves a sequence of unitary matrices for a system of N qubits,
    evaluated at the provided time steps.

    Parameters
    ----------
    N : int
        Number of qubits. Must be between 2 and 8 inclusive.
    times : list or array-like of float
        Time values at which to compute the unitary matrices.
    test : bool, optional
        If True, saves the array as 'unitary_N{N}.npy'.
        If False, saves as 'unitary_N{N}_test.npy'.

    Raises
    ------
    ValueError
        If N is not between 2 and 8, or if `times` is not provided or invalid.
    """

    # --- Validations ---
    if N not in range(2, 9):
        raise ValueError("N must be an integer between 2 and 8 inclusive.")
    
    if times is None or not hasattr(times, '__iter__') or len(times) == 0:
        raise ValueError("You must provide a non-empty iterable of time values.")

    Us_list = []

    for time in tqdm(times, desc=f"Calculating unitaries for N = {N}"):
        if N in range(2, 7):
            U = unitary_magnus(N=N, time=time)
        elif N in [7, 8]:
            U = unitary_magnus_78(time=time, n_qubits=N)

        Us_list.append(np.expand_dims(U, axis=0))

    Us = np.concatenate(Us_list, axis=0)

    if test == True:
        filename = f"dataframe/unitary_N{N}_test.npy"
    else:
        filename = f"dataframe/unitary_N{N}.npy"

    np.save(file=filename, arr=Us)

# Define the times that we know
times_train     = np.array([0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0])
times_test      = np.linspace(0, 1, 100)

# Compute unitaries required for generate training dataset's and testing the models (not for generate the validation dataset)
for N in range(2, 9):
    gen_unitary(N = N, times = times_train)
    gen_unitary(N = N, times = times_test, test = True)