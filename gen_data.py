# This program generates the required data for the training stage of the models. Different parameters can be choosen here like the rank of the density matrices used for training, the max_time of the time domain of the unitary time-evolution, or the rank of the generated density matrices.
import os
import glob
import numpy as np
from req_prog.useful_functions import random_state
from tqdm import tqdm
from IPython.display import clear_output

# Define the parameters of the data generation
rank        = None      # None means that the rank gonna be choosen randomly between 1 and 2**N (with N the number of qubits)
N_data      = 11000     # 11000 density matrices are gonna be generated for the training
N_test      = 1000      # 1000 density matrices are gonna be generated for the validation

# Define some useful functions to gen data
def gen_tupla(N=2, Unitary=np.ndarray, rank=None):
    """
    Generates a tuple of input and output quantum states (density matrices) related by a unitary transformation.

    Parameters
    ----------
    N : int, optional
        Number of qubits. Determines the dimension of the Hilbert space (default is 2).
    Unitary : np.ndarray
        A unitary matrix representing the quantum evolution to apply to the input state.
    rank : int or None, optional
        Rank of the initial density matrix. If None, a full-rank random state is generated.

    Returns
    -------
    rho_in : np.ndarray
        The randomly generated input density matrix (quantum state).
    rho_out : np.ndarray
        The output density matrix after applying the unitary evolution: U ρ U†.
    """
    rho_in  = random_state(n_qubits=N, rank=rank)

    rho_out = Unitary @ rho_in @ Unitary.conj().T

    return rho_in, rho_out


def generation_npy_train(N=2, N_data=11000, rank=None, delta_t=0.1):
    """
    Generates a dataset of input-output quantum states evolved under a unitary operator at specific time steps.
    The dataset is saved in a compressed .npz file.

    Parameters
    ----------
    N : int, optional
        Number of qubits in the system (default is 2).
    N_data : int, optional
        Total number of data samples to generate (default is 11000).
    rank : int or None, optional
        Rank of the input density matrix. If None, a full-rank random state is generated.
        (Note: rank is passed to `gen_tupla` but currently not used directly in this function.)
    delta_t : float, optional
        Time step resolution. Must be one of [0.1, 0.25, 0.5, 1.0]. Determines which time indices are used.

    Raises
    ------
    ValueError
        If `delta_t` is not one of the supported values.

    Behavior
    --------
    - Loads a precomputed array of unitaries from `unitary_N{N}.npy`.
    - For each selected time index (based on `delta_t`), generates `n_data_per_time` samples.
    - Each sample consists of:
        - `rho_in`: the input density matrix,
        - `time_matrix`: a matrix filled with the corresponding time value,
        - `rho_out`: the evolved density matrix.
    - Concatenates all samples into a single NumPy array of shape (N_data, 3, 2^N, 2^N).
    - Saves the resulting array in a compressed `.npz` file named according to `N` and `delta_t`.
    """
    if delta_t == 0.1:
        time_index = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12]
    elif delta_t == 0.25:
        time_index = [0, 3, 6, 9, 12]
    elif delta_t == 0.5:
        time_index = [0, 6, 12]
    elif delta_t == 1.0:
        time_index = [0, 12]
    else:
        print("delta_t must be 0.1, 0.25, 0.5 oe 1.0")

    Us      = np.load(f"dataframe/unitary_N{N}.npy")
    times   = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]

    n_data_per_time = N_data // len(time_index)

    files = sorted(glob.glob("dataframe/temp*.npy"))
    for f in files:
        os.remove(f)

    for index in time_index:
        U = Us[index, :, :]
        clear_output(True)

        for _ in tqdm(range(n_data_per_time), desc=f"Generating data for N={N} with delta_t={delta_t} and index {index}"):
            rho_in, rho_out = gen_tupla(N = N, Unitary = U)   
            time_matrix = times[index] * np.ones_like(rho_in, dtype = np.float64)

            if _ == 0:
                dataframe = np.expand_dims(np.array([rho_in, time_matrix, rho_out]), axis = 0)
            else:
                dataframe = np.concatenate((dataframe, np.expand_dims(np.array([rho_in, time_matrix, rho_out]), axis = 0)), axis = 0)

        np.save(file = f"dataframe/temp{index}.npy", arr = dataframe)

    files = sorted(glob.glob("dataframe/temp*.npy"))
    arrays = [np.load(f) for f in files]
    dataframe = np.concatenate(arrays, axis=0)

    np.savez_compressed(os.path.join("./dataframe", f"data_model{N}_deltat-{delta_t}"), dataframe)


# Generated the data for the training
Ns          = [n for n in range(2, 9)]
delta_ts    = [0.1, 0.25, 0.5, 1.0]

for N in Ns:
    for dt in delta_ts:
        generation_npy_train(N = N, delta_t = dt, N_data = N_data)