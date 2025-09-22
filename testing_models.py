import numpy as np
import pandas as pd

import torch
import os

from tqdm import tqdm
from req_prog.h_magnus import fid_pros
from req_prog.h_torch import unitary_magnus_torch, unitary_trotter_torch
from req_prog.architectures import UnitaryModel
from req_prog.useful_functions import reset_weights, closest_unitary


def fidelity_test(model, N_qbits, N_time=100, close_U=False, method = "trotter"):
    """
    Evaluate the fidelity between a model’s predicted unitary and the theoretical unitary
    over a series of time steps.

    Args:
        model:            The neural network or function that predicts the unitary.
        N_qbits (int):    Number of qubits (dimension of the unitary is 2**N_qbits).
        N_time (int):     Number of time points in [0,1] at which to evaluate fidelity.
        close_U (bool):   If True, project the model’s output onto the nearest unitary.

    Returns:
        np.ndarray:       Array of fidelity values of length N_time.
    """
    # 1. Create a 1D array of time points from 0 to 1
    times = np.linspace(0, 1, N_time)
    fidelities = []

    # 2. Load the pre-computed “true” unitaries (via Magnus expansion) for all time steps
    Us_teo = np.load(f"dataframe/unitary_N{N_qbits}_test.npy")

    # 3. Loop over each time step (with a progress bar)
    for idx in tqdm(range(times.shape[0])):
        # 3a. Extract the target unitary for the current time
        U_magnus = Us_teo[idx, :, :]

        # 3b. Predict the unitary from the model
        if N_qbits in [7, 8]:
            # For larger qubit counts, use a special helper that handles batched inputs
            if method == "magnus":
                U_model = unitary_magnus_torch(
                    model=model,
                    time=torch.tensor([[times[idx]]], dtype=torch.float32),
                    n=N_qbits
                )
            elif method == "trotter":
                U_model = unitary_trotter_torch(
                    model=model,
                    time=torch.tensor([[times[idx]]], dtype=torch.float32),
                    n=N_qbits
                )
            else:
                print("method must be magnus or trotter")
                break

        else:
            # For smaller systems, run the model directly on a single time value
            U_model = model(torch.tensor([times[idx]], dtype=torch.float32))

        # 3c. Strip off any batch dimension and move data back to NumPy
        U_model = U_model[0].detach().numpy()

        # 3d. Optionally “snap” the prediction to the nearest valid unitary matrix
        if close_U:
            U_model = closest_unitary(U_model)

        # 3e. Compute fidelity between the predicted and true unitaries
        fid = fid_pros(U_model, U_magnus)
        fidelities.append(fid)

    # 4. Return all fidelities as a NumPy array
    return np.array(fidelities)

# This script tests the fidelity of trained models against the theoretical unitary matrices
Ns      = [n for n in range(2, 9)]
data    = []
close_U = True
method  = "trotter"

for N_qbits in Ns:
    print(f"Testing model with {N_qbits} qubits...")
    # Prepare the models for different delta_t values

    model_01    = UnitaryModel(n_qubits = N_qbits)
    model_025   = UnitaryModel(n_qubits = N_qbits)
    model_05    = UnitaryModel(n_qubits = N_qbits)
    model_10    = UnitaryModel(n_qubits = N_qbits)

    # Reset the weights of the models
    model_01.apply(reset_weights)
    model_025.apply(reset_weights)
    model_05.apply(reset_weights)
    model_10.apply(reset_weights)

    # Load the trained weights
    if N_qbits <= 6:
        model_01.load_state_dict(torch.load(f'models/model{N_qbits}_deltat-0.1.pth', map_location='cpu'))
        model_025.load_state_dict(torch.load(f'models/model{N_qbits}_deltat-0.25.pth', map_location='cpu'))
        model_05.load_state_dict(torch.load(f'models/model{N_qbits}_deltat-0.5.pth', map_location='cpu'))
        model_10.load_state_dict(torch.load(f'models/model{N_qbits}_deltat-1.0.pth', map_location='cpu'))
    else:
        model_01.load_state_dict(torch.load(f'models/model{N_qbits}_deltat-0.1-{method}.pth', map_location='cpu'))
        model_025.load_state_dict(torch.load(f'models/model{N_qbits}_deltat-0.25-{method}.pth', map_location='cpu'))
        model_05.load_state_dict(torch.load(f'models/model{N_qbits}_deltat-0.5-{method}.pth', map_location='cpu'))
        model_10.load_state_dict(torch.load(f'models/model{N_qbits}_deltat-1.0-{method}.pth', map_location='cpu'))

    # Perform the fidelity test
    print(f"Testing fidelity for model{N_qbits} with delta_t=0.1...")
    fid_01 = fidelity_test(model_01, N_qbits, N_time=100, close_U = close_U)

    print(f"Testing fidelity for model{N_qbits} with delta_t=0.25...")
    fid_025 = fidelity_test(model_025, N_qbits, N_time=100, close_U = close_U)

    print(f"Testing fidelity for model{N_qbits} with delta_t=0.5...")
    fid_05 = fidelity_test(model_05, N_qbits, N_time=100, close_U = close_U)

    print(f"Testing fidelity for model{N_qbits} with delta_t=1.0...")
    fid_10 = fidelity_test(model_10, N_qbits, N_time=100, close_U = close_U)

    # Save the results
    data.append({
            "qubits": N_qbits,
            "fid_01": fid_01,
            "fid_025": fid_025,
            "fid_05": fid_05,
            "fid_10": fid_10
        })
    
# Save the data to a file
df = pd.DataFrame(data)
df.to_csv(os.path.join("./dataframe", f"fidelity_test.csv"), index=False)