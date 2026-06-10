from req_prog_statistic.h_magnus import unitary_evolution, fid_pros, unitary_trotter_78
from req_prog_statistic.h_torch import unitary_trotter_torch, unitary_magnus_torch
from req_prog_statistic.useful_functions import random_state, closest_unitary
from req_prog_statistic.training import train_model
from req_prog_statistic.architectures import UnitaryModel
from tqdm import tqdm

import numpy as np
import torch

# -----------------------------------------------------------
# Time grids used for training and testing
# -----------------------------------------------------------
time_test = np.linspace(0, 1, 101) # 101 time points (testing)

# -----------------------------------------------------------
# Generates a family of time-dependent unitary operators U(t)
# using random coefficients for a parameterized Hamiltonian.
# The time evolution is computed through the Magnus expansion
# (implemented in the external function unitary_magnus).
# -----------------------------------------------------------
def random_U(N=2, time_grid=10, method = "trotter"):
    if N <= 6:
        n_coeff = 4 ** N  # number of coefficients in the Pauli basis for N qubits
    else:
        n_coeff = 2 * N - 1  # simplified model for N = 7, 8 qubits

    # --- Random Hamiltonian parameters ---
    amplitude = np.random.rand(n_coeff)             # amplitudes for Pauli terms
    omega     = 2 * np.pi * np.random.rand(n_coeff) # angular frequencies
    phase     = 2 * np.pi * np.random.rand(n_coeff) # phase shifts

    Us = []  # list to store the generated unitaries U(t)

    # Compute U(t) for each time step in the global variable `time`
    time = np.linspace(0, 1, time_grid + 1)       # 11 time points (training)
    for t in tqdm(time, desc="Generating unitaries for training"):
        if N <= 6:
            U = unitary_evolution(
                method=method,
                amplitudes=amplitude,
                omega=omega,
                phase=phase,
                time=t,
                N=N,
                steps=10  # number of Trotter steps in the Magnus expansion
            )
            Us.append(U)
        else:
            U = unitary_trotter_78(
                time=t,
                amplitudes=amplitude,
                omega=omega,
                phase=phase,
                steps=50,  # number of Trotter steps
                n_qubits=N
            )
            Us.append(U)

    # Return the family of unitaries and their defining parameters
    Us = np.array(Us)
    return Us, np.array([amplitude, omega, phase]), time

# -----------------------------------------------------------
# Reconstructs the same family of unitaries U(t) using
# previously saved Hamiltonian coefficients (from random_U).
# Useful for validation or testing with a denser time grid.
# -----------------------------------------------------------
def U_from_coeffs(coeffs, N=2, method = "trotter"):
    amplitude, omega, phase = coeffs

    Us = []
    for t in tqdm(time_test, desc="Generating unitaries for testing"):
        if N <= 6:
            U = unitary_evolution(
                method=method,
                amplitudes=amplitude,
                omega=omega,
                phase=phase,
                time=t,
                N=N,
                steps=10
            )
            Us.append(U)
        else:
            U = unitary_trotter_78(
                time=t,
                amplitudes=amplitude,
                omega=omega,
                phase=phase,
                steps=10,
                n_qubits=N
            )
            Us.append(U)

    return np.array(Us)

# -----------------------------------------------------------
# Generates a pair (ρ₀, ρₜ) by evolving a random initial state
# under a given unitary operator U.
# -----------------------------------------------------------
def gen_tupla(U, N=2):
    rho_0 = random_state(n_qubits=N)   # random initial density matrix
    rho_t = U @ rho_0 @ U.conj().T     # evolved state: ρₜ = U ρ₀ U†
    return rho_0, rho_t

# -----------------------------------------------------------
# Generates a dataset of triplets (ρ₀, t, ρₜ) for training or validation.
# Each time step contributes multiple random state evolutions.
# -----------------------------------------------------------
def gen_data(Us, time, N=2, n_data=1000):
    data = []
    for U_ in tqdm(range(Us.shape[0]), desc="Running data gen"):          # iterate over each time step
        for _ in range(n_data):            # generate multiple samples per time
            rho_0, rho_t = gen_tupla(Us[U_], N=N)
            t = time[U_] * np.ones_like(rho_0)  # encode time as constant matrix
            data.append(np.array([rho_0, t, rho_t], dtype=np.complex64))
    return np.array(data)

# -----------------------------------------------------------
# Computes the fidelity between predicted and true unitaries.
# Optionally projects predicted matrices onto the closest
# unitary group element before comparison.
# -----------------------------------------------------------
def fidelity_test(model, U_test, ops, close_U=False, model_H = "default", device="cpu"):
    times = torch.tensor(time_test, dtype=torch.float32, device=device).view(-1, 1)
    model   = model.to(device)
    ops     = ops.to(device)
    fidelities = []

    # Predict U(t) from the trained model
    if model_H == "trotter":
        U_model = unitary_trotter_torch(model, times, ops=ops)
        U_model = U_model.cpu().detach().numpy()
    if model_H == "magnus":
        U_model = unitary_magnus_torch(model, times, ops=ops)
        U_model = U_model.cpu().detach().numpy()
    else:
        U_model = model(times).detach().numpy()

    # Evaluate fidelity at each time step
    for i in tqdm(range(times.shape[0]), desc="Testing model"):
        if close_U:
            U_pred = closest_unitary(U_model[i])  # ensure unitarity
            fid = fid_pros(U_pred, U_test[i])
        else:
            fid = fid_pros(U_model[i], U_test[i])
        fidelities.append(fid)

    return np.array(fidelities)

# -----------------------------------------------------------
# Complete testing pipeline:
# 1. Generates a random unitary evolution U(t)
# 2. Builds training data (ρ₀, t, ρₜ)
# 3. Initializes and trains the model
# 4. Reconstructs U(t) over a fine grid
# 5. Computes fidelity between predicted and true unitaries
# -----------------------------------------------------------
def test_model(ops, N=2, device="cpu", method = "trotter", model_H = "default"):
    # Step 1: generate unitaries and corresponding training data
    data_U, coeffs, time = random_U(N=N, method=method)
    data = gen_data(Us=data_U, time=time, N=N)

    # Step 2: initialize the model
    model = UnitaryModel(n_qubits=N)

    # Step 3: train the model
    model = train_model(model=model, 
                        data=data, 
                        data_U=data_U, 
                        device=device, 
                        model_H=model_H,
                        batch_epoch=10,
                        num_epochs=1000,
                        sch=300
                        )

    # Step 4: reconstruct unitaries for testing
    data_U = U_from_coeffs(coeffs=coeffs, N=N, method=method)

    # Step 5: compute fidelity
    F_false = fidelity_test(ops=ops, model=model, U_test=data_U, close_U=False, model_H=model_H)
    F_true = fidelity_test(ops=ops, model=model, U_test=data_U, close_U=True, model_H=model_H)
    
    return F_false, F_true, coeffs

# -----------------------------------------------------------
# Complete testing pipeline:
# 1. Generates a random unitary evolution U(t)
# 2. Builds training data (ρ₀, t, ρₜ)
# 3. Initializes and trains the model with and without the unitarity term in the loss function
# 4. Reconstructs U(t) over a fine grid
# 5. Computes fidelity between predicted and true unitaries
# -----------------------------------------------------------

# -----------------------------------------------------------
# Resets all learnable parameters of a PyTorch model to their
# initial random state (useful before retraining).
# -----------------------------------------------------------
def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def test_model_unitarity(ops, N=2, time_grid=10, device="cpu", method = "trotter", model_H = "default"):
    # Step 1: generate unitaries and corresponding training data
    data_U_train, coeffs, time  = random_U(N=N, method=method, time_grid=time_grid)
    data_U_test                 = U_from_coeffs(coeffs=coeffs, N=N, method=method)
    data                        = gen_data(Us=data_U_train, time=time, N=N)

    # Step 2: initialize the model
    model = UnitaryModel(n_qubits=N)

    # Step 3: train the model
    model   = train_model(model=model, 
                        data=data, 
                        data_U=data_U_train, 
                        device=device, 
                        model_H=model_H,
                        batch_epoch=100,
                        batch_size=10,
                        learning_rate=1e-3,
                        num_epochs=400,
                        sch=100,
                        unitarity=True,
                        time_grid=time_grid
                        )
    F_true = fidelity_test(ops=ops, model=model, U_test=data_U_test, close_U=False, model_H=model_H)

    model   = train_model(model=model, 
                        data=data, 
                        data_U=data_U_train, 
                        device=device, 
                        model_H=model_H,
                        batch_epoch=100,
                        batch_size=10,
                        learning_rate=1e-3,
                        num_epochs=400,
                        sch=100,
                        unitarity=False,
                        time_grid=time_grid
                        )
    F_false = fidelity_test(ops=ops, model=model, U_test=data_U_test, close_U=False, model_H=model_H)

    # # Step 4: reconstruct unitaries for testing
    # data_U = U_from_coeffs(coeffs=coeffs, N=N, method=method)

    # # Step 5: compute fidelity
    # F_true = fidelity_test(ops=ops, model=model_with, U_test=data_U, close_U=False, model_H=model_H)
    # F_false = fidelity_test(ops=ops, model=model_without, U_test=data_U, close_U=False, model_H=model_H)
    
    return F_false, F_true, coeffs