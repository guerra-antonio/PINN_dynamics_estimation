"""
statistic_test_xyz.py

Same statistical testing pipeline as statistic_test.py, adapted for the
XYZ Heisenberg Hamiltonian with nearest-neighbor couplings.

Hamiltonian:
    H(t) = sum_{i} [ Jx_i(t) X_i X_{i+1}
                   + Jy_i(t) Y_i Y_{i+1}
                   + Jz_i(t) Z_i Z_{i+1} ]
with Ja_i(t) = A_a * sin(omega_a * t + phi_a), all independently random.

Systems 2-6 qubits : model predicts U(t) directly (mode='matrix').
Systems 7-8 qubits : model predicts 3*(N-1) H_eff coefficients (mode='coeffs'),
                     U(t) computed via Trotter with 10 steps.
"""

import numpy as np
import torch
import random

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

# Local imports — place this file alongside req_prog_statistic or adjust paths
from req_prog_statistic.h_magnus_xyz import (
    unitary_evolution_xyz,
    fid_pros,
    get_xyz_coeffs,
    xyz_hamiltonian_from_params,
)
from req_prog_statistic.h_torch_xyz import (
    build_ops_xyz,
    unitary_trotter_xyz,
    unitary_magnus_xyz,
)
from req_prog_statistic.useful_functions import random_state, closest_unitary
from req_prog_statistic.architectures import UnitaryModel


# ----------------------------------------------------------
#  Architecture override for large XYZ systems
# ----------------------------------------------------------
def make_model_xyz(n_qubits: int) -> torch.nn.Module:
    """
    For n_qubits <= 6 : use standard UnitaryModel (matrix mode).
    For n_qubits >= 7 : override output_dim to 3*(N-1) coefficients.
    """
    if n_qubits <= 6:
        return UnitaryModel(n_qubits=n_qubits)
    else:
        # Build a small coefficient-prediction network
        import torch.nn as nn
        output_dim = 3 * (n_qubits - 1)
        model = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, output_dim)
        )
        model.n_qubits = n_qubits
        model.mode     = 'coeffs'
        return model


# ----------------------------------------------------------
#  Time grids
# ----------------------------------------------------------
# time      = np.linspace(0, 1, 11)    # training grid
time_test = np.linspace(0, 1, 101)   # testing grid

time_grid = {
    0.1 : np.linspace(0, 1, 11),
    0.25 : np.linspace(0, 1, 5),
    0.5 : np.linspace(0, 1, 3),
    1.0 : np.linspace(0, 1, 2),
}
time_index = {
    0.1 : list(set(list(range(0, 13, 1))) - set([3, 9])),
    0.25 : list(range(0, 13, 3)),
    0.5 : list(range(0, 13, 6)),
    1.0 : list(range(0, 13, 12)),
}
time_all = np.concatenate((time_grid[0.1], time_grid[0.25]))
time_all = np.unique(time_all)
time_all.sort()

# ----------------------------------------------------------
#  Data generation
# ----------------------------------------------------------
scale_by_N = {2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 0.5, 7: 3.0, 8: 3.0}

def random_U_xyz(N: int = 2, method: str = "trotter", time_grid=time_grid[0.1]):
    n_coeffs  = 3 * (N - 1)
    scale     = scale_by_N.get(N, 1.0)
    amplitude = np.random.uniform(0.5, 1.5, n_coeffs) / scale
    omega     = 2 * np.pi * np.random.rand(n_coeffs)
    phase     = 2 * np.pi * np.random.rand(n_coeffs)

    Us = []
    for t in tqdm(time_grid, desc=f"Generating XYZ unitaries N={N}"):
    # for t in time_grid:
        U = unitary_evolution_xyz(
            amplitudes=amplitude,
            omega=omega,
            phase=phase,
            time=t,
            N=N,
            method=method,
            steps=25
        )
        Us.append(U)

    return np.array(Us), np.array([amplitude, omega, phase])


def U_from_coeffs_xyz(coeffs, N: int = 2, method: str = "trotter"):
    """
    Reconstruct U(t) on the test grid from saved Hamiltonian coefficients.
    """
    amplitude, omega, phase = coeffs
    Us = []
    for t in tqdm(time_test, desc=f"Generating XYZ test unitaries N={N}"):
        U = unitary_evolution_xyz(
            amplitudes=amplitude,
            omega=omega,
            phase=phase,
            time=t,
            N=N,
            method=method,
            steps=15
        )
        Us.append(U)
    return np.array(Us)


def gen_tupla_xyz(U, N: int = 2):
    rho_0 = random_state(n_qubits=N)
    rho_t = U @ rho_0 @ U.conj().T
    return rho_0, rho_t


def gen_data_xyz(Us, N: int = 2, n_data: int = 1000, time=time_grid[0.1]):
    """
    Generate dataset of (rho_0, t, rho_t) triples.
    """
    data = []
    for U_idx in tqdm(range(Us.shape[0]), desc="Generating dataset"):
        for _ in range(n_data):
            rho_0, rho_t = gen_tupla_xyz(Us[U_idx], N=N)
            t = time[U_idx] * np.ones_like(rho_0)
            data.append(np.array([rho_0, t, rho_t], dtype=np.complex64))
    return np.array(data)


# ----------------------------------------------------------
#  Dataset class
# ----------------------------------------------------------
class XYZDataset(Dataset):
    def __init__(self, data_array):
        if isinstance(data_array, np.ndarray):
            self.data = torch.tensor(data_array, dtype=torch.complex64)
        else:
            self.data = data_array

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample  = self.data[idx]
        rho_in  = sample[0]
        t_mat   = sample[1]
        rho_out = sample[2]
        t_scalar = t_mat.real[0, 0].unsqueeze(0).to(torch.float32)
        return (rho_in, t_scalar), rho_out


# ----------------------------------------------------------
#  Training
# ----------------------------------------------------------
def train_model_xyz(
    model,
    data,
    data_U,
    ops,
    N_qubits: int,
    num_epochs: int = 400,
    batch_size: int = 100,
    batch_epoch: int = 10,
    learning_rate: float = 1e-3,
    N_domain: int = 100,
    sch: int = 100,
    device: str = "cpu",
    method: str = "trotter",   # integration method for 7-8 qubit models,
    loss_variation: bool = False
):
    """
    Train the model for the XYZ Hamiltonian.

    For N <= 6: model predicts U(t) directly, loss uses direct model output.
    For N >= 7: model predicts H_eff coefficients, U(t) computed via Trotter/Magnus.
    """
    U_teo = torch.tensor(data_U, dtype=torch.complex64, device=device)
    model = model.to(device)
    ops   = ops.to(device)

    # Dataset and dataloaders
    n_dataload = max(1, int(data.shape[0] / batch_size))
    dataset    = XYZDataset(data)
    index      = np.arange(data.shape[0])
    np.random.shuffle(index)
    split_index = np.array_split(index, n_dataload)
    dataloaders = [
        DataLoader(dataset, batch_size=batch_size,
                   sampler=SubsetRandomSampler(subset))
        for subset in split_index
    ]

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch)

    d       = 2 ** N_qubits
    I_batch = torch.eye(d, dtype=torch.complex64, device=device)\
                   .unsqueeze(0).repeat(N_domain, 1, 1)

    t_domain = torch.linspace(0, 1, N_domain, device=device).reshape(-1, 1)
    t_u      = torch.linspace(0, 1, U_teo.shape[0], device=device).reshape(-1, 1)

    is_large = N_qubits >= 7
    loss_values  = []
    epoch_losses = []  # acumula loss por batch dentro de la época

    for epoch in tqdm(range(num_epochs), desc="Training XYZ model"):
        model.train()
        epoch_losses = []  # reset al inicio de cada época

        selected_dls = random.sample(dataloaders, min(batch_epoch, len(dataloaders)))

        for dl in selected_dls:
            for (rho_in, t_scalar), rho_out in dl:
                rho_in   = rho_in.to(device)
                t_scalar = t_scalar.to(device)
                rho_out  = rho_out.to(device)

                optimizer.zero_grad()

                if is_large:
                    fn = unitary_magnus_xyz if method == "magnus" else unitary_trotter_xyz
                    U_pred = fn(model, t_scalar, ops)
                    U_time = fn(model, t_domain, ops)
                    U_t    = fn(model, t_u, ops)
                else:
                    U_pred = model(t_scalar)
                    U_time = model(t_domain)
                    U_t    = model(t_u)

                rho_pred = U_pred @ rho_in @ U_pred.conj().transpose(-2, -1)
                loss1    = torch.mean(torch.abs(rho_pred - rho_out))
                loss3    = torch.mean(torch.abs(U_t - U_teo))

                if is_large:
                    loss = loss1 + loss3
                else:
                    loss2 = torch.mean(torch.abs(
                        U_time.conj().transpose(-2, -1) @ U_time - I_batch
                    ))
                    loss = loss1 + loss2 + loss3

                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())  # acumula por batch

        # ── Promedio de la época ──────────────────────────────────────
        loss_values.append(np.mean(epoch_losses))
        scheduler.step()

    if loss_variation:
        return model, loss_values
    else:
        return model


# ----------------------------------------------------------
#  Fidelity evaluation
# ----------------------------------------------------------
# def fidelity_test_xyz(model, U_test, ops, N_qubits, close_U=False, method="magnus"):
#     """
#     Evaluate gate fidelity on the test grid.
#     """
#     times = torch.tensor(time_test, dtype=torch.float32).view(-1, 1)
#     is_large = N_qubits >= 7

#     with torch.no_grad():
#         if is_large:
#             fn      = unitary_magnus_xyz if method == "magnus" else unitary_trotter_xyz
#             U_model = fn(model, times, ops).cpu().numpy()
#         else:
#             U_model = model(times).detach().numpy()

#     fidelities = []
#     for i in range(len(time_test)):
#         U_pred = closest_unitary(U_model[i]) if close_U else U_model[i]
#         fidelities.append(fid_pros(U_pred, U_test[i]))

#     return np.array(fidelities)

def fidelity_test_xyz(model, U_test, ops, N_qubits, method="magnus", svd = True):
    """
    Evaluate gate fidelity on the test grid.
    """
    times = torch.tensor(time_test, dtype=torch.float32, device="cpu").view(-1, 1)
    is_large = N_qubits >= 7
    model   = model.to(device="cpu")
    ops     = ops.to(device="cpu")

    with torch.no_grad():
        if is_large:
            fn      = unitary_magnus_xyz if method == "magnus" else unitary_trotter_xyz
            U_model = fn(model, times, ops).cpu().numpy()
        else:
            U_model = model(times).detach().numpy()
    
    if svd:
        fidelities_true     = []
        fidelities_false    = []
        for i in range(len(time_test)):
            U_pred_false = U_model[i]
            U_pred_true = closest_unitary(U_model[i])

            fidelities_false.append(fid_pros(U_pred_false, U_test[i]))
            fidelities_true.append(fid_pros(U_pred_true, U_test[i]))

        return np.array(fidelities_true), np.array(fidelities_false)
    else:
        fidelities     = []
        for i in range(len(time_test)):
            U_pred = U_model[i]

            fidelities.append(fid_pros(U_pred, U_test[i]))

        return np.array(fidelities)


def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


# ----------------------------------------------------------
#  Full test pipeline
# ----------------------------------------------------------
def test_model_xyz(
    N: int = 2,
    device: str = "cpu",
    method: str = "trotter"
):
    """
    Full run: generate data → train → evaluate fidelity.

    Parameters
    ----------
    N      : number of qubits
    method : 'magnus' or 'trotter' (used for data generation and 7-8 qubit integration)

    Returns
    -------
    F_false : np.ndarray — fidelity without SVD correction
    F_true  : np.ndarray — fidelity with SVD correction
    coeffs  : np.ndarray — Hamiltonian parameters used in this run
    """
    # ── Step 1: build operator basis ──────────────────────────────────────────
    ops = build_ops_xyz(n=N, device=device)

    # ── Step 2: generate unitaries and dataset ────────────────────────────────
    data_U, coeffs = random_U_xyz(N=N, method=method)
    data           = gen_data_xyz(Us=data_U, N=N)

    # ── Step 3: initialize model ──────────────────────────────────────────────
    model = make_model_xyz(N)
    reset_weights(model)

    # ── Step 4: train ─────────────────────────────────────────────────────────
    model = train_model_xyz(
        model=model,
        data=data,
        data_U=data_U,
        ops=ops,
        N_qubits=N,
        device=device,
        method=method,
        batch_epoch=100,
        batch_size=10,
        learning_rate=1e-3,
        num_epochs=400,
        sch=100
    )

    # ── Step 5: reconstruct test unitaries ────────────────────────────────────
    data_U_test = U_from_coeffs_xyz(coeffs=coeffs, N=N, method=method)

    # ── Step 6: evaluate fidelity ─────────────────────────────────────────────
    F_true, F_false = fidelity_test_xyz(
        model=model, U_test=data_U_test, ops=ops,
        N_qubits=N, method=method
    )

    return F_false, F_true, coeffs


# ----------------------------------------------------------
#  Statistical analysis over multiple runs
# ----------------------------------------------------------
def run_statistics_xyz(
    N: int = 2,
    n_runs: int = 50,
    device: str = "cpu",
    method: str = "magnus"
):
    """
    Run the full pipeline n_runs times with different random Hamiltonians
    and collect fidelity statistics.

    Returns
    -------
    dict with keys:
        'mean_false', 'std_false' — stats without SVD correction
        'mean_true',  'std_true'  — stats with SVD correction
        'all_false', 'all_true'   — raw arrays (n_runs, n_time_test)
    """
    all_false = []
    all_true  = []

    for run in range(n_runs):
        print(f"\n── Run {run + 1}/{n_runs} | N={N} ──")
        F_false, F_true, _ = test_model_xyz(N=N, device=device, method=method)
        all_false.append(F_false)
        all_true.append(F_true)

    all_false = np.array(all_false)   # (n_runs, n_time_test)
    all_true  = np.array(all_true)

    return {
        "mean_false" : all_false.mean(axis=0),
        "std_false"  : all_false.std(axis=0),
        "mean_true"  : all_true.mean(axis=0),
        "std_true"   : all_true.std(axis=0),
        "all_false"  : all_false,
        "all_true"   : all_true,
    }