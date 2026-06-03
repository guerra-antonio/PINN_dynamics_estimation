import pandas as pd
import numpy as np
import torch
import random

from req_prog_statistic.h_torch import unitary_trotter_torch, build_ops_ising, unitary_magnus_torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler


# -----------------------------------------------------------
# Dataset class: converts input arrays into PyTorch tensors
# -----------------------------------------------------------
class fc_Dataset(Dataset):
    def __init__(self, data_array):
        """
        Args:
            data_array (np.ndarray or torch.Tensor): shape (N, 3, 4, 4)
        """
        if isinstance(data_array, np.ndarray):
            self.data = torch.tensor(data_array, dtype=torch.complex64)
        else:
            self.data = data_array

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]

        rho_in  = sample[0]  # (4, 4)
        t_mat   = sample[1]  # (4, 4), constant time matrix
        rho_out = sample[2]  # (4, 4)

        # Extract scalar time value from the constant matrix
        t_scalar = t_mat.real[0, 0].unsqueeze(0).to(torch.float32)

        # Input (ρ₀, t), output (ρₜ)
        x = (rho_in, t_scalar)
        y = rho_out
        return x, y


# -----------------------------------------------------------
# Resets all learnable parameters of a PyTorch model to their
# initial random state (useful before retraining).
# -----------------------------------------------------------
def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

# -----------------------------------------------------------
# Training routine for a model that predicts U(t)
# -----------------------------------------------------------
def train_model(
    model, 
    data,
    data_U,
    num_epochs = 1000, 
    batch_size = 30,
    batch_epoch = 10, 
    learning_rate = 1e-2, 
    N_domain = 100,
    sch = 300,
    device = "cpu",
    model_H = "default",
    unitarity = True,
    time_grid = 10
):
    reset_weights(model)
    N_qbits = int(np.log2(data.shape[-1]))
    ops = build_ops_ising(n=N_qbits, device=device)
    U_teo = torch.tensor(data_U, dtype=torch.complex64, device=device)

    # Prepare dataset + loaders
    n_dataload = int(data.shape[0] / batch_size)
    dataset = fc_Dataset(data)

    index = np.arange(data.shape[0])
    np.random.shuffle(index)
    split_index = np.array_split(index, n_dataload)

    dataloaders = [
        DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(subset.tolist())) 
        for subset in split_index
    ]

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch)

    # Identity batch for unitarity loss over full domain (we will subsample it)
    d = 2 ** N_qbits
    I_single = torch.eye(d, dtype=torch.complex64, device=device)
    I_batch = I_single.unsqueeze(0).repeat(N_domain, 1, 1)  # (N_domain, d, d)

    # Time grids
    t_domain = torch.linspace(0, 1, N_domain, device=device).reshape(-1, 1)   # (N_domain, 1)
    t_u = torch.linspace(0, 1, time_grid+1, device=device).reshape(-1, 1)              # (11, 1)

    # ---------------- TRAINING LOOP ----------------
    for epoch in tqdm(range(num_epochs), desc="Training model"):
        model.train()
        total_loss = 0.0

        # Select random dataloaders for this epoch
        selected_dataloaders = random.sample(dataloaders, min(batch_epoch, len(dataloaders)))

        for dl in selected_dataloaders:
            for (rho_in, t_scalar), rho_out in dl:
                rho_in   = rho_in.to(device)
                t_scalar = t_scalar.to(device)
                rho_out  = rho_out.to(device)
                
                # ---- Zero gradients ----
                optimizer.zero_grad()

                # ---- Recompute U_time and U_t before each backward (current model state) ----
                if model_H == "magnus":
                    U_time = unitary_magnus_torch(model = model, time = t_domain, ops=ops)
                    U_t    = unitary_magnus_torch(model = model, time = t_u, ops=ops)
                elif model_H == "trotter":
                    U_time = unitary_trotter_torch(model = model, time = t_domain, ops=ops)
                    U_t    = unitary_trotter_torch(model = model, time = t_u, ops=ops)
                elif model_H == "default":
                    U_time  = model(t_domain)
                    U_t     = model(t_u)
                else:
                    print("Error! Please insert method as magnus, trotter or default")
                    break

                # ---- Predict U(t) for current batch time ----
                if model_H == "trotter":
                    U_pred = unitary_trotter_torch(model, t_scalar, ops=ops)  # (B, d, d) or (1, d, d)
                else:
                    U_pred = model(t_scalar)
                # State evolution: ρ_pred = U ρ_in U†
                rho_pred = U_pred @ rho_in @ U_pred.conj().transpose(-2, -1)

                # ---- Losses ----

                # 1) Data fidelity loss
                loss_func1 = torch.mean(torch.abs(rho_pred - rho_out))

                # 2) Theoretical consistency loss at fixed probe times t_u
                loss_func2 = torch.mean(torch.abs(U_t - U_teo))

                # Total loss
                if unitarity:
                    # 3) Unitarity loss on a random subset of the time domain
                    loss_func3 = torch.mean(torch.abs(
                        U_time.conj().transpose(-2, -1) @ U_time - I_batch
                        ))
                    loss_func = loss_func1 + loss_func2 + loss_func3
                else:
                    loss_func = loss_func1 + loss_func2

                loss_func.backward()
                optimizer.step()

                total_loss += loss_func.item()

        scheduler.step()
    return model