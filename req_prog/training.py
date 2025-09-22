import pandas as pd
import numpy as np
import torch
import os
import random

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

# Definimos la clase que transforma los datos en tensores de torch para el entrenamiento
class fc_Dataset(Dataset):
    def __init__(self, data_array):
        """
        Args:
            data_array (np.ndarray or torch.Tensor): de forma (N, 3, 4, 4)
        """
        if isinstance(data_array, np.ndarray):
            self.data = torch.tensor(data_array, dtype=torch.complex64)
        else:
            self.data = data_array

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        rho_in = sample[0]     # (4, 4)
        t_mat  = sample[1]     # (4, 4) — constante
        rho_out = sample[2]    # (4, 4)

        # Opcional: extraer el escalar t desde la matriz constante
        t_scalar = t_mat[0, 0].unsqueeze(0)  # (1,) como tensor

        # Entrada al modelo: rho_in + t_scalar
        x = (rho_in, t_scalar.real.to(torch.float32))
        y = rho_out

        return x, y

elements = {
    "dt0.1": [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12], 
    "dt0.25": [0, 3, 6, 9, 12], 
    "dt0.5": [0, 6, 12], 
    "dt1.0": [0, 12]
}

def train_model(
    model, 
    N_qbits = 2,
    delta_t = 0.1,
    num_epochs = 3000, 
    batch_size = 25,
    batch_epoch = 100, 
    learning_rate= 1e-3, 
    N_domain = 100,
    sch = None,
    device="cpu"
):
    
    data_train  = np.load(f"dataframe/data_model{N_qbits}_deltat-{delta_t}.npz")["arr_0"]

    # Load the theoretical unitary matrices
    elements = elements["dt"+str(delta_t)]
    U_teo = np.load(f"dataframe/unitary_N{N_qbits}.npy")
    U_teo = torch.tensor(U_teo[elements, :, :], dtype=torch.complex64, device = device)

    # Set some parameters for the data preparation
    n_dataload  = int(len(data_train)/batch_size)
    dataset     = fc_Dataset(data_train)

    # Shuffle the dataset
    index = np.arange(len(data_train))
    np.random.shuffle(index)
    split_index = np.array_split(index, n_dataload)

    # Create the differents dataloaders for trianing
    dataloaders = []
    for _ in range(len(split_index)):
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(split_index[_])
        )
        dataloaders.append(dataloader)

    # Set the parameters for the training
    model       = model.to(device)
    optimizer   = torch.optim.AdamW(model.parameters(), lr = learning_rate)

    if sch != None:
        scheduler   = torch.optim.lr_scheduler.StepLR(optimizer = optimizer, step_size = sch)

    I_batch     = torch.eye(2 ** N_qbits, dtype=torch.complex64, device=device).unsqueeze(0).repeat(N_domain, 1, 1)

    t_domain    = torch.linspace(0, 1, N_domain, device=device)
    t_domain    = t_domain.reshape(-1, 1).requires_grad_(True)
    t_test      = torch.linspace(0, 1, N_domain, device=device).reshape(-1, 1)


    times = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]
    times = np.array(times)
    times = times[elements]
    t_u   = torch.tensor(times, dtype=torch.float32, device=device).reshape(-1, 1)
    t_u   = t_u.requires_grad_(True)
    t_u_test = torch.tensor(times, dtype=torch.float32, device=device).reshape(-1, 1)

    # Entrenamiento
    loss_train  = []
    loss_test   = []

    for epoch in tqdm(range(num_epochs), desc=f'Training model{N_qbits} with delta_t={delta_t}', unit='epoch'):
        model.train()
        total_loss = 0

        # Elegir aleatoriamente batch_epoch dataloaders
        selected_dataloaders = random.sample(dataloaders, batch_epoch)

        for dl in selected_dataloaders:
            for (rho_in, t_scalar), rho_out in dl:
                rho_in = rho_in.to(device)
                t_scalar = t_scalar.to(device)
                rho_out = rho_out.to(device)

                optimizer.zero_grad()

                U_pred = model(t_scalar)
                U_time = model(t_domain)
                U_t    = model(t_u)

                # dUdt = get_dU_dt(model, t_domain)
                # dUdt_dagger = dUdt.conj().transpose(-2, -1)

                rho_pred    = U_pred @ rho_in @ U_pred.conj().transpose(-2, -1)
                loss_func1  = torch.mean(torch.abs(rho_pred - rho_out))
                loss_func2  = torch.mean(torch.abs(U_time.conj().transpose(-2, -1) @ U_time - I_batch))
                loss_func3  = torch.mean(torch.abs(U_t - U_teo))
                # loss_func4  = torch.mean(torch.abs(dUdt_dagger @ U_time + U_time.conj().transpose(-2, -1) @ dUdt))
            
                loss_func = loss_func1 + loss_func2 + loss_func3
                # loss_func = loss_func2 + loss_func3

                loss_func.backward()
                optimizer.step()

                total_loss += loss_func.item()

        avg_loss = total_loss / (batch_epoch * batch_size)
        loss_train.append(avg_loss)
        
        if sch != None:
            scheduler.step()
    
    # Save the model
    torch.save(model.state_dict(), os.path.join("models", f"model{N_qbits}_deltat-{delta_t}.pth"))

    # Save the losses
    df = pd.DataFrame({
        'train_loss': loss_train
    })

    df.to_csv(os.path.join("./data_loss", f"dataloss-model{N_qbits}-delta_t{delta_t}.csv"), index=False)