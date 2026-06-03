import pandas as pd
import numpy as np
import os
import torch

from req_prog_statistic.statistic_test_xyz import *
from req_prog_statistic.h_torch_xyz import build_ops_xyz
from req_prog_statistic.check_H import check_at_T

# ─────────────────────────────────────────────────────────────
# Create output folders
# ─────────────────────────────────────────────────────────────
os.makedirs("dataframe", exist_ok=True)
os.makedirs("models/models-full-loss", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Global dataframe container
# ─────────────────────────────────────────────────────────────
all_results = []
Ns = list(range(7, 9))
device = "cuda"

for N in Ns:
    converges = False
    while not converges:
        data_U, coeffs = random_U_xyz(
            N=N,
            method="trotter",
            time_grid=time_all
        )
        converges = check_at_T(coeffs=coeffs, T=1.0, norm_N=3.0)

    data = gen_data_xyz(
            Us=data_U,
            N=N,
            time=time_all
        )
    data_U_test = U_from_coeffs_xyz(
            coeffs=coeffs,
            N=N,
            method="trotter"
        )

    for dt in [0.1, 0.5]:
        for method in ["trotter", "magnus"]:
            print(f"\nTraining -> N={N} | dt={dt} | method={method}")

            # ─────────────────────────────────────────────────
            # Step 1: build operator basis
            # ─────────────────────────────────────────────────
            ops = build_ops_xyz(n=N, device=device)

            # ─────────────────────────────────────────────────
            # Step 2: extract dataset
            # ─────────────────────────────────────────────────
            data_U_ = data_U[time_index[dt], :, :]
            data_   = data[np.isin(np.real(data[:, 1, 0, 0]), np.float32(time_grid[dt])), :, :, :]

            # ─────────────────────────────────────────────────
            # Step 3: initialize model
            # ─────────────────────────────────────────────────
            model = make_model_xyz(N)
            reset_weights(model)

            # ─────────────────────────────────────────────────
            # Step 4: train
            # ─────────────────────────────────────────────────
            model, loss = train_model_xyz(
                model=model,
                data=data_,
                data_U=data_U_,
                ops=ops,
                N_qubits=N,
                device=device,
                method=method,
                batch_epoch=10,
                batch_size=10,
                learning_rate=1e-3,
                num_epochs=400,
                sch=100,
                loss_variation=True
            )

            # ─────────────────────────────────────────────────
            # Step 5: evaluate fidelity
            # ─────────────────────────────────────────────────
            Fidelity = fidelity_test_xyz(
                model=model,
                U_test=data_U_test,
                ops=ops,
                N_qubits=N,
                method=method,
                svd=False
            )
            print(f"Fidelity -> mean: {np.mean(Fidelity):.4f} | std: {np.std(Fidelity):.4f}")

            # ─────────────────────────────────────────────────
            # Save model
            # ─────────────────────────────────────────────────
            model_name = (
                f"model{N}_XYZ_deltat-{dt}-{method}.pth"
            )

            model_path = os.path.join(
                "models/models-full-loss",
                model_name
            )

            torch.save(model.state_dict(), model_path)

            print(f"Saved model -> {model_path}")

            # ─────────────────────────────────────────────────
            # Save experiment row
            # ─────────────────────────────────────────────────
            row = {
                "N": N,
                "dt": dt,
                "method": method,
                "device": device,
                "coeffs": coeffs.tolist()
                    if isinstance(coeffs, np.ndarray)
                    else coeffs,
                "fidelity_mean": float(np.mean(Fidelity)),
                "fidelity_std": float(np.std(Fidelity)),
                "fidelity_all": Fidelity.tolist()
                    if isinstance(Fidelity, np.ndarray)
                    else Fidelity,
                "loss": loss
                    if isinstance(loss, (np.ndarray, list))
                    else [loss],
                "model_path": model_path
            }

            all_results.append(row)

# ─────────────────────────────────────────────────────────────
# Build dataframe
# ─────────────────────────────────────────────────────────────
df = pd.DataFrame(all_results)

# ─────────────────────────────────────────────────────────────
# Save dataframe
# ─────────────────────────────────────────────────────────────
df.to_csv(
    f"dataframe/results_xyz_full_loss_{device}.csv",
    index=False
)

print("\nDataframe saved!")