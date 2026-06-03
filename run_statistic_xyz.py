import pandas as pd
import numpy as np
import os
from req_prog_statistic.statistic_test_xyz import test_model_xyz
from req_prog_statistic.h_torch_xyz import build_ops_xyz

# ── Parameters ────────────────────────────────────────────────────────────────
n_test = 50
device = "cpu"
method = "trotter"   # 'magnus' or 'trotter' — used for data generation
                    # and for 7-8 qubit integration

# device = input("Enter the device to work: ")
# print("Device:", device)

Ns = list(range(2, 9))
save_path = "dataframe/fidelity_statistic_results_xyz24.pkl"

# ── Load or initialize results ────────────────────────────────────────────────
os.makedirs(os.path.dirname(save_path), exist_ok=True)

if os.path.exists(save_path):
    df      = pd.read_pickle(save_path)
    results = df.to_dict("records")
    print(f"🔁 Loaded existing file with {len(df)} records.")
else:
    results = []
    print("🆕 Starting new results file.")

# ── Main loop ─────────────────────────────────────────────────────────────────
for N in Ns:
    ops = build_ops_xyz(N, device=device)

    for test_idx in range(n_test):
        print(f"\nRunning test: N={N}, iteration={test_idx + 1}/{n_test}")

        F_false, F_true, coeffs = test_model_xyz(
            N=N,
            device=device,
            method=method
        )

        results.append({
            "N"       : N,
            "close_U" : True,
            "F"       : F_true.astype(np.float32),
            "coeffs"  : coeffs.astype(np.float32),
            "method"  : method
        })

        results.append({
            "N"       : N,
            "close_U" : False,
            "F"       : F_false.astype(np.float32),
            "coeffs"  : coeffs.astype(np.float32),
            "method"  : method
        })

        # Checkpoint after every run
        df = pd.DataFrame(results)
        df.to_pickle(save_path)
        print(f"   ✅ Saved checkpoint ({len(results)} entries so far)")

print(f"\n✅ Finished. Total entries saved: {len(results)}")
