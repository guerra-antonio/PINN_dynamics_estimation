import pandas as pd
import numpy as np
import os
from req_prog_statistic.statistic_test import test_model
from req_prog_statistic.h_torch import build_ops_ising

# Parameters
n_test = 25

device = input("Enter the device to work: ")
print("Device:", device)

Ns = [8]
model_H_ = "trotter"
save_path = f'dataframe/fidelity_statistic_results_8.pkl'

# Ensure directory exists
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# If the file exists, load it to continue appending
if os.path.exists(save_path):
    df = pd.read_pickle(save_path)
    results = df.to_dict('records')
    print(f"🔁 Loaded existing file with {len(df)} records.")
else:
    results = []
    print("🆕 Starting new results file.")

# Main loop
for N in Ns:
    ops = build_ops_ising(N, device=device)

    if N in [7, 8]:
        model_H = model_H_
    else:
        model_H = "default"

    for test_idx in range(n_test):
        print(f"Running test: N={N}, iteration={test_idx+1}/{n_test}")
        F_false, F_true, coeffs = test_model(ops=ops, N=N, device=device, model_H=model_H)

        results.append({
            'N': N,
            'close_U': True,
            'F': F_true.astype(np.float32),
            'coeffs': coeffs.astype(np.float32)
        })

        results.append({
            'N': N,
            'close_U': False,
            'F': F_false.astype(np.float32),
            'coeffs': coeffs.astype(np.float32)
        })

        # Convert to DataFrame and save checkpoint
        df = pd.DataFrame(results)
        df.to_pickle(save_path)

print(f"\n✅ Finished. Total entries saved: {len(results)}")