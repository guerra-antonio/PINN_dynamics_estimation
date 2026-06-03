import pandas as pd
import numpy as np
import os
from req_prog_statistic.statistic_test import test_model_unitarity
from req_prog_statistic.h_torch import build_ops_ising

# Parameters
n_test = 25

# Ns = list(range(4,7))
Ns = [6]
save_path = f'dataframe/fidelity_statistic_results_unitary6.pkl'

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
    if N < 5:
        device = "cpu"
    else:
        device = "cuda:1"

    ops = build_ops_ising(N, device=device)

    for test_idx in range(n_test):
        for tdx_grid in [1, 2, 4, 10]:
            print(f"Running test: N={N}, time_grid: dt={tdx_grid}, iteration={test_idx+1}/{n_test}")
            F_false, F_true, coeffs = test_model_unitarity(ops=ops, N=N, time_grid=tdx_grid, device=device, model_H="default")
            delta_t = np.float32(1/tdx_grid)

            results.append({
                'N': N,
                'Unitarity': True,
                'F': F_true.astype(np.float32),
                'coeffs': coeffs.astype(np.float32),
                "delta_t": delta_t
            })

            results.append({
                'N': N,
                'Unitarity': False,
                'F': F_false.astype(np.float32),
                'coeffs': coeffs.astype(np.float32),
                "delta_t": delta_t
            })

            # Convert to DataFrame and save checkpoint
            df = pd.DataFrame(results)
            df.to_pickle(save_path)

print(f"\n✅ Finished. Total entries saved: {len(results)}")