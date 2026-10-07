import pandas as pd
import numpy as np
import os
from req_prog_statistic.statistic_test import (
    test_model_12, sample_H_coeffs, build_ops, check_setup, H_FAMILIES,
)

# ---------------------------------------------------------------------------
# Experiment definition. The Hamiltonian family is fixed first: it determines
# the coupling strength, the frequency range, the number of Trotter slices used
# to build the targets, the tractable size range, and which ansatz is allowed.
#
#   'general' : all 4^N Pauli strings, with the couplings of the published
#               paper. The proof-of-fire case; only admits model_H='default'
#               (direct U(t) prediction), since an effective-Hamiltonian ansatz
#               would assume a support the true H does not have.
#   'ising'   : nearest-neighbor Ising, strong couplings, N up to 8.
#   'xyz'     : nearest-neighbor XYZ Heisenberg, strong couplings, N up to 8.
# ---------------------------------------------------------------------------
H_family = "ising"
model_H = "default"          # 'default' -> predict U(t);  'trotter'/'magnus' -> predict H_eff
Ns = list(range(2, 7))

n_test = 10
n_train = 1000                # total (rho_0, t, rho_t) samples, split over the time grid.
                              # Raised from 50: the 'default' ansatz outputs the full d x d
                              # unitary (9M+ weights at N=6), so 50 samples badly
                              # underdetermined it -- unlike 'trotter', whose output is
                              # only 2N-1 coefficients and stayed fine at n_train=50.
num_epochs = 400
sch = 150                    # StepLR: lr x0.1 cada 150 epocas
device = "cuda"

save_path = f'dataframe/fid_stat_{H_family}_{model_H}_n{Ns[0]}{Ns[-1]}-1.pkl'

# Fail on an invalid (family, size, ansatz) combination before spending compute
for N in Ns:
    check_setup(H_family, N, model_H)
print(f"Family: {H_family} | ansatz: {model_H} | N={Ns} | "
      f"coupling={H_FAMILIES[H_family]['coupling'](Ns[0]):.4g}..{H_FAMILIES[H_family]['coupling'](Ns[-1]):.4g} | "
      f"steps={H_FAMILIES[H_family]['steps']}")

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
    # The operator basis is only used by the effective-Hamiltonian ansatz; for
    # direct U(t) prediction it is passed through but never contracted.
    ops = build_ops(N, H_family if H_family != 'general' else 'ising', device=device)

    for test_idx in range(n_test):
        # One Hamiltonian realization shared by the four temporal grids, so the
        # resulting curves describe the same physical system at different sampling
        # densities and are comparable within a panel.
        coeffs_run = sample_H_coeffs(N, H_family)

        for tdx_grid in [1, 2, 4, 10]:
            print(f"Running test: N={N}, time_grid: dt={tdx_grid}, iteration={test_idx+1}/{n_test}")
            F_1, F_12, coeffs = test_model_12(
                ops=ops, N=N, time_grid=tdx_grid, device=device,
                model_H=model_H, H_family=H_family,
                n_train=n_train, num_epochs=num_epochs, sch=sch, coeffs=coeffs_run,
            )
            delta_t = np.float32(1/tdx_grid)

            for loss_type, F in [('1', F_1), ('12', F_12)]:
                results.append({
                    'N': N,
                    'H_family': H_family,
                    'model_H': model_H,
                    'loss_type': loss_type,
                    'F': F.astype(np.float32),
                    'coeffs': coeffs.astype(np.float32),
                    "delta_t": delta_t
                })

            # Convert to DataFrame and save checkpoint
            df = pd.DataFrame(results)
            df.to_pickle(save_path)

print(f"\n✅ Finished. Total entries saved: {len(results)}")
