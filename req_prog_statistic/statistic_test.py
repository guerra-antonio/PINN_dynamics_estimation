from req_prog_statistic.h_magnus import fid_pros, unitary_general, unitary_ising
from req_prog_statistic.h_magnus_xyz import unitary_evolution_xyz
from req_prog_statistic.h_torch import unitary_trotter_torch, unitary_magnus_torch, build_ops_ising
from req_prog_statistic.h_torch_xyz import build_ops_xyz
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
# Hamiltonian families
# -----------------------------------------------------------
# The family is the first thing an experiment fixes; it then determines how many
# coefficients define a realization, how strong the couplings are, how many
# Trotter slices the target unitaries need, up to which system size it is
# tractable, and which ansatz can be trained against it.
#
# 'general' keeps the couplings and per-N normalization of the published paper:
#   it is the proof-of-fire case, the densest possible N-qubit Hamiltonian, and
#   its numbers must stay comparable with the figures already in the manuscript.
# 'ising' and 'xyz' are the physically-motivated sparse families, and they use
#   deliberately stronger couplings so the dynamics is not close to a constant
#   generator (where a trivial two-point interpolation already scores ~0.99 and
#   high fidelities would say more about the regime than about the model).
#
# n_draw vs n_params: the Ising family draws one extra coefficient and uses the
#   slice [1 : 2N], mirroring unitary_magnus_78 in the original pipeline.
# steps: chosen so the per-slice discretization parameter dt*||H|| stays well
#   below 1 at the family's coupling strength.
# ansatz: which model_H values are valid. Predicting effective-Hamiltonian
#   coefficients presupposes that the support of H is known, which is true for
#   the sparse families but false for 'general', so the general family only
#   admits direct U(t) prediction.

_PAPER_GENERAL_DIVISOR = {2: 2, 3: 3, 4: 4, 5: 16, 6: 24}

H_FAMILIES = {
    'general': {
        'n_draw':    lambda N: 4 ** N,
        'n_params':  lambda N: 4 ** N,
        'coupling':  lambda N: 1.0 / _PAPER_GENERAL_DIVISOR[N],
        'omega_max': 8 * np.pi,   # matches the published figures: req_prog/h_magnus.py
                                  # draws omega as 4 * uniform[0, 2*pi), verified by
                                  # reproducing dataframe/unitary_N*.npy exactly
        'steps':     100,
        'max_N':     6,
        'ansatz':    ('default',),
        'ops':       None,
    },
    'ising': {
        'n_draw':    lambda N: 2 * N,
        'n_params':  lambda N: 2 * N - 1,
        # Calibrated so the trivial two-anchor constant-generator baseline
        # (Hbar = i*log U(1), U(t) = exp(-i*Hbar*t)) scores ~0.90 mean fidelity
        # at delta_t=1.0, with 15% margin over the measured threshold. A fixed
        # coupling=2.0 across N was inconsistent: it sat BELOW the non-trivial
        # threshold at N=2,4,5 (baseline still ~wins) and unnecessarily ABOVE it
        # at N=6-8 (harder to train than needed, since more terms accumulate
        # operator norm even at fixed per-term coupling). Fit: coupling ~ N^-0.457,
        # from bisection at N=2..8 (calibrate_coupling.py).
        #
        # Halved relative to that calibrated value: at N=2 the full calibration
        # concentrated a lot of amplitude onto only 3 Ising terms, producing
        # locally sharper dynamics than the small H_eff network could reliably
        # track with n_train=50. Splitting the distance between the uncoupled
        # baseline (coupling=1) and the calibrated value keeps the regime
        # clearly past the non-trivial threshold while easing the training
        # difficulty this raised.
        'coupling':  lambda N: 1 + (3.775 * N**-0.457 * 1.15 - 1) / 2,
        'omega_max': 6 * np.pi,
        'steps':     50,
        'max_N':     8,
        'ansatz':    ('default', 'trotter', 'magnus'),
        'ops':       build_ops_ising,
    },
    'xyz': {
        'n_draw':    lambda N: 3 * (N - 1),
        'n_params':  lambda N: 3 * (N - 1),
        'coupling':  lambda N: 2.0,
        'omega_max': 6 * np.pi,
        'steps':     50,
        'max_N':     8,
        'ansatz':    ('default', 'trotter', 'magnus'),
        'ops':       build_ops_xyz,
    },
}


def family_config(H_family):
    """Look up a family, failing loudly on a typo rather than silently defaulting."""
    if H_family not in H_FAMILIES:
        raise ValueError(f"H_family must be one of {sorted(H_FAMILIES)}, got {H_family!r}")
    return H_FAMILIES[H_family]


def check_setup(H_family, N, model_H):
    """
    Validate a (family, size, ansatz) combination before any compute is spent.

    Catches the two mistakes that are silent but ruin a run: training an
    effective-Hamiltonian ansatz against a Hamiltonian whose support it cannot
    represent, and running a family past the size where it is tractable.
    """
    cfg = family_config(H_family)
    if model_H not in cfg['ansatz']:
        raise ValueError(
            f"H_family={H_family!r} admits model_H in {cfg['ansatz']}, got {model_H!r}. "
            "Predicting H_eff coefficients assumes the support of H is known, "
            "which does not hold for the general Pauli family."
        )
    if N > cfg['max_N']:
        raise ValueError(f"H_family={H_family!r} is defined up to N={cfg['max_N']}, got N={N}")
    return cfg


def n_coeffs_for(N, H_family):
    """Number of coefficients drawn for one realization of this family."""
    return family_config(H_family)['n_draw'](N)


def n_params_for(N, H_family):
    """Number of physical parameters an effective-Hamiltonian ansatz must output."""
    return family_config(H_family)['n_params'](N)


def build_ops(N, H_family, device="cpu"):
    """Operator basis for the effective-Hamiltonian ansatz of this family."""
    cfg = family_config(H_family)
    if cfg['ops'] is None:
        raise ValueError(f"H_family={H_family!r} has no effective-Hamiltonian basis")
    return cfg['ops'](N, device=device)


def sample_H_coeffs(N, H_family):
    """
    Draw one Hamiltonian realization: amplitudes, frequencies and phases.

    Sampling it here, rather than inside random_U, lets a single realization be
    shared by every delta_t of a run, so the four grids describe the SAME physical
    system sampled at different temporal densities. Drawing a new Hamiltonian per
    delta_t makes the curves within a panel incomparable.
    """
    cfg = family_config(H_family)
    n_coeff = cfg['n_draw'](N)

    amplitude = np.random.rand(n_coeff)                      # amplitudes per term
    omega     = cfg['omega_max'] * np.random.rand(n_coeff)   # angular frequencies
    phase     = 2 * np.pi * np.random.rand(n_coeff)          # phase shifts
    return np.array([amplitude, omega, phase])


def build_U(coeffs, time, N, H_family, method="trotter"):
    """U(t) for one realization of the given family, at a single time."""
    cfg = family_config(H_family)
    amplitude, omega, phase = coeffs
    kwargs = dict(amplitudes=amplitude, omega=omega, phase=phase, time=time,
                  method=method, steps=cfg['steps'], coupling=cfg['coupling'](N))

    if H_family == 'general':
        return unitary_general(N=N, **kwargs)
    elif H_family == 'ising':
        return unitary_ising(n_qubits=N, **kwargs)
    else:
        return unitary_evolution_xyz(N=N, **kwargs)


def random_U(N=2, time_grid=10, H_family='ising', coeffs=None, method="trotter"):
    # --- Hamiltonian parameters: reuse the given realization, or draw a new one ---
    if coeffs is None:
        coeffs = sample_H_coeffs(N, H_family)

    # Compute U(t) at each point of the training grid
    time = np.linspace(0, 1, time_grid + 1)
    Us = [build_U(coeffs, t, N, H_family, method)
          for t in tqdm(time, desc="Generating unitaries for training")]

    return np.array(Us), np.asarray(coeffs), time

# -----------------------------------------------------------
# Reconstructs the same family of unitaries U(t) using
# previously saved Hamiltonian coefficients (from random_U).
# Useful for validation or testing with a denser time grid.
# -----------------------------------------------------------
def U_from_coeffs(coeffs, N=2, H_family='ising', method="trotter"):
    Us = [build_U(coeffs, t, N, H_family, method)
          for t in tqdm(time_test, desc="Generating unitaries for testing")]
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

    # Predict U(t) from the trained model. This is inference only: without
    # no_grad the Trotter/Magnus unrolling keeps the autograd graph of all 50
    # sequential matrix exponentials over the 101 test times, which for N = 8
    # means several GB of GPU memory for nothing.
    with torch.no_grad():
        if model_H == "trotter":
            U_model = unitary_trotter_torch(model, times, ops=ops)
        elif model_H == "magnus":
            U_model = unitary_magnus_torch(model, times, ops=ops)
        else:
            U_model = model(times)
        U_model = U_model.cpu().numpy()

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
def test_model(ops, N=2, device="cpu", model_H = "default", H_family='ising'):
    check_setup(H_family, N, model_H)

    # Step 1: generate unitaries and corresponding training data
    data_U, coeffs, time = random_U(N=N, H_family=H_family)
    data = gen_data(Us=data_U, time=time, N=N)

    # Step 2: initialize the model
    model = UnitaryModel(n_qubits=N, type='H' if model_H in ('trotter', 'magnus') else 'U',
                         n_params=n_params_for(N, H_family))

    # Step 3: train the model
    model = train_model(model=model, 
                        data=data, 
                        data_U=data_U, 
                        device=device, 
                        model_H=model_H,
                        batch_epoch=10,
                        num_epochs=1000,
                        sch=300,
                        ops=ops
                        )

    # Step 4: reconstruct unitaries for testing
    data_U = U_from_coeffs(coeffs=coeffs, N=N, H_family=H_family)

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

def test_model_unitarity(ops, N=2, time_grid=10, device="cpu", model_H = "default",
                         H_family='ising'):
    check_setup(H_family, N, model_H)

    # Step 1: generate unitaries and corresponding training data
    data_U_train, coeffs, time  = random_U(N=N, time_grid=time_grid, H_family=H_family)
    data_U_test                 = U_from_coeffs(coeffs=coeffs, N=N, H_family=H_family)
    data                        = gen_data(Us=data_U_train, time=time, N=N)

    # Step 2: initialize the model
    if model_H in ['trotter', 'trotter']:
        model = UnitaryModel(n_qubits=N, type='H', n_params=n_params_for(N, H_family))
    else:
        model = UnitaryModel(n_qubits=N, type='U')

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
                        time_grid=time_grid,
                        ops=ops
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
                        time_grid=time_grid,
                        ops=ops
                        )
    F_false = fidelity_test(ops=ops, model=model, U_test=data_U_test, close_U=False, model_H=model_H)

    return F_false, F_true, coeffs

def test_model_12(ops, N=2, time_grid=10, device="cpu", model_H = "default",
                  H_family='ising', n_train=11000, batch_epoch=None, n_data=None,
                  coeffs=None, num_epochs=1000, sch=300):
    """
    H_family selects the target Hamiltonian ('general', 'ising' or 'xyz'), which
    fixes the coupling strength, the frequency range and the number of Trotter
    slices used to build the targets. It also constrains model_H: the general
    Pauli family only admits direct U(t) prediction, since an effective-
    Hamiltonian ansatz would assume a support the true H does not have.

    coeffs is one Hamiltonian realization, as returned by sample_H_coeffs. Passing
    the same realization for every delta_t makes the resulting curves describe the
    same physical system at different temporal sampling densities, which is what
    makes them comparable within a panel. Leaving it as None draws a new
    Hamiltonian on every call.

    n_train is the total number of (rho_0, t, rho_t) samples, split evenly over
    the time points of the grid, following gen_data.py in the original pipeline
    (N_data = 11000 for every system size). Keeping the total fixed rather than
    the number of samples per time means that the comparison across delta_t
    isolates the effect of the temporal sampling instead of also changing the
    amount of data.

    n_data overrides that split with a fixed number of samples per time point.
    It is offered as an escape hatch for memory-constrained runs; note that it
    makes the dataset size grow with the number of time points, so comparisons
    across delta_t no longer hold the amount of data fixed.

    The training budget reproduces training_models.py of the original pipeline:
    num_epochs = 1000, sch = 300, and batch_epoch = 100 when the model predicts
    U(t) directly (train_model, N <= 6) or 10 when it predicts the effective
    Hamiltonian (train_model_h, N = 7, 8). Note that one "epoch" here does not
    traverse the dataset: it draws batch_epoch loaders of batch_size samples each,
    so the number of weight updates is num_epochs * batch_epoch.
    """
    check_setup(H_family, N, model_H)

    if batch_epoch is None:
        batch_epoch = 100 if model_H == "default" else 10

    # Step 1: generate unitaries and corresponding training data
    data_U_train, coeffs, time  = random_U(N=N, time_grid=time_grid,
                                           H_family=H_family, coeffs=coeffs)
    data_U_test                 = U_from_coeffs(coeffs=coeffs, N=N, H_family=H_family)
    if n_data is None:
        n_data                  = max(1, n_train // len(time))
    data                        = gen_data(Us=data_U_train, time=time, N=N, n_data=n_data)

    # Step 2: initialize the model
    if model_H in ['trotter', 'magnus']:
        model = UnitaryModel(n_qubits=N, type='H', n_params=n_params_for(N, H_family))
        unitarity = False
    else:
        model = UnitaryModel(n_qubits=N, type='U', residual=True)
        unitarity = True

    # Step 3: train the model
    model   = train_model(model=model, 
                        data=data, 
                        data_U=data_U_train, 
                        device=device, 
                        model_H=model_H,
                        batch_epoch=batch_epoch,
                        batch_size=10,
                        learning_rate=1e-3,
                        num_epochs=num_epochs,
                        sch=sch,
                        unitarity=unitarity,
                        time_grid=time_grid,
                        ops=ops,
                        data_input='1'
                        )
    F_1 = fidelity_test(ops=ops, model=model, U_test=data_U_test, close_U=False, model_H=model_H)

    model   = train_model(model=model,
                        data=data, 
                        data_U=data_U_train, 
                        device=device, 
                        model_H=model_H,
                        batch_epoch=batch_epoch,
                        batch_size=10,
                        learning_rate=1e-3,
                        num_epochs=num_epochs,
                        sch=sch,
                        unitarity=unitarity,
                        time_grid=time_grid,
                        ops=ops,
                        data_input='12'
                        )
    F_12 = fidelity_test(ops=ops, model=model, U_test=data_U_test, close_U=False, model_H=model_H)

    return F_1, F_12, coeffs