import torch
import torch.nn as nn

class UnitaryModel(nn.Module):
    def __init__(self, n_qubits: int, hidden_dim: int = 50, scale: int = 1, type: str = 'U',
                 residual: bool = False, n_params: int = None):
        """
        Unified model that predicts, from a scalar time input, either:
        - a complex matrix (2^N x 2^N), i.e. the evolution operator U(t), or
        - a vector of `n_params` real coefficients defining an effective
          Hamiltonian H_eff(t) in the operator basis of a given family.

        Args:
            n_qubits (int): Number of qubits.
            hidden_dim (int): Base hidden layer size (used in 'H' mode).
            scale (int): Width scaling factor (used in 'U' mode).
            type (str): 'U' to predict the unitary, 'H' to predict the Ising
                coefficients. If None, falls back to the size-based rule used
                previously ('U' for N <= 6, 'H' for N >= 7).
            residual (bool): if True (only for type='U'), the network output is
                treated as a time-scaled correction to the identity,

                    U(t) = 1 + t * f_theta(t),

                instead of the full matrix. Because the correction is multiplied
                by t, the initial condition U(0) = 1 holds exactly for any value
                of the weights and at every point of training: it becomes a hard
                constraint of the parametrisation rather than something the loss
                has to learn. The network only has to represent how the dynamics
                departs from the identity, which is small at short times.

        Notes:
            In 'H' mode the output order must match the family's operator basis:
            build_ops_ising expects the N transverse-field coefficients of X_j
            first and the N-1 couplings of Z_j Z_{j+1} after; build_ops_xyz
            expects them grouped per bond as [XX, YY, ZZ] for each of the N-1
            bonds.

            Weights are randomly initialised in both parametrisations.
        """
        super().__init__()

        if type not in ('U', 'H'):
            raise ValueError("type must be 'U' (unitary) or 'H' (Ising coefficients)")
        if residual and type != 'U':
            raise ValueError("residual=True is only defined for type='U'")

        self.n_qubits = n_qubits
        self.d = 2 ** n_qubits
        self.type = type
        self.residual = residual

        if type == 'U':
            # Full matrix prediction: (2^N x 2^N) complex
            output_dim = self.d * self.d * 2  # real + imag
            layers = [nn.Linear(1, 64 * scale), nn.Tanh()]
            layers += [nn.Linear(64 * scale, 128 * scale), nn.Tanh()]
            if n_qubits >= 4:
                layers += [nn.Linear(128 * scale, 256 * scale), nn.Tanh()]
            if n_qubits >= 5:
                layers += [nn.Linear(256 * scale, 512 * scale), nn.Tanh()]
            if n_qubits >= 6:
                layers += [nn.Linear(512 * scale, 1024 * scale), nn.Tanh()]
            layers += [nn.Linear(layers[-2].out_features, output_dim)]
            self.net = nn.Sequential(*layers)
            self.mode = 'matrix'
        else:
            # Effective Hamiltonian: one coefficient per operator in the family's
            # basis (2N-1 for nearest-neighbor Ising, 3(N-1) for XYZ). Must match
            # the operator basis it will be contracted against, so it is passed
            # in explicitly rather than assumed.
            if n_params is None:
                raise ValueError(
                    "type='H' requires n_params, the number of coefficients of the "
                    "target Hamiltonian family (e.g. n_params_for(N, H_family))"
                )
            output_dim = n_params
            self.net = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, output_dim)
            )
            self.mode = 'coeffs'

        self.output_dim = output_dim

    def reset_parameters(self):
        """
        Re-initialise every layer of the network.

        Defined explicitly because reset_weights() in the training routines only
        inspects the immediate children of the model, which is the Sequential
        container and carries no reset_parameters of its own.
        """
        for layer in self.net:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t (torch.Tensor): shape (B, 1), time input

        Returns:
            torch.Tensor:
                - (B, d, d) complex matrix if type == 'U'
                - (B, 2N - 1) real vector of Ising coefficients if type == 'H'
        """
        x = self.net(t)

        if self.type == 'U':
            x = x.view(-1, 2, self.d, self.d)
            U = x[:, 0] + 1j * x[:, 1]  # complex matrix

            if self.residual:
                # U(t) = 1 + t * f(t): the identity is exact at t = 0 by construction
                U = U * t.reshape(-1, 1, 1).to(U.dtype)
                U = U + torch.eye(self.d, dtype=U.dtype, device=U.device)

            return U
        else:
            return x  # real vector of coefficients
