import torch
import torch.nn as nn

class UnitaryModel(nn.Module):
    def __init__(self, n_qubits: int, hidden_dim: int = 50, scale: int = 1):
        """
        Unified model that predicts either:
        - a complex matrix (2^N x 2^N) for small N (2 <= N <= 6), or
        - a vector of Hamiltonian coefficients (2N - 1) for large N (N >= 7).

        Args:
            n_qubits (int): Number of qubits.
            hidden_dim (int): Base hidden layer size (used for large N).
            scale (int): Width scaling factor (used for small N).
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.d = 2 ** n_qubits

        if n_qubits <= 6:
            # Full matrix prediction: (2^N x 2^N) complex
            output_dim = self.d * self.d * 2  # real + imag
            layers = [nn.Linear(1, 64 * scale), nn.Tanh()]
            layers += [nn.Linear(64 * scale, 128 * scale), nn.Tanh()]
            if n_qubits >= 4:
                layers += [nn.Linear(128 * scale, 256 * scale), nn.Tanh()]
            if n_qubits >= 5:
                layers += [nn.Linear(256 * scale, 512 * scale), nn.Tanh()]
            if n_qubits == 6:
                layers += [nn.Linear(512 * scale, 1024 * scale), nn.Tanh()]
            layers += [nn.Linear(layers[-2].out_features, output_dim)]
            self.net = nn.Sequential(*layers)
            self.mode = 'matrix'
        else:
            # Hamiltonian coefficient prediction: (2N - 1)
            output_dim = 2 * n_qubits - 1
            self.net = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, output_dim)
            )
            self.mode = 'coeffs'

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t (torch.Tensor): shape (B, 1), time input

        Returns:
            torch.Tensor:
                - (B, d, d) complex matrix if mode == 'matrix'
                - (B, 2N - 1) real vector if mode == 'coeffs'
        """
        x = self.net(t)

        if self.mode == 'matrix':
            x = x.view(-1, 2, self.d, self.d)
            return x[:, 0] + 1j * x[:, 1]  # complex matrix
        else:
            return x  # real vector of coefficients