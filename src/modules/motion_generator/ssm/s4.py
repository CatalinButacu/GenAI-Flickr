"""
#WHERE
    Used by mamba.py, temporal.py, and factory.py.
    Indirect user: src/modules/motion_generator (via factory).

#WHAT
    S4 (Structured State Space Sequence) layer implementation.
    A continuous-time SSM discretised with bilinear/ZOH for digital processing.
    Uses HiPPO matrix initialisation for optimal long-range memory.

#INPUT
    Tensor of shape (batch, length, d_model) — a sequence of feature vectors.

#OUTPUT
    Tensor of shape (batch, length, d_model) — same shape, SSM-transformed.

References:
    [1] S4: "Efficiently Modeling Long Sequences with Structured State Spaces"
        Gu et al., ICLR 2022  —  https://arxiv.org/abs/2111.00396
    [2] HiPPO: "Recurrent Memory with Optimal Polynomial Projections"
        Gu et al., NeurIPS 2020  —  https://arxiv.org/abs/2008.07669
"""

import math

import torch
import torch.nn as nn

from .config import SSMConfig  # noqa: F401 — re-exported for convenience


class S4Layer(nn.Module):
    """
    Simplified S4 (Structured State Space Sequence) layer.

    Implements the continuous-time SSM:
        x'(t) = Ax(t) + Bu(t)
        y(t)  = Cx(t) + Du(t)

    Discretised via Euler method.  A is initialised with HiPPO-LegS.

    Formula (HiPPO, Equation 9):
        A_hippo[n,k] = -(2n+1)^{1/2} (2k+1)^{1/2}  if n > k
                     = -(n+1)                         if n = k
                     = 0                               if n < k
    """

    def __init__(self, d_model: int, d_state: int = 64, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # State-space matrices (learned)
        self.A = nn.Parameter(torch.randn(d_state, d_state) * 0.01)
        self.B = nn.Parameter(torch.randn(d_state, d_model) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.D = nn.Parameter(torch.ones(d_model))  # skip connection

        # Learnable discretisation step size
        log_dt = (
            torch.rand(d_model) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        )
        self.log_dt = nn.Parameter(log_dt)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._init_hippo()

    # ── HiPPO initialisation ────────────────────────────────────────────────

    def _init_hippo(self):
        """Initialise A with HiPPO-LegS structure (optimal polynomial memory)."""
        n_dim = self.d_state
        a_mat = torch.zeros(n_dim, n_dim)
        for n in range(n_dim):
            for k in range(n_dim):
                if n > k:
                    a_mat[n, k] = -math.sqrt(2 * n + 1) * math.sqrt(2 * k + 1)
                elif n == k:
                    a_mat[n, k] = -(n + 1)
        self.A.data = a_mat * 0.5

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (batch, length, d_model)
        Returns:
            (batch, length, d_model)
        """
        _, length, _ = u.shape
        dt = torch.exp(self.log_dt)

        # Euler discretisation
        a_bar = torch.eye(self.d_state, device=u.device) + dt.mean() * self.A
        b_bar = dt.mean() * self.B

        # Recurrence: x_t = A_bar x_{t-1} + B_bar u_t, y_t = C x_t + D u_t
        x = torch.zeros(u.shape[0], self.d_state, device=u.device)
        outputs = []
        for t in range(length):
            x = x @ a_bar.T + u[:, t, :] @ b_bar.T
            y = x @ self.C.T + u[:, t, :] * self.D
            outputs.append(y)

        return self.dropout(torch.stack(outputs, dim=1))
