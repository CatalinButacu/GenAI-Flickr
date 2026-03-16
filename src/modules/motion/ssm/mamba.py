
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SSMConfig


class MambaLayer(nn.Module):
    """
    Simplified Mamba (Selective SSM) layer.

    Equation (from paper):
        x_t = A_bar x_{t-1} + B_bar(u_t) u_t     # B depends on input
        y_t = C(u_t) x_t                          # C depends on input
    Where:
        A_bar = exp(delta(u_t) * A)                # delta depends on input
        B_bar = delta(u_t) * B(u_t)
    """

    def __init__(self, config: SSMConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model
        d_state = config.d_state
        d_inner = config.d_inner
        self._dt_rank = config.dt_rank if isinstance(config.dt_rank, int) else d_model // 16

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=d_inner, out_channels=d_inner,
            kernel_size=config.d_conv, padding=config.d_conv - 1,
            groups=d_inner,
        )

        # A is diagonal and NOT input-dependent (core SSM parameter)
        a_init = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(a_init))
        self.D = nn.Parameter(torch.ones(d_inner))

        # Input-dependent projections (the "selection" mechanism)
        self.x_proj = nn.Linear(d_inner, self._dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self._dt_rank, d_inner, bias=True)

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        self._init_dt()

    def _init_dt(self):
        """Special initialisation for delta = softplus(Linear(x))."""
        dt_init_std = self._dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.config.d_inner)
            * (math.log(self.config.dt_max) - math.log(self.config.dt_min))
            + math.log(self.config.dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_proj.bias.data = inv_dt

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, length, d_model) → (batch, length, d_model)."""
        _, length, _ = x.shape

        # Split into two gated paths
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # Causal convolution + SiLU
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :length]
        x = x.transpose(1, 2)
        x = F.silu(x)

        y = self._ssm_forward(x)

        z = F.silu(z)
        return self.out_proj(y * z)

    def _ssm_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Selective scan: B, C, delta all depend on input x."""
        batch, length, d_inner = x.shape
        d_state = self.config.d_state

        # Input-dependent projections
        x_dbl = self.x_proj(x)
        dt, b_sel, c_sel = x_dbl.split([self._dt_rank, d_state, d_state], dim=-1)

        dt = F.softplus(self.dt_proj(dt))       # (batch, length, d_inner)
        a_diag = -torch.exp(self.A_log)         # (d_state,)

        # Recurrent scan: h_t = A_bar * h_{t-1} + B_bar * x_t; y_t = C * h_t + D * x_t
        h = torch.zeros(batch, d_inner, d_state, device=x.device)
        outputs = []
        for t in range(length):
            dt_t = dt[:, t, :, None]
            a_bar = torch.exp(dt_t * a_diag)
            b_bar = dt_t * b_sel[:, t, None, :]
            h = a_bar * h + b_bar * x[:, t, :, None]
            y = (h * c_sel[:, t, None, :]).sum(dim=-1) + self.D * x[:, t, :]
            outputs.append(y)

        return torch.stack(outputs, dim=1)
