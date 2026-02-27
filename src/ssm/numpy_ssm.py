"""
#WHERE
    Used by tests/test_modules.py (TestSSMCore), benchmarks, and
    any context where PyTorch GPU is not required.

#WHAT
    Pure-NumPy SSM implementation for testing and visualisation.
    Implements the classical discrete state-space recurrence:
        x_{t+1} = A x_t + B u_t
        y_t     = C x_t + D u_t

#INPUT
    sequence — numpy array of shape (length, d_input).

#OUTPUT
    numpy array of shape (length, d_output).
"""

import numpy as np


class SimpleSSMNumpy:
    """Lightweight SSM in NumPy — no PyTorch dependency."""

    def __init__(self, d_state: int = 16, d_input: int = 64, d_output: int = 64):
        self.d_state = d_state
        self.d_input = d_input
        self.d_output = d_output

        rng = np.random.default_rng(seed=42)
        self.A = np.eye(d_state) * 0.9 + rng.standard_normal((d_state, d_state)) * 0.01
        self.B = rng.standard_normal((d_state, d_input)) * 0.1
        self.C = rng.standard_normal((d_output, d_state)) * 0.1
        self.D = np.eye(d_output) if d_input == d_output else np.zeros((d_output, d_input))
        self.state = np.zeros(d_state)

    def reset(self):
        self.state = np.zeros(self.d_state)

    def step(self, u: np.ndarray) -> np.ndarray:
        """Single-step: (d_input,) → (d_output,)."""
        self.state = self.A @ self.state + self.B @ u
        return self.C @ self.state + self.D @ u

    def forward(self, sequence: np.ndarray) -> np.ndarray:
        """Full sequence: (length, d_input) → (length, d_output)."""
        self.reset()
        return np.array([self.step(sequence[t]) for t in range(len(sequence))])
