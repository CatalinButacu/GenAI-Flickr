"""
#WHERE
    Imported by pipeline.py (stub — not yet used in production).

#WHAT
    RL Controller Module (Module 6) — PPO-based humanoid motion control.
    Currently a stub; planned: gymnasium.Env wrapping M5 Scene.

#INPUT
    Observation from M5 physics env.

#OUTPUT
    Action vector for humanoid joint torques.
"""

from .controller import RLController

__all__ = ["RLController"]
