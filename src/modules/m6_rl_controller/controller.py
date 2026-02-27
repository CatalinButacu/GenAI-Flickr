"""
#WHERE
    Imported by m6_rl_controller/__init__.py.

#WHAT
    PPO-based RL controller for humanoid motion (stub).
    Plan: wrap M5 Scene as gymnasium.Env, train with stable-baselines3 PPO.

#INPUT
    Observation array from M5 physics environment.

#OUTPUT
    Action vector (joint torques / position targets).
"""


class RLController:
    def setup(self) -> None:
        raise NotImplementedError("M6 RLController not yet implemented")

    def act(self, observation):
        raise NotImplementedError("M6 RLController not yet implemented")
