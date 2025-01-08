import gymnasium as gym
from gymnasium.envs.classic_control import Continuous_MountainCarEnv
from typing import Optional
import numpy as np
from gymnasium.envs.classic_control import utils

class MountainCarFixPos(Continuous_MountainCarEnv):
    def __init__(self, render_mode: Optional[str] = None, goal_velocity=0, init_x = 0, x_limit = 0):
        super().__init__(render_mode, goal_velocity)
        self.init_x = init_x # -1.2 ~ 0.6
        self.x_limit = x_limit

    def set_init(self, init_x, x_limit):
        self.init_x = init_x
        self.x_limit = x_limit

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed, options=options)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        init_low = self.init_x - self.x_limit
        init_high = self.init_x + self.x_limit
        if options is not None:
            options.update({"low": init_low, "high": init_high})
        else:
            options = {"low": init_low, "high": init_high}
        low, high = utils.maybe_parse_reset_bounds(options, -0.6, -0.4)
        self.state = np.array([self.np_random.uniform(low=low, high=high), 0])

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}