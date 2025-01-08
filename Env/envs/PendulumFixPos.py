import gymnasium as gym
from gymnasium.envs.classic_control import PendulumEnv
from typing import Optional
import numpy as np
from gymnasium.envs.classic_control import utils

class PendulumFixPos(PendulumEnv):
    def __init__(self, render_mode: Optional[str] = None, goal_velocity=0, init_theta = np.pi, init_thetadot = 1.0):
        super().__init__(render_mode, goal_velocity)
        self.init_theta = init_theta # 0:↑, np.pi/2:←, np.pi: ↓, -np.pi/2: →
        self.init_thetadot = init_thetadot #-8 ~ 8

    def set_init(self, init_theta, init_thetadot):
        self.init_theta= init_theta
        self.init_thetadot = init_thetadot

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if options is None:
            high = np.array([self.init_theta, self.init_thetadot])
        else:
            # Note that if you use custom reset bounds, it may lead to out-of-bound
            # state/observations.
            x = options.get("x_init") if "x_init" in options else self.init_theta
            y = options.get("y_init") if "y_init" in options else self.init_thetadot
            x = utils.verify_number_and_cast(x)
            y = utils.verify_number_and_cast(y)
            high = np.array([x, y])
        low = -high  # We enforce symmetric limits.
        low[0] = high[0]
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}