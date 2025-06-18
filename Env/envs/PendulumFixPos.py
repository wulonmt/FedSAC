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
            # high = np.array([self.init_theta, self.init_thetadot])
            x = self.init_theta
            y = self.init_thetadot
        else:
            # Note that if you use custom reset bounds, it may lead to out-of-bound
            # state/observations.
            x = options.get("x_init") if "x_init" in options else self.init_theta
            y = options.get("y_init") if "y_init" in options else self.init_thetadot
            x = utils.verify_number_and_cast(x)
            y = utils.verify_number_and_cast(y)

        if x == -1: # stand for random init theta
            x = self.random_inti_theta()

        if y == -1:
            y = 1

        high = np.array([x, y])
        low = -high  # We enforce symmetric limits.
        low[0] = high[0]
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}
    
    def random_inti_theta(self):
        # 定義兩個目標區間的上下限
        # 區間一: [-np.pi, -np.pi/2)
        interval1_low = -np.pi
        interval1_high = -np.pi / 2

        # 區間二: [np.pi/2, np.pi)
        interval2_low = np.pi / 2
        interval2_high = np.pi

        # 隨機選擇要從哪個區間產生數字
        # np.random.choice([0, 1]) 會隨機返回 0 或 1，代表選擇區間一或區間二
        chosen_interval_index = self.np_random.choice([0, 1])

        # 根據選擇的區間生成隨機數
        if chosen_interval_index == 0:
            # 從第一個區間生成隨機數
            return self.np_random.uniform(interval1_low, interval1_high)
        else:  # chosen_interval_index == 1
            # 從第二個區間生成隨機數
            return self.np_random.uniform(interval2_low, interval2_high)
    
class PendulumFixPosGoalOriented(PendulumFixPos):
    def __init__(self, render_mode = None, goal_velocity=0, init_theta=np.pi, init_thetadot=1, total_stable_steps = 50, reward_threshold = -0.2):
        super().__init__(render_mode, goal_velocity, init_theta, init_thetadot)
        self.total_stable_steps = total_stable_steps
        self.current_stable_steps = 0
        self.reward_threshold = reward_threshold

    def step(self, u):
        obs, old_reward, terminal, truncated, _ = super().step(u) # -16.27 < old_reward < 0
        
        if old_reward > self.reward_threshold:
            self.current_stable_steps += 1
        else:
            self.current_stable_steps = 0
        
        new_reward = 0
        if self.current_stable_steps > self.total_stable_steps:
            terminal = True
            new_reward = 1

        return obs, new_reward, terminal, truncated, {}
    
    def reset(self, *, seed = None, options = None):
        self.current_stable_steps = 0
        return super().reset(seed=seed, options=options)
