import random

from highway_env.envs.highway_env import HighwayEnvFast, HighwayEnv
from highway_env.envs.common.action import Action

class CrowdedHighway(HighwayEnvFast):
    
    def __init__(self, render_mode: str | None = None, density = 2, count = 50) -> None:
        self.density = density
        self.count = count
        if density == count == -1:
            self.count = 50
            self.density = random.uniform(0.1, 3)
        config = self._create_config()
        super().__init__(config = config, render_mode = render_mode)

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {"observation": {
                       "type": "GrayscaleObservation",
                       "observation_shape": (84, 84),
                       "stack_size": 4,
                       "weights": [0.9, 0.1, 0.5],  # weights for RGB conversion
                       "scaling": 1.75,
                   },
            "action": {
                    "type": "ContinuousAction",
                },
            "collision_reward": -1,
            "right_lane_reward": 0.1,
            "high_speed_reward": 0.4,
            "merging_speed_reward": -0.5,
            "lane_change_reward": 0,
            "vehicles_density": 2,
            "vehicles_count": 50,
            "distance": 1000,
            })
        return cfg
    
    def _create_config(self) -> dict:
        """根據實例參數創建配置"""
        cfg = self.default_config()
        cfg["vehicles_density"] = self.density
        cfg["vehicles_count"] = self.count
        return cfg
    
    def _reward(self, action: Action) -> float:
        if self.vehicle.position[0] > self.config["distance"]:
            return 1
        else:
            return 0
            
    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (
            self.vehicle.crashed
            or self.config["offroad_terminal"]
            and not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.vehicle.position[0] > self.config["distance"]
    
class CrowdedHighway_V1(HighwayEnvFast):
    
    def __init__(self, render_mode: str | None = None, density = 2, count = 50) -> None:
        self.density = density
        self.count = count
        if density == count == -1:
            self.count = 50
            self.density = random.uniform(0.1, 3)
        config = self._create_config()
        super().__init__(config = config, render_mode = render_mode)

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {"observation": {
                       "type": "GrayscaleObservation",
                       "observation_shape": (84, 84),
                       "stack_size": 4,
                       "weights": [0.9, 0.1, 0.5],  # weights for RGB conversion
                       "scaling": 1.75,
                   },
            "action": {
                    "type": "ContinuousAction",
                },
            "collision_reward": -1,
            "right_lane_reward": 0.1,
            "high_speed_reward": 0.4,
            "merging_speed_reward": -0.5,
            "lane_change_reward": 0,
            "vehicles_density": 2,
            "vehicles_count": 50,
            "distance": 1000,
            })
        return cfg
    
    def _create_config(self) -> dict:
        """根據實例參數創建配置"""
        cfg = self.default_config()
        cfg["vehicles_density"] = self.density
        cfg["vehicles_count"] = self.count
        return cfg
            
    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (
            self.vehicle.crashed
            or self.config["offroad_terminal"]
            and not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.vehicle.position[0] > self.config["distance"]