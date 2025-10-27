import gymnasium as gym
import numpy as np


class CustomEnvironmentWrapper(gym.Wrapper):
    """
    環境包裝器，用於組合state和額外輸入
    """
    def __init__(self, env, additional_input):
        super().__init__(env)
        self.additional_input= additional_input
        
        # 獲取原始observation space的維度
        original_obs_dim = env.observation_space.shape[0]
        
        # 獲取額外輸入的維度（假設是固定維度）
        additional_input_dim = len(self.additional_input)
        
        # 更新observation space
        new_obs_dim = original_obs_dim + additional_input_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(new_obs_dim,),
            dtype=np.float32
        )
        
        self.state_dim = original_obs_dim
        self.additional_input_dim = additional_input_dim
    
    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        combined_obs = np.concatenate([state, self.additional_input])
        return combined_obs, info
    
    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        combined_obs = np.concatenate([state, self.additional_input])
        return combined_obs, reward, terminated, truncated, info