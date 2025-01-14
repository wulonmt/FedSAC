import gymnasium as gym
from matplotlib import pyplot as plt
import argparse
import Env
from stable_baselines3 import SAC
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from utils.init_pos_config import get_init_pos, get_init_list, assert_alarm

import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--log_model", help="modle to be logged", type=str)
parser.add_argument("-e", "--environment", help="which my- env been used", type=str, required=True)
args = parser.parse_args()
RECORD = False
SNAPSHOT = True
EVALUATE = False
DISPLAY = False

class RewardDisplayWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.cumulative_reward = 0
        
    def reset(self, **kwargs):
        self.cumulative_reward = 0
        return super().reset(**kwargs)
        
    def step(self, action):
        result = super().step(action)
        # print(result)
        obs, reward, done, truncated, info = result
        self.cumulative_reward += reward
        if done:
            self.cumulative_reward = 0
        return obs, reward, done, truncated, info
        
    def render(self, mode='rgb_array'):
        # 獲取原始圖像
        img = super().render()
        
        # 在圖像上添加文字
        img = np.ascontiguousarray(img, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f'Reward: {self.cumulative_reward:.2f}'
        
        # 獲取文字大小以調整位置
        (text_width, text_height), _ = cv2.getTextSize(text, font, 0.5, 1)
        
        # 添加半透明背景以提高文字可讀性
        overlay = img.copy()
        cv2.rectangle(
            overlay, 
            (5, 5), 
            (10 + text_width, 10 + text_height), 
            (0, 0, 0), 
            -1
        )
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        
        # 添加文字
        cv2.putText(
            img, 
            text, 
            (10, 20), 
            font, 
            0.5, 
            (255, 255, 255), 
            1, 
            cv2.LINE_AA
        )
        
        return img
    
def make_wrapped_env(env_name, i=0):
    def _init():
        env = gym.make(env_name, render_mode="rgb_array", **get_init_pos(env_name, i))
        return RewardDisplayWrapper(env)
    return _init


if __name__ == "__main__":
    index = 4
    assert_alarm(args.environment)
    env_name = args.environment
    log_env = args.log_model.split('/')[0].split('_')[0]
    rgb_env = gym.make(env_name, render_mode="rgb_array", **get_init_pos(env_name, index))
    human_env = gym.make(env_name, render_mode="rgb_array", **get_init_pos(env_name, index))
    rgb_env.reset()
    human_env.reset()

    model = SAC.load(args.log_model+"model")
    
    if RECORD:
        video_length = 1000
        for i in range(len(get_init_list(env_name))):
            vec_env = DummyVecEnv([make_wrapped_env(env_name, i=i)])
            obs = vec_env.reset()
            vec_env = VecVideoRecorder(vec_env, args.log_model + "videos",
                        record_video_trigger=lambda x: x == 0,
                        name_prefix=f"env_index_{i}",
                        video_length=video_length,)
            vec_env.reset()
            for _ in range(video_length + 1):
                action, _states = model.predict(obs, deterministic=True)
                obs, _, _, _ = vec_env.step(action)
            vec_env.close()
    
    if EVALUATE:
        reward_mean, reward_std = evaluate_policy(model, rgb_env)
        print(f"{reward_mean = }, {reward_std = }")
    
    # for _ in range(10):
    #     obs, info = env.reset()
    #     done = truncated = False
    #     while not (done or truncated):
    #         action, _ = model.predict(obs)
    #         obs, reward, done, truncated, info = env.step(action)
    #         env.render()
    if DISPLAY:
        for _ in range(10):
            obs, info = human_env.reset()
            done = truncated = False
            counter = 0
            while not (done or truncated):
                action, _ = model.predict(obs)
                obs, reward, done, truncated, info = human_env.step(action)
                human_env.render()
                counter += 1
                if counter > 100:
                    break
    
    if SNAPSHOT:
        print(f"{rgb_env.action_space = }")
        rgb_env.reset()
        for _ in range(3):
            obs, reward, done, truncated, info = rgb_env.step([0.5])

            fig, axes = plt.subplots(ncols=4, figsize=(12, 5))
            for i, ax in enumerate(axes.flat):
                ax.imshow(obs[i, ...].T, cmap=plt.get_cmap('gray'))
        plt.show()