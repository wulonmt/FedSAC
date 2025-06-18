import gymnasium as gym
import torch as th
from stable_baselines3 import SAC, PPO
from torch.distributions import Categorical
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv

import argparse
from datetime import datetime
import Env
from utils.init_pos_config import get_init_pos, is_valid_env, get_available_envs
from utils.CustomSAC import CustomSAC
from record import RewardDisplayWrapper
import cv2 #show the RewrdDisplayWrapper render

def train():
    n_cpu = 4
    batch_size = 64
    # env_name = "PendulumFixPos-v0"
    # env_name = "PendulumFixPos-v1"
    # env_name = "MountainCarFixPos-v0"
    # env_name = "MountainCarFixPos-v1"
    # env_name = "CartPoleSwingUpFixInitState-v1"
    env_name = "CartPoleSwingUpFixInitState-v2"
    # env_name = "HopperFixLength-v0"
    # env_name = "HalfCheetahFixLength-v0"
    assert is_valid_env(env_name), f"Only environments {', '.join(get_available_envs(env_name))} are available"
    index = 3
    #trained_env = GrayScale_env
    trained_env = make_vec_env(env_name, n_envs=n_cpu, vec_env_cls=SubprocVecEnv, seed = 1, env_kwargs = get_init_pos(env_name, index))
    tensorboard_log = f"./{env_name}_"

    #trained_env = make_vec_env(GrayScale_env, n_envs=n_cpu,)
    #env = gym.make("highway-fast-v0", render_mode="human")
    
    time_str = datetime.now().strftime("%Y%m%d%H%M")
    model = CustomSAC(
        "MlpPolicy",
        trained_env,
        batch_size=batch_size,
        learning_rate=5e-4,
        verbose=1,
        # ent_coef=0.2,
        tensorboard_log=tensorboard_log + time_str,
        device = "cuda:0",
        add_kl=False,
        kl_coef=0,
        n_envs=n_cpu
    )
    # model = SAC(
    #     "MlpPolicy",
    #     trained_env,
    #     batch_size=batch_size,
    #     learning_rate=5e-4,
    #     verbose=1,
    #     tensorboard_log=tensorboard_log + time_str,
    #     device = "cuda:0",
    # )
    # model = PPO("MlpPolicy",
    #             trained_env,
    #             policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    #             n_steps=batch_size * 12 // n_cpu,
    #             batch_size=batch_size,
    #             n_epochs=1,
    #             learning_rate=5e-4,
    #             gamma=0.8,
    #             verbose=1,
    #             target_kl=0.2,
    #             ent_coef=0.03,
    #             vf_coef=0.8,
    #             tensorboard_log=None)


    # Train the agent
    model.learn(total_timesteps=int(3e6), tb_log_name=time_str)
    # model.learn(total_timesteps=int(5e3 * 2), tb_log_name=time_str)
    print("log name: ", tensorboard_log + time_str)
    model.save(tensorboard_log + time_str + "/model")

    # model = CustomSAC.load(tensorboard_log + "model")
    # env = gym.make(env_name, render_mode="human")
    env = gym.make(env_name, render_mode="human", **get_init_pos(env_name, index))
    while True:
        obs, info = env.reset()
        done = truncated = False
        counter = 0
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            counter += 1
            if counter > 1000:
                break

fps = 30
delay = int(1000 / fps)
def show_reward_frame(window_name, img):
    if img is not None:
        # OpenCV 使用 BGR 格式，而 gymnasium 通常返回 RGB
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, img_bgr)
        
        # 等待 1ms，讓畫面更新，ESC 鍵退出
        key = cv2.waitKey(delay) & 0xFF

def eval(): 
    # model = SAC.load("multiagent\\2025_01_14_03_12_c5_PendulumFixPos-v0_VW0\\PendulumFixPos-v0\\0_entcoefauto_klcoef_0.0e+00_addKL_False_VW_False\\" + "model")
    model = SAC.load("MountainCarFixPos-v1_202506032236\\" + "model")
    env_name = "MountainCarFixPos-v1"

    assert is_valid_env(env_name), f"Only environments {', '.join(get_available_envs(env_name))} are available"
    index = 4
    # env = gym.make(env_name, render_mode="human")
    # env = gym.make(env_name, render_mode="human", **get_init_pos(env_name, index))

    #Show Reward at window
    env = gym.make(env_name, render_mode="rgb_array", **get_init_pos(env_name, index))
    env = RewardDisplayWrapper(env)

    while True:
        obs, info = env.reset()
        done = truncated = False

        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            # env.render()
            img = env.render()
            show_reward_frame(env_name, img)
            if done or truncated:
                print("done")
                cv2.waitKey(1000)  # 暫停 1 秒

if __name__ == "__main__":
    # train()
    eval()