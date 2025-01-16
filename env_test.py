import gymnasium as gym
import torch as th
from stable_baselines3 import SAC
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

def train():
    n_cpu = 3
    batch_size = 64
    # env_name = "PendulumFixPos-v0"
    # env_name = "MountainCarFixPos-v0"
    # env_name = "CartPoleSwingUpFixInitState-v1"
    env_name = "HopperFixLength-v0"
    assert is_valid_env(env_name), f"Only environments {', '.join(get_available_envs(env_name))} are available"
    index = 3
    #trained_env = GrayScale_env
    trained_env = make_vec_env(env_name, n_envs=n_cpu, vec_env_cls=SubprocVecEnv, seed = 1, env_kwargs = get_init_pos(env_name, index))
    tensorboard_log = "./"

    #trained_env = make_vec_env(GrayScale_env, n_envs=n_cpu,)
    #env = gym.make("highway-fast-v0", render_mode="human")
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
    # Train the agent
    model.learn(total_timesteps=int(5e3 * 50), tb_log_name=time_str)
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

def eval():
    model = SAC.load("SAC_CartPole/" + "model")
    env_name = "CartPoleSwingUpFixInitState-v1"
    assert is_valid_env(env_name), f"Only environments {', '.join(get_available_envs(env_name))} are available"
    index = 3
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

if __name__ == "__main__":
    train()
    # eval()