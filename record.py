import gymnasium as gym
from matplotlib import pyplot as plt
import argparse
import Env
from utils.CustomPPO import CustomPPO
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from utils.init_pos_config import get_init_pos, get_init_list, assert_alarm

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--log_model", help="modle to be logged", type=str)
parser.add_argument("-e", "--environment", help="which my- env been used", type=str, required=True)
args = parser.parse_args()
RECORD = True
SNAPSHOT = False
EVALUATE = False
DISPLAY = False

if __name__ == "__main__":
    index = 3
    assert_alarm(args.environment)
    env_name = args.environment
    log_env = args.log_model.split('/')[0].split('_')[0]
    rgb_env = gym.make(env_name, render_mode="rgb_array", **get_init_pos(env_name, index))
    human_env = gym.make(env_name, render_mode="rgb_array", **get_init_pos(env_name, index))
    rgb_env.reset()
    human_env.reset()

    model = PPO.load(args.log_model+"model")
    
    if RECORD:
        video_length = 10000
        for i in range(len(get_init_list(env_name))):
            vec_env = DummyVecEnv([lambda: gym.make(env_name, render_mode="rgb_array", **get_init_pos(env_name, i))])
            obs = vec_env.reset()
            vec_env = VecVideoRecorder(vec_env, args.log_model + "/videos",
                        record_video_trigger=lambda x: x == 0,
                        name_prefix=f"env_index_{i}")
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
        rgb_env.reset()
        for _ in range(3):
            obs, reward, done, truncated, info = rgb_env.step(rgb_env.action_type.actions_indexes["IDLE"])

            fig, axes = plt.subplots(ncols=4, figsize=(12, 5))
            for i, ax in enumerate(axes.flat):
                ax.imshow(obs[i, ...].T, cmap=plt.get_cmap('gray'))
        plt.show()