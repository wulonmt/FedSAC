import gymnasium as gym
import argparse
import Env
from stable_baselines3 import PPO
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
import os
import csv
from utils.init_pos_config import get_init_list, get_param_names, assert_alarm
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--log_model", help="model to be logged", type=str) # <log_model>/0_...
parser.add_argument("-e", "--environment", help="which my- env been used", type=str, required=True)
parser.add_argument("-p", "--prefix", type=str, default="", help="prefix_name")
parser.add_argument("-s", "--save_dir", help="directory to save plots", type=str, required=True)
args = parser.parse_args()

def round_floats(value):
    return round(value, 2) if isinstance(value, float) else value

def write_results(writer, env_name, dir_name, para_dict, reward_mean, reward_std):
    param_names = get_param_names(env_name)
    
    row_data = {
        'agent': dir_name.split("_")[0],
        'reward_mean': round_floats(reward_mean),
        'reward_std': round_floats(reward_std)
    }
    
    # 添加環境特定的參數
    for param_name in param_names:
        row_data[param_name] = round_floats(para_dict[param_name])
    
    writer.writerow(row_data)

if __name__ == "__main__":
    assert_alarm(args.environment)
    env_name = args.environment
    folder_path = [f.path for f in os.scandir(args.log_model) if f.is_dir()]
    
    # Open a CSV file to write the results
    with open(args.save_dir + f"/{args.prefix}_evaluation_results.csv", 'w', newline='') as csvfile:
        fieldnames = ['agent', 'reward_mean', 'reward_std'] + get_param_names(env_name)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        eval_env_mean = defaultdict(list)
        eval_env_std = defaultdict(list)

        for path in folder_path:
            model = PPO.load(path+"/model")
            dir_name = path.split("\\")[-1]
            print(f"{dir_name = }")
            for index, para_dict in enumerate(get_init_list(env_name)):
                env = gym.make(env_name, render_mode="rgb_array", **para_dict)
                env.reset()
                reward_mean, reward_std = evaluate_policy(model, env)
                print(f"{para_dict = }, {reward_mean = }, {reward_std = }")
                write_results(writer, env_name, dir_name, para_dict, reward_mean, reward_std)
                eval_env_mean[index].append(reward_mean)
                eval_env_std[index].append(reward_std)

    AVG = lambda x: sum(x)/len(x)
    with open(args.save_dir + f"/{args.prefix}_evaluation_averages.csv", 'w', newline='') as csvfile:
        fieldnames = ['env_index', 'avg_reward', 'avg_std']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for env, (mean, std) in enumerate(zip(eval_env_mean.values(), eval_env_std.values())):
            avg_reward = AVG(mean)
            avg_std = AVG(std)
            print(f"env {env}, reward {avg_reward}, std {avg_std}")
            
            writer.writerow({
                'env_index': env,
                'avg_reward': round_floats(avg_reward),
                'avg_std': round_floats(avg_std)
            })
    print("Results have been saved to evaluation_results.csv")