import gymnasium as gym
import torch as th
from torch.distributions import Categorical
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import argparse
from utils.Ptime import Ptime

import flwr as fl
from collections import OrderedDict
import os
import sys
# import Env
from utils.CustomSAC import CustomSAC
from utils.init_pos_config import get_init_pos, assert_alarm

from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents

def paser_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log_name", help="modified log name", type=str, default ="auto")
    parser.add_argument("-s", "--save_log", help="whether save log or not", type=str, default = "True") #parser can't pass bool
    parser.add_argument("-e", "--environment", help="which my- env been used", type=str, required=True)
    # parser.add_argument("-e", "--environment", help="which my- env been used", type=str, default="Pendulum-v0")
    parser.add_argument("-t", "--train", help="training or not", type=str, default = "True")
    parser.add_argument("-r", "--render_mode", help="h for human & r for rgb_array", type=str, default = "r")
    parser.add_argument("-i", "--index", help="client index", type=int, default = 0, required = True)
    parser.add_argument("-p", "--port", help="local port", type=str, default="8080")
    parser.add_argument("-m", "--time_step", help="training time steps", type=int, default=5e3)
    parser.add_argument("--kl_coef", help="kl divergence coef", type=float, default=0)
    parser.add_argument("--ent_coef", help="entropy coef", type=float, default=0)
    parser.add_argument("--value_weight", help="value weighted, 1 for true and 0 for false",  type=int, default=0)
    parser.add_argument("--add_kl", help="KLD regilization add, 1 for true and 0 for false",  type=int, default=0)
    parser.add_argument("--log_dir", help="server & client log dir", type=str, default = None)

    return parser.parse_args()

# MountainCarFixPos-v0, PendulumFixPos-v0, CartPoleSwingUpFixInitState-v1 are available

class FixPosClient(fl.client.NumPyClient):
    def __init__(self,
                client_index,
                value_weight,
                environment,
                log_dir,
                time_step,
                log_name="auto",
                save_log="True",
                add_kl=0,
                ent_coef=None,
                kl_coef=None):
        n_cpu = 8
        batch_size = 64
        add_kl = True if add_kl == 1 else False
        value_as_weight = True if value_weight == 1 else False
        #self.env = gym.make(f"my-{environment}-v0", render_mode=rm)
        # self.env = make_vec_env(f"{environment}", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
        if(client_index < 4):
            self.env = make_vec_env(f"{environment}", n_envs=n_cpu, vec_env_cls=SubprocVecEnv, env_kwargs = get_init_pos(environment, client_index))
            # self.env = make_vec_env(f"{environment}", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
        else:
            self.env = make_vec_env(f"{environment}", n_envs=n_cpu, vec_env_cls=SubprocVecEnv, env_kwargs = get_init_pos(environment, client_index))
            # self.env = make_vec_env(f"{.environment}", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
        self.tensorboard_log=f"{environment}/" if save_log == "True" else None
        time_str = Ptime()
        time_str.set_time_now()
        if save_log == "True":
            self.tensorboard_log = f"multiagent/{time_str.get_time_to_minute()}_{environment}_VW_{value_as_weight}/{self.tensorboard_log}"
        self.tensorboard_log = log_dir + f"/{environment}/" if log_dir else self.tensorboard_log
        trained_env = self.env
        self.model = CustomSAC("MlpPolicy",
                    trained_env,
                    batch_size=batch_size,
                    learning_rate=5e-4,
                    verbose=1,
                    # ent_coef=ent_coef,
                    tensorboard_log=self.tensorboard_log,
                    device = "cuda:0",
                    # add_kl=add_kl,
                    # kl_coef=kl_coef
                    )
        # policy_kwargs=dict(net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256])),

        self.n_round = int(0)
        
        if save_log == "True":
            description = log_name if log_name != "auto" else \
                        f"entcoef{self.model.ent_coef}_klcoef_{self.model.kl_coef:.1e}_addKL_{add_kl}_VW_{value_as_weight}"
            self.log_name = f"{client_index}_{description}"
        else:
            self.log_name = None

        self.value_as_weight = value_as_weight
        self.time_step = time_step
        self.save_log = save_log
        self.client_index = client_index
        
        
    def get_parameters(self, config):
        # print(self.model.policy)
        # print(self.model.policy.state_dict().keys())
        # print([key for key, value in self.model.policy.state_dict().items() if "policy_net" in key])
        policy_state = [value.cpu().numpy() for key, value in self.model.policy.state_dict().items()]
        policy_state.append(self.model.log_ent_coef.cpu().detach().numpy())
        # policy_state = [value.cpu().numpy() for key, value in self.model.policy.state_dict().items() if "policy_net" in key]
        return policy_state

    def set_parameters(self, parameters):
        # -----specific key-----
        # parameters = [th.tensor(v) for v in parameters]
        # features_extractor_keys = [key for key in self.model.policy.state_dict().keys() if "policy_net" in key]
        # params_dict = zip(features_extractor_keys, parameters)
        # state_dict = self.model.policy.state_dict()
        # state_dict.update(params_dict)
        # -----all parameters-----
        # params_dict = zip(self.model.policy.state_dict().keys(), parameters)
        # add log_ent_coef
        params_dict = zip(self.model.policy.state_dict().keys(), parameters[:-1])
        state_dict = OrderedDict({k: th.tensor(v) for k, v in params_dict})
        # -----set to policy-----
        self.model.policy.load_state_dict(state_dict, strict=True)
        self.model.regul_policy.load_state_dict(state_dict, strict=True)
        # set log_ent_coef
        init_value = np.exp(parameters[-1])
        print("inti value: ")
        print(init_value)
        self.model.log_ent_coef = th.log(th.ones(1, device=self.model.device) * float(init_value[0])).requires_grad_(True)
        self.model.ent_coef_optimizer = th.optim.Adam([self.model.log_ent_coef], lr=self.model.lr_schedule(1))

    def fit(self, parameters, config):
        print(f"[Client {self.client_index}] fit, config: {config}")
        self.n_round += 1
        if "server_round" in config.keys():
            self.n_round = config["server_round"]
            self.model.n_rounds = self.n_round
            # self.time_step * (self.n_round - 1) + self.time_step
            self.model.num_timesteps = self.time_step * (self.n_round - 1)
            # self.time_step = self.time_step * self.n_round

        self.set_parameters(parameters)
        if("learning_rate" in config.keys()):
            self.model.learning_rate = config["learning_rate"]
        print(f"Training learning rate: {self.model.learning_rate}")
        # Train the agent
        self.model.learn(total_timesteps=self.time_step,
                         tb_log_name=(self.log_name + f"/round_{self.n_round:0>2d}") if self.log_name is not None else None ,
                         reset_num_timesteps=False,
                         )
        # Save the agent
        if self.save_log == "True":
            print("log name: ", self.tensorboard_log + self.log_name)
            self.model.save(self.tensorboard_log + self.log_name + "/model")

        # value may be negtive, so using soft-max
        R_list = self.model.last_R
        print(f"{R_list = }")
        merge_weight = int(sum(R_list)/len(R_list)) if self.value_as_weight else int(max(self.model.num_timesteps, 1))
            
        return self.get_parameters(config={}), merge_weight, {}

    def evaluate(self, parameters, config):
        print("evaluating model")
        self.set_parameters(parameters)
        reward_mean, reward_std = evaluate_policy(self.model, self.env)

        return -reward_mean, 1, {"reward mean": reward_mean, "reward std": reward_std}

def client_fn(context: Context) -> Client:
        """Create a Flower client representing a single organization."""

        partition_id = context.node_config["partition-id"]
        print("___________________________")
        print("node config:", context.node_config)
        print("run id:", context.run_id)
        print("node id:", context.node_id)
        print("state:", context.state)
        print("run config:", context.run_config)
        print("___________________________")
        return FixPosClient(client_index=partition_id,
                            value_weight=context.run_config["value_weight"],
                            environment=context.run_config['environment'],
                            log_dir=context.run_config['save_dir'],
                            time_step=context.run_config['time_step']).to_client()

# Flower ClientApp
app = ClientApp(client_fn)

def main():
    args = paser_argument()
    assert_alarm(args.environment)

    # Start Flower client
    #port = 8080 + args.index
    client = FixPosClient(client_index=args.index,
                          add_kl=args.add_kl,
                          value_weight=args.value_weight,
                          ent_coef=args.ent_coef,
                          kl_coef=args.kl_coef,
                          environment=args.environment,
                          save_log=args.save_log,
                          log_dir=args.log_dir,
                          log_name=args.log_name,
                          time_step=args.time_step,
                          )
    fl.client.start_client(
        server_address=f"127.0.0.1:" + args.port,
        client=client.to_client(),
    )
    # sys.exit()

    if args.index < 4:
        env = gym.make(args.environment, render_mode="human", **get_init_pos(args.environment, args.index))
    else:
        env = gym.make(args.environment, render_mode="human", **get_init_pos(args.environment, 4))
    # env = gym.make(args.environment, render_mode="human")

    while True:
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = client.model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            env.render()

if __name__ == "__main__":
    main()
    # test()

