from typing import List, Tuple, Dict

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
import argparse


import requests
import sys
import numpy as np
from utils.Ptime import Ptime
import os
from math import exp
import csv

from VWserver import SaveModelStrategy
from FERclient_FixState import FixPosClient

import torch
DEVICE = torch.device("cuda")  # Try "cuda" to train on GPU
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")

def create_client_fn(environment, value_weight, log_dir, time_step):
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
                            value_weight=value_weight,
                            environment=environment,
                            log_dir=log_dir,
                            time_step=time_step).to_client()
    return client_fn

def create_server_fn(save_dir: str, clients: int, total_rounds: int):
    def configured_server_fn(context: Context) -> ServerAppComponents:
        strategy = SaveModelStrategy(
            save_dir=save_dir,
            num_rounds=total_rounds,
            min_fit_clients=clients,
            min_evaluate_clients=clients,
            min_available_clients=clients,
        )
        config = ServerConfig(num_rounds=total_rounds)
        return ServerAppComponents(strategy=strategy, config=config)
    
    return configured_server_fn

def recursive_run():
    env_rounds = {"PendulumFixPos-v0": 20, "MountainCarFixPos-v0": 80, "CartPoleSwingUpFixInitState-v1": 80}
    clients_list = [5, 10, 20]
    value_weight_list = [0, 1]
    pass_list = [("PendulumFixPos-v0", 5, 0), ("PendulumFixPos-v0", 5, 1)]
    for clients in clients_list:
        for env, total_rounds in env_rounds.items():
            for value_weight in value_weight_list:
                if (env, clients, value_weight) in pass_list:
                    print("pass: ", env, clients, value_weight)
                    continue
                client_timesteps = 5e3
                time_str = Ptime()
                time_str.set_time_now()
                save_dir = f"multiagent/{time_str.get_time_to_minute()}_{clients}clients_{env}_VW{value_weight}"

                # Create the ClientApp
                client = ClientApp(client_fn=create_client_fn(env, value_weight, save_dir, client_timesteps))

                server = ServerApp(server_fn=create_server_fn(save_dir, clients, total_rounds))

                # Num of cpus are the same as in creating vec_envs in custom client.
                backend_config = {"client_resources": {"num_cpus": 8, "num_gpus": 0.0}}

                if DEVICE.type == "cuda":
                    backend_config = {"client_resources": {"num_cpus": 8, "num_gpus": 1.0}}
                send_line(f"Starting Lab: {clients}, {env}, {value_weight}")
                # Start Flower server
                # Run simulation
                run_simulation(
                    server_app=server,
                    client_app=client,
                    num_supernodes=clients,
                    backend_config=backend_config,
                )
                send_line('Lab done !')
    
    send_line('ALL Lab DONE !!!!!')
    sys.exit()

def main():

    clients = 5
    env = "MountainCarFixPos-v0"
    value_weight = 1
    total_rounds = 3
    client_timesteps = 3e3

    time_str = Ptime()
    time_str.set_time_now()
    save_dir = f"multiagent/{time_str.get_time_to_minute()}_{clients}clients_{env}_VW{value_weight}"

    # Create the ClientApp
    client = ClientApp(client_fn=create_client_fn(env, value_weight, save_dir, client_timesteps))

    server = ServerApp(server_fn=create_server_fn(save_dir, clients, total_rounds))

    backend_config = {"client_resources": {"num_cpus": 6, "num_gpus": 0.0}}

    if DEVICE.type == "cuda":
        backend_config = {"client_resources": {"num_cpus": 6, "num_gpus": 1.0}}
    send_line('Starting experiment')
    # Start Flower server
    # Run simulation
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=clients,
        backend_config=backend_config,
    )
    send_line('Experiment done !!!!!')
    sys.exit()
    
def send_line(message:str):
    token = '7ZPjzeQrRcI70yDFnhBd4A6xpU8MddE7MntCSdbLBgC'
    url = 'https://notify-api.line.me/api/notify'
    headers = {
        'Authorization': f'Bearer {token}'
    }
    data = {
        'message':message
    }
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        print("LINE message send sucessfuly")
    else:
        print("LINE message send error：", response.status_code)
    
if __name__ == "__main__":
    # main()
    recursive_run()
    
