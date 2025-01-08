from typing import List, Tuple, Dict

import flwr as fl
from flwr.common import Metrics, FitIns
from flwr.common.typing import Parameters
from flwr.server.client_manager import ClientManager
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import GetParametersIns
from flwr.server.utils.tensorboard import tensorboard
from flwr.server.strategy import FedAvg, FedAdam
import argparse

import requests
import sys
import numpy as np
from utils.Ptime import Ptime
import os

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--port", help="local port", type=str, default="8080")
parser.add_argument("-r", "--rounds", help="total rounds", type=int, default=300)
parser.add_argument("-c", "--clients", help="number of clients", type=int, default=2)
args = parser.parse_args()

def save_model(parameters: Parameters, save_dir: str, filename: str = "final_model.npz"):
    """Save the model parameters to a file in the specified directory."""
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Create the full file path
    file_path = os.path.join(save_dir, filename)
    
    # Convert parameters to numpy arrays
    ndarrays = parameters_to_ndarrays(parameters)
    
    # Save the numpy arrays to a file
    np.savez(file_path, *ndarrays)
    print(f"Model saved to {file_path}")

def aggregate_log_std(metrics: List[Tuple[int, Metrics]]) -> Dict[str, float]:
    total_samples = sum([num_samples for num_samples, _ in metrics])
    weighted_special_param = sum(
        [num_samples * m["log_std"] for num_samples, m in metrics]
    )
    return {"log_std": weighted_special_param / total_samples}

class SaveModelStrategy(FedAvg):
    def __init__(self, save_dir: str, num_rounds: int, **kwargs):
        super().__init__(**kwargs)
        self.save_dir = save_dir
        self.num_rounds = num_rounds
        self.aggregated_log_std = 0.0

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Parameters, dict]:
        # Aggregate parameters and metrics using the super class method
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        # Save the model after the final round
        if server_round == self.num_rounds:
            save_model(parameters, self.save_dir, f"final_model_round_{server_round}.npz")

        return parameters, metrics

def main():
    total_rounds = args.rounds
    clients = args.clients
    print(f"Starting Server, total rounds {total_rounds}, clients {clients}")
    time_str = Ptime()
    time_str.set_time_now()
    save_dir = "multiagent/" + time_str.get_time_to_minute()
    # Decorated strategy
    strategy = SaveModelStrategy(
        save_dir=save_dir,
        num_rounds=total_rounds,
        min_fit_clients=clients,
        min_evaluate_clients=clients,
        min_available_clients=clients,
    )

    send_line('Starting experiment')
    # Start Flower server
    flwr_server = fl.server
    flwr_server.start_server(
        server_address="127.0.0.1:" + args.port,
        config=fl.server.ServerConfig(num_rounds=total_rounds),
        strategy=strategy,
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
    main()
    
