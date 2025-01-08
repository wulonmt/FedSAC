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
from logging import WARNING
from typing import Callable, Optional, Union
from flwr.server.client_proxy import ClientProxy

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    NDArray,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log

from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents

from flwr.server.strategy.aggregate import aggregate, aggregate_inplace
from functools import partial, reduce

import requests
import sys
import numpy as np
from utils.Ptime import Ptime
import os
from math import exp
import csv

def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", help="local port", type=str, default="8080")
    parser.add_argument("-r", "--rounds", help="total rounds", type=int, default=300)
    parser.add_argument("-c", "--clients", help="number of clients", type=int, default=2)
    parser.add_argument("--log_dir", help="server & client log dir", type=str, default = None)

    return parser.parse_args()

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
        self.client_weights = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Parameters, dict]:
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if self.inplace:
            print("________inplace_______")
            # Does in-place weighted average of results
            aggregated_ndarrays = self.VW_aggregate_inplace(results)
        else:
            print("________out place_______")
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)

        parameters = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
        
        # Save the model after the final round
        if server_round == self.num_rounds:
            save_model(parameters, self.save_dir, f"final_model_round_{server_round}.npz")
            with open(self.save_dir + '/client_weights.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(self.client_weights)

        return parameters, metrics
    
    def VW_aggregate_inplace(self, results: list[tuple[ClientProxy, FitRes]]) -> NDArrays:
        """Compute in-place RL-value weighted soft-max average."""
        all_values = [fit_res.num_examples for (_, fit_res) in results]
        max_value = max(all_values)
        values_exp_sum = sum([exp(value - max_value) for value in all_values])
        multiple_weights = [exp(value - max_value) / values_exp_sum for value in all_values]
        self.client_weights.append(all_values + multiple_weights)

        scaling_factors = np.asarray(multiple_weights)

        def _try_inplace(
            x: NDArray, y: Union[NDArray, np.float64], np_binary_op: np.ufunc
        ) -> NDArray:
            return (  # type: ignore[no-any-return]
                np_binary_op(x, y, out=x)
                if np.can_cast(y, x.dtype, casting="same_kind")
                else np_binary_op(x, np.array(y, x.dtype), out=x)
            )

        # Let's do in-place aggregation
        # Get first result, then add up each other
        params = [
            _try_inplace(x, scaling_factors[0], np_binary_op=np.multiply)
            for x in parameters_to_ndarrays(results[0][1].parameters)
        ]

        for i, (_, fit_res) in enumerate(results[1:], start=1):
            res = (
                _try_inplace(x, scaling_factors[i], np_binary_op=np.multiply)
                for x in parameters_to_ndarrays(fit_res.parameters)
            )
            params = [
                reduce(partial(_try_inplace, np_binary_op=np.add), layer_updates)
                for layer_updates in zip(params, res)
            ]

        return params
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = self.VW_weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated
    
    def VW_weighted_loss_avg(self, results: list[tuple[int, float]]) -> float:
        """Aggregate evaluation results obtained from multiple clients."""
        print("_____________VW weighted loss avg_____________")
        print(results)
        total_value_sum = sum(exp(value) for (value, _) in results)
        weighted_losses = [exp(value) * loss for value, loss in results]
        return sum(weighted_losses) / total_value_sum
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        config["server_round"] = server_round
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

def server_fn(context: Context) -> ServerAppComponents:
        strategy = SaveModelStrategy(
            save_dir=context.run_config["save_dir"],
            num_rounds=context.run_config["num-server-rounds"],
            min_fit_clients=context.run_config["clients"],
            min_evaluate_clients=context.run_config["clients"],
            min_available_clients=context.run_config['clients'],
        )
        config = ServerConfig(num_rounds=context.run_config["num-server-rounds"])
        return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)

def main():
    args = parser_arguments()
    total_rounds = args.rounds
    clients = args.clients
    print(f"Starting Server, total rounds {total_rounds}, clients {clients}")
    time_str = Ptime()
    time_str.set_time_now()
    save_dir = "multiagent/" + time_str.get_time_to_minute()
    save_dir = args.log_dir if args.log_dir else save_dir
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
    
