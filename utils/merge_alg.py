
from flwr.common.parameter import parameters_to_ndarrays
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
from functools import partial, reduce

import numpy as np
from math import exp

from enum import Enum

class AggregationStrategy(Enum):
    UNIFORM = 0
    VW_SOFTMAX = 1
    VW_MIN_ADJUSTED = 2

def aggregate_strategy(results: list[tuple[ClientProxy, FitRes]], option: AggregationStrategy) -> tuple[NDArrays, list]:
    if option == AggregationStrategy.UNIFORM:
        """Examples-weighted average"""

        # Compute scaling factors for each result
        multiple_weights = [1/len(results) for _, fit_res in results]

    elif option == AggregationStrategy.VW_SOFTMAX:
        """Compute in-place RL-value weighted soft-max average."""
        all_values = [fit_res.num_examples for (_, fit_res) in results]
        max_value = max(all_values)
        exp_values = [exp(value - max_value) for value in all_values]
        exp_values = [round(x, 6) for x in exp_values]
        values_exp_sum = sum(exp_values)
        multiple_weights = [value/ values_exp_sum for value in exp_values]

    elif option == AggregationStrategy.VW_MIN_ADJUSTED:
        """Compute in-place RL-value weighted soft-max average."""
        all_values = [fit_res.num_examples for (_, fit_res) in results]

        # Find minimum value
        min_value = min(all_values)

        # Calculate adjusted values by subtracting minimum
        adjusted_values = [value - min_value for value in all_values]

        # Sum of adjusted values
        adjusted_sum = sum(adjusted_values) + 1e-4

        # Calculate weights: (adjusted_value / sum) + (1 / len(values))
        h = len(adjusted_values)
        assert h > 0, "h should be greater than 0"

        multiple_weights = [(value / adjusted_sum + (1 / h)) / 2 for value in adjusted_values]
    
    else:
        raise ValueError(f"Unknown aggregation strategy: {option}")

    all_values = [fit_res.num_examples for (_, fit_res) in results]
    show_weights = all_values + multiple_weights
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

    return params, show_weights