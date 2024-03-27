from abc import ABC, abstractmethod

import numpy as np


class AggregationStrategy(ABC):
    """
    The Strategy for federated aggregation algorithms
    """

    @abstractmethod
    def aggregate_weights(self, weights: dict) -> np.ndarray:
        pass


class FedAvgStrategy(AggregationStrategy):
    """
        Aggregating weights computing the mean
    """
    def aggregate_weights(self, weights: dict) -> np.ndarray:
        weights_sum = None

        for key, client_weight in weights.items():
            if weights_sum is None:
                weights_sum = np.array(client_weight, dtype='object')
            else:
                weights_sum += np.array(client_weight, dtype='object')

        # aggregate weights, computing the mean
        fed_avg_weights = weights_sum / len(weights)

        return fed_avg_weights

