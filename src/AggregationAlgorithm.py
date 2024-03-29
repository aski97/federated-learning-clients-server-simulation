from abc import ABC, abstractmethod

import numpy as np


class AggregationAlgorithm(ABC):
    """
    The Strategy for federated aggregation algorithms
    """

    @abstractmethod
    def aggregate_weights(self, clients_weights: dict, federated_model: np.ndarray) -> np.ndarray:
        pass


class FedAvgAlgorithm(AggregationAlgorithm):
    """
        Aggregating weights computing the mean of clients weights
    """
    def aggregate_weights(self, clients_weights: dict, federated_model: np.ndarray) -> np.ndarray:
        weights_sum = None

        for key, client_weight in clients_weights.items():
            if weights_sum is None:
                weights_sum = np.array(client_weight, dtype='object')
            else:
                weights_sum += np.array(client_weight, dtype='object')

        # aggregate weights, computing the mean
        fed_avg_weights = weights_sum / len(clients_weights)

        return fed_avg_weights


class FedMiddleAvgAlgorithm(AggregationAlgorithm):
    """
    Aggregating weights computing the mean between the old federated model with the fed_avg_weights
    of the client weights
    """
    def aggregate_weights(self, clients_weights: dict, federated_model: np.ndarray) -> np.ndarray:
        fed_avg = FedAvgAlgorithm().aggregate_weights(clients_weights, federated_model)
        fed_middle_avg_weights = (fed_avg + federated_model) / 2
        return fed_middle_avg_weights
