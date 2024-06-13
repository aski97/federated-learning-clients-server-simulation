from abc import ABC, abstractmethod

import numpy as np


class AggregationAlgorithm(ABC):
    """
    The Strategy for federated aggregation algorithms
    """

    @abstractmethod
    def aggregate_weights(self, clients_weights: dict, federated_model: np.ndarray) -> np.ndarray:
        pass


class FedAvg(AggregationAlgorithm):
    """
        Aggregating weights computing the mean of clients weights
    """
    def aggregate_weights(self, clients_weights: dict, federated_model: np.ndarray) -> np.ndarray:
        weights_sum = None

        for key, client_weights in clients_weights.items():
            weights = client_weights["weights"]
            if weights_sum is None:
                weights_sum = np.array(weights, dtype='object')
            else:
                weights_sum += np.array(weights, dtype='object')

        # aggregate weights, computing the mean
        aggregated_weights = weights_sum / len(clients_weights)

        return aggregated_weights


class FedWeightedAvg(AggregationAlgorithm):
    """
        Aggregating weights computing the weighted average of clients weights. Giving more importance to
        nodes that have more training samples.
    """
    def aggregate_weights(self, clients_weights: dict, federated_model: np.ndarray) -> np.ndarray:
        weights_sum = None
        total_training_samples = 0
        for key, client_weights in clients_weights.items():
            weights = client_weights["weights"]
            n_training_samples = client_weights["n_training_samples"]
            if weights_sum is None:
                weights_sum = np.array(weights, dtype='object') * n_training_samples
            else:
                weights_sum += np.array(weights, dtype='object') * n_training_samples

            total_training_samples += n_training_samples

        # aggregate weights, computing the mean
        aggregated_weights = weights_sum / total_training_samples

        return aggregated_weights


class FedMiddleAvg(AggregationAlgorithm):
    """
    Aggregating weights computing the mean between the latest federated model with the fed_avg_weights
    of the client weights
    """
    def aggregate_weights(self, clients_weights: dict, federated_model: np.ndarray) -> np.ndarray:
        fed_avg = FedAvg().aggregate_weights(clients_weights, federated_model)
        fed_middle_avg_weights = (fed_avg + federated_model) / 2
        return fed_middle_avg_weights


class FedAvgServerMomentum(AggregationAlgorithm):
    """
    Implementation of the server momentum aggregation algorithm for Federated Learning.

    Args:
        beta (float): Momentum factor.
        learning_rate (float): Learning rate for updating weights.
    """
    def __init__(self, beta=0.9, learning_rate=0.1):
        self.beta = beta
        self.learning_rate = learning_rate
        self.momentum = None

    def aggregate_weights(self, clients_weights: dict, federated_model: np.ndarray) -> np.ndarray:
        """
        Aggregates client weights using server momentum.

        Args:
            clients_weights (dict): Dictionary containing client weights and respective contributions.
            federated_model (np.ndarray): Current federated model.

        Returns:
            np.ndarray: New federated model weights.
        """
        if self.momentum is None:
            self.momentum = np.zeros_like(federated_model)

        # Computing the average operation
        avg = FedWeightedAvg().aggregate_weights(clients_weights, federated_model)

        # Computing the difference btw the model and the avg
        delta = avg - federated_model

        # Updating momentum
        self.momentum = self.beta * self.momentum + delta

        # Computing new weights

        new_weights = federated_model + self.learning_rate * self.momentum

        return new_weights
