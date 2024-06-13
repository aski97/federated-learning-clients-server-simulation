from abc import ABC, abstractmethod

import numpy as np


class AggregationAlgorithm(ABC):
    """
    Abstract base class for federated aggregation algorithms.
    Defines the strategy for aggregation.
    """

    @abstractmethod
    def aggregate_weights(self, clients_weights: dict, federated_model: np.ndarray) -> np.ndarray:
        """
        Abstract method to aggregate client weights.

        Args:
            clients_weights (dict): Dictionary of client weights.
            federated_model (np.ndarray): Current federated model weights.

        Returns:
            np.ndarray: Aggregated weights.
        """
        pass


class FedAvg(AggregationAlgorithm):
    """
    Implements Federated Averaging (FedAvg) algorithm.
    Aggregates weights by computing the mean of client weights.
    """
    def __init__(self, weighted=True):
        """
        Initializes the FedAvg algorithm.

        Args:
            weighted (bool): If True, use a weighted average based on the number of training samples per client.
                             If False, use a simple arithmetic mean.
        """
        self.weighted = weighted

    def aggregate_weights(self, clients_weights: dict, federated_model: np.ndarray) -> np.ndarray:
        """
        Aggregates client weights using the FedAvg algorithm.

        Args:
            clients_weights (dict): Dictionary containing client weights and respective contributions.
            federated_model (np.ndarray): Current federated model weights.

        Returns:
            np.ndarray: Aggregated weights.
        """
        if not clients_weights:
            raise ValueError("clients_weights dictionary is empty")

        weights_sum = None

        if self.weighted:
            # Compute a weighted average based on the number of training samples per client
            total_samples = 0
            for key, client_weights in clients_weights.items():
                weights = np.array(client_weights["weights"])
                w = client_weights["n_training_samples"]
                total_samples += w

                if weights_sum is None:
                    weights_sum = weights * w
                else:
                    weights_sum += weights * w

            if total_samples == 0:
                raise ValueError("Total number of training samples is zero")

            aggregated_weights = weights_sum / total_samples
        else:
            total_clients = len(clients_weights)
            if total_clients == 0:
                raise ValueError("No clients available for aggregation")

            for key, client_weights in clients_weights.items():
                weights = np.array(client_weights["weights"])

                if weights_sum is None:
                    weights_sum = weights
                else:
                    weights_sum += weights

            aggregated_weights = weights_sum / total_clients

        return aggregated_weights


class FedMiddleAvg(AggregationAlgorithm):
    """
    Implements a middle averaging algorithm.
    Computes the mean between the latest federated model and the FedAvg of client weights.
    """

    def __init__(self, weighted=True):
        """
        Initializes the FedMiddleAvg algorithm.

        Args:
            weighted (bool): If True, use a weighted average based on the number of training samples per client.
                             If False, use a simple arithmetic mean.
        """
        self.weighted = weighted

    def aggregate_weights(self, clients_weights: dict, federated_model: np.ndarray) -> np.ndarray:
        """
        Aggregates client weights using the FedMiddleAvg algorithm.

        Args:
            clients_weights (dict): Dictionary containing client weights and respective contributions.
            federated_model (np.ndarray): Current federated model weights.

        Returns:
            np.ndarray: Aggregated weights.
        """
        if federated_model is None or not isinstance(federated_model, np.ndarray):
            raise ValueError("federated_model must be a valid numpy ndarray")

        fed_avg = FedAvg(self.weighted).aggregate_weights(clients_weights, federated_model)
        fed_middle_avg_weights = (fed_avg + federated_model) / 2
        return fed_middle_avg_weights


class FedAvgServerMomentum(AggregationAlgorithm):
    """
    Implements the server momentum aggregation algorithm for Federated Learning.
    Combines momentum with FedAvg to stabilize and accelerate the training process.

    Args:
        beta (float): Momentum factor.
        learning_rate (float): Learning rate for updating weights.
        weighted (bool): If True, use a weighted average based on the number of training samples per client.
                         If False, use a simple arithmetic mean.
    """

    def __init__(self, beta=0.9, learning_rate=0.1, weighted=True):
        self.beta = beta
        self.learning_rate = learning_rate
        self.weighted = weighted
        self.momentum = None

    def aggregate_weights(self, clients_weights: dict, federated_model: np.ndarray) -> np.ndarray:
        """
        Aggregates client weights using server momentum.

        Args:
            clients_weights (dict): Dictionary containing client weights and respective contributions.
            federated_model (np.ndarray): Current federated model weights.

        Returns:
            np.ndarray: New federated model weights.
        """
        if federated_model is None or not isinstance(federated_model, np.ndarray):
            raise ValueError("federated_model must be a valid numpy ndarray")

        if self.momentum is None:
            self.momentum = np.zeros_like(federated_model)

        # Compute the average operation
        avg = FedAvg(self.weighted).aggregate_weights(clients_weights, federated_model)

        # Compute the difference between the model and the average
        delta = avg - federated_model

        # Update momentum
        self.momentum = self.beta * self.momentum + delta

        # Compute new weights
        new_weights = federated_model + self.learning_rate * self.momentum

        return new_weights
