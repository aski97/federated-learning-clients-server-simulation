from abc import ABC, abstractmethod

import numpy as np


class AggregationAlgorithm(ABC):
    """
    Abstract base class for federated aggregation algorithms.
    Defines the strategy for aggregation.
    """

    @staticmethod
    def compute_avg(clients_values: dict, subject: str = "weights", weighted: bool = True):
        """
        Compute per-layer average. Assumes clients_values[client][subject] is an iterable
        (list/tuple) of numpy arrays (one per layer). Returns a list of numpy arrays.
        """
        if not clients_values:
            raise ValueError("No clients provided for aggregation")

        # get first client's layer structure to infer number of layers
        first_item = next(iter(clients_values.values()))
        if subject not in first_item:
            raise KeyError(f"Subject '{subject}' not present in client data")
        first_layers = first_item[subject]
        n_layers = len(first_layers)

        # initialize accumulators per layer
        accum = [None] * n_layers

        if weighted:
            total_samples = 0
            for client_vals in clients_values.values():
                layers = client_vals[subject]
                w = int(client_vals.get("n_training_samples", 0))
                total_samples += w
                for i, arr in enumerate(layers):
                    arr = np.array(arr, dtype=float)
                    if accum[i] is None:
                        accum[i] = arr * w
                    else:
                        accum[i] = accum[i] + arr * w

            if total_samples == 0:
                raise ValueError("Total number of training samples is zero")

            avg = [accum[i] / total_samples for i in range(n_layers)]
        else:
            total_clients = len(clients_values)
            for client_vals in clients_values.values():
                layers = client_vals[subject]
                for i, arr in enumerate(layers):
                    arr = np.array(arr, dtype=float)
                    if accum[i] is None:
                        accum[i] = arr
                    else:
                        accum[i] = accum[i] + arr
            avg = [accum[i] / total_clients for i in range(n_layers)]

        return avg

    @staticmethod
    def compute_avg_weights(clients_weights: dict, weighted: bool = True):
        return AggregationAlgorithm.compute_avg(clients_weights, "weights", weighted)

    @staticmethod
    def compute_avg_gradients(clients_gradients: dict, weighted: bool = True):
        return AggregationAlgorithm.compute_avg(clients_gradients, "gradients", weighted)

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

        return self.compute_avg_weights(clients_weights, self.weighted)


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

        fed_avg = self.compute_avg_weights(clients_weights, self.weighted)
        fed_middle_avg_weights = (fed_avg + federated_model) / 2
        return fed_middle_avg_weights


class FedAvgMomentum(AggregationAlgorithm):
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
        avg = self.compute_avg_weights(clients_weights, self.weighted)

        # Compute the difference between the model and the average
        delta = avg - federated_model

        # Update momentum
        self.momentum = self.beta * self.momentum + delta

        # Compute new weights
        new_weights = federated_model + self.learning_rate * self.momentum

        return new_weights


class FedAdam(AggregationAlgorithm):
    """
    Implements the Adam (Adaptive Moment Estimation) aggregation algorithm for Federated Learning.
    Combines first and second moment estimates with FedAvg to stabilize and accelerate the training process.

    Args:
        beta1 (float): Exponential decay rate for the first moment estimates.
        beta2 (float): Exponential decay rate for the second moment estimates.
        epsilon (float): Small constant for numerical stability.
        learning_rate (float): Learning rate for updating weights.
    """

    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-7, learning_rate=0.001):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.m = None  # First moment vectors for each client
        self.v = None  # Second moment vectors for each client
        self.t = 0  # Timestep

    def aggregate_weights(self, clients_data: dict, federated_model: np.ndarray) -> np.ndarray:
        """
        Aggregates client gradients using the Adam algorithm.

        Args:
            clients_data (dict): Dictionary containing client weights, gradients and respective contributions.
            federated_model (np.ndarray): Current federated model weights.

        Returns:
            np.ndarray: New federated model weights.
        """
        if federated_model is None or not isinstance(federated_model, np.ndarray):
            raise ValueError("federated_model must be a valid numpy ndarray")

        # Initialize m and v for each client if not already done
        if self.m is None:
            self.m = np.zeros_like(federated_model)
        if self.v is None:
            self.v = np.zeros_like(federated_model)
        self.t += 1

        avg = self.compute_avg_weights(clients_data)

        delta_w = avg - federated_model

        # Update biased first moment estimate for client
        self.m = self.beta1 * self.m + (1 - self.beta1) * delta_w

        # Update biased second raw moment estimate for client
        self.v = self.beta2 * self.v + (1 - self.beta2) * (delta_w ** 2)

        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.t)

        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.t)

        v_hat_sqr_list = []

        for arr in v_hat:
            sqrt_arr = np.sqrt(arr)
            v_hat_sqr_list.append(sqrt_arr)

        v_hat_squared = np.array(v_hat_sqr_list, dtype='object')

        new_params = federated_model + self.learning_rate * m_hat / (v_hat_squared + self.epsilon)

        return new_params


class FedSGD(AggregationAlgorithm):
    """
    Implements Federated SGD (Stochastic Gradient Descent).
    Aggregates gradients directly and updates the model.
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def aggregate_weights(self, clients_gradients: dict, federated_model: np.ndarray) -> np.ndarray:
        if federated_model is None or not isinstance(federated_model, np.ndarray):
            raise ValueError("federated_model must be a valid numpy ndarray")

        # Compute the weighted average of gradients
        avg_gradient = self.compute_avg_gradients(clients_gradients, True)

        new_weights = federated_model - self.learning_rate * avg_gradient

        return new_weights