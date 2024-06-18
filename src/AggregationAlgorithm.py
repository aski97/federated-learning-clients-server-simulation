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
        avg = FedAvg(self.weighted).aggregate_weights(clients_weights, federated_model)

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
        weighted (bool): If True, use a weighted average based on the number of training samples per client.
                         If False, use a simple arithmetic mean.
    """

    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-7, learning_rate=0.001, weighted=True):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.weighted = weighted
        self.m = None  # First moment vector
        self.v = None  # Second moment vector
        self.t = 0  # Timestep

    def aggregate_weights(self, clients_gradients: dict, federated_model: np.ndarray) -> np.ndarray:
        """
        Aggregates client gradients using the Adam algorithm.

        Args:
            clients_gradients (dict): Dictionary containing client gradients and respective contributions.
            federated_model (np.ndarray): Current federated model weights.

        Returns:
            np.ndarray: New federated model weights.
        """
        if federated_model is None or not isinstance(federated_model, np.ndarray):
            raise ValueError("federated_model must be a valid numpy ndarray")

        if self.m is None:
            self.m = np.zeros_like(federated_model)
        if self.v is None:
            self.v = np.zeros_like(federated_model)

        self.t += 1

        # Compute the weighted average of gradients
        if self.weighted:
            total_samples = sum(client_info["n_training_samples"] for client_info in clients_gradients.values())
            avg_gradient = sum(np.array(client_info["gradients"]) * client_info["n_training_samples"] / total_samples for client_info in clients_gradients.values())
        else:
            total_clients = len(clients_gradients)
            avg_gradient = sum(np.array(client_info["gradients"]) for client_info in clients_gradients.values()) / total_clients

        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * avg_gradient

        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (avg_gradient ** 2)

        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.t)

        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.t)

        v_hat_sqr_list = []

        # Itera su ogni array nella lista originale e calcola la radice quadrata
        for arr in v_hat:
            sqrt_arr = np.sqrt(arr)
            v_hat_sqr_list.append(sqrt_arr)

        v_hat_squared = np.array(v_hat_sqr_list, dtype='object')

        # # Compute new weights
        new_weights = federated_model - self.learning_rate * m_hat / (v_hat_squared + self.epsilon)

        return new_weights

class FedProx(AggregationAlgorithm):
    """
    Implements FedProx algorithm.
    Adds a regularization term to penalize differences between local and global models.
    """
    def __init__(self, mu=0.1, weighted=True):
        self.mu = mu
        self.weighted = weighted

    def aggregate_weights(self, clients_weights: dict, federated_model: np.ndarray) -> np.ndarray:
        if not clients_weights:
            raise ValueError("clients_weights dictionary is empty")

        if federated_model is None or not isinstance(federated_model, np.ndarray):
            raise ValueError("federated_model must be a valid numpy ndarray")

        weights_sum = None
        if self.weighted:
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

            avg_weights = weights_sum / total_samples
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

            avg_weights = weights_sum / total_clients

        prox_weights = avg_weights + self.mu * (federated_model - avg_weights)
        return prox_weights


class FedAdagrad(AggregationAlgorithm):
    """
    Implements FedAdagrad algorithm.
    Uses adaptive gradient scaling to adjust learning rates.
    """
    def __init__(self, learning_rate=0.01, epsilon=1e-8, weighted=True):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.weighted = weighted
        self.h = None  # Accumulator for squared gradients

    def aggregate_weights(self, clients_gradients: dict, federated_model: np.ndarray) -> np.ndarray:
        if federated_model is None or not isinstance(federated_model, np.ndarray):
            raise ValueError("federated_model must be a valid numpy ndarray")

        if self.h is None:
            self.h = np.zeros_like(federated_model)

        if self.weighted:
            total_samples = sum(client_info["n_training_samples"] for client_info in clients_gradients.values())
            avg_gradient = sum(np.array(client_info["gradients"]) * client_info["n_training_samples"] / total_samples for client_info in clients_gradients.values())
        else:
            total_clients = len(clients_gradients)
            avg_gradient = sum(np.array(client_info["gradients"]) for client_info in clients_gradients.values()) / total_clients

        self.h += avg_gradient ** 2
        adjusted_gradients = avg_gradient / (np.sqrt(self.h) + self.epsilon)
        new_weights = federated_model - self.learning_rate * adjusted_gradients

        return new_weights


class FedYogi(AggregationAlgorithm):
    """
    Implements FedYogi algorithm.
    A variant of FedAdam designed to be robust to noisy and non-i.i.d. data.
    """
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-3, learning_rate=0.001, weighted=True):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.weighted = weighted
        self.m = None  # First moment vector
        self.v = None  # Second moment vector
        self.t = 0  # Timestep

    def aggregate_weights(self, clients_gradients: dict, federated_model: np.ndarray) -> np.ndarray:
        if federated_model is None or not isinstance(federated_model, np.ndarray):
            raise ValueError("federated_model must be a valid numpy ndarray")

        if self.m is None:
            self.m = np.zeros_like(federated_model)
        if self.v is None:
            self.v = np.zeros_like(federated_model)

        self.t += 1

        if self.weighted:
            total_samples = sum(client_info["n_training_samples"] for client_info in clients_gradients.values())
            avg_gradient = sum(np.array(client_info["gradients"]) * client_info["n_training_samples"] / total_samples for client_info in clients_gradients.values())
        else:
            total_clients = len(clients_gradients)
            avg_gradient = sum(np.array(client_info["gradients"]) for client_info in clients_gradients.values()) / total_clients

        self.m = self.beta1 * self.m + (1 - self.beta1) * avg_gradient

        self.v = self.v - (1 - self.beta2) * (avg_gradient ** 2) * np.sign(self.v - avg_gradient ** 2)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        m_hat = self.m / (1 - self.beta1 ** self.t)

        new_weights = federated_model - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return new_weights


class FedNova(AggregationAlgorithm):
    """
    Implements FedNova algorithm.
    Normalizes updates for each client before aggregation.
    """
    def __init__(self, learning_rate=0.01, tau=1, weighted=True):
        self.learning_rate = learning_rate
        self.tau = tau
        self.weighted = weighted

    def aggregate_weights(self, clients_gradients: dict, federated_model: np.ndarray) -> np.ndarray:
        if federated_model is None or not isinstance(federated_model, np.ndarray):
            raise ValueError("federated_model must be a valid numpy ndarray")

        if self.weighted:
            total_samples = sum(client_info["n_training_samples"] for client_info in clients_gradients.values())
            normalized_updates = sum((np.array(client_info["gradients"]) * client_info["n_training_samples"] / total_samples) / (self.tau + 1) for client_info in clients_gradients.values())
        else:
            total_clients = len(clients_gradients)
            normalized_updates = sum(np.array(client_info["gradients"]) / (self.tau + 1) for client_info in clients_gradients.values()) / total_clients

        new_weights = federated_model - self.learning_rate * normalized_updates

        return new_weights


class FedSGD(AggregationAlgorithm):
    """
    Implements Federated SGD (Stochastic Gradient Descent).
    Aggregates gradients directly and updates the model.
    """
    def __init__(self, learning_rate=0.01, weighted=True):
        self.learning_rate = learning_rate
        self.weighted = weighted

    def aggregate_weights(self, clients_gradients: dict, federated_model: np.ndarray) -> np.ndarray:
        if federated_model is None or not isinstance(federated_model, np.ndarray):
            raise ValueError("federated_model must be a valid numpy ndarray")

        if self.weighted:
            total_samples = sum(client_info["n_training_samples"] for client_info in clients_gradients.values())
            avg_gradient = sum(np.array(client_info["gradients"]) * client_info["n_training_samples"] / total_samples for client_info in clients_gradients.values())
        else:
            total_clients = len(clients_gradients)
            avg_gradient = sum(np.array(client_info["gradients"]) for client_info in clients_gradients.values()) / total_clients

        new_weights = federated_model - self.learning_rate * avg_gradient

        return new_weights