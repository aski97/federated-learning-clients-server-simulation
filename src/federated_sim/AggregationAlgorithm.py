from abc import ABC, abstractmethod

import numpy as np


class AggregationAlgorithm(ABC):
    """
    Abstract base class for federated aggregation algorithms.
    Defines the strategy for aggregation.

    NOTE: All algorithms operate layer-wise and return a list[numpy.ndarray]
    compatible with Keras `Model.get_weights()` / `set_weights()`.
    """

    @staticmethod
    def _to_layer_list(obj):
        """
        Normalize federated model / client layer containers to a list of numpy arrays.
        Accepts:
         - list/tuple of ndarray-like
         - numpy.ndarray with dtype=object (as used in some places)
        Returns: list[numpy.ndarray]
        """
        if obj is None:
            return None

        if isinstance(obj, np.ndarray):
            # common pattern: np.array(list_of_ndarrays, dtype='object')
            try:
                # If it's an object-dtype array containing arrays, extract them
                if obj.dtype == np.object_:
                    return [np.array(x) for x in obj.tolist()]
                # otherwise it's a regular numeric ndarray -> treat as single-layer
                return [obj.copy()]
            except Exception:
                return [np.array(x) for x in obj]
        elif isinstance(obj, (list, tuple)):
            return [np.array(x) for x in obj]
        else:
            # single array-like
            return [np.array(obj)]

    @staticmethod
    def compute_avg(clients_values: dict, subject: str = "weights", weighted: bool = True):
        """
        Compute per-layer average. Assumes clients_values[client][subject] is an iterable
        (list/tuple) of numpy arrays (one per layer). Returns a list of numpy arrays.

        Weighted averaging uses 'n_training_samples' key from each client dict.
        """
        if not clients_values:
            raise ValueError("No clients provided for aggregation")

        # infer structure from first client
        first_item = next(iter(clients_values.values()))
        if subject not in first_item:
            raise KeyError(f"Subject '{subject}' not present in client data")
        first_layers = AggregationAlgorithm._to_layer_list(first_item[subject])
        n_layers = len(first_layers)

        # initialize accumulators per layer
        accum = [None] * n_layers

        if weighted:
            total_samples = 0
            for client_vals in clients_values.values():
                layers = AggregationAlgorithm._to_layer_list(client_vals[subject])
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
                layers = AggregationAlgorithm._to_layer_list(client_vals[subject])
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
    def aggregate_weights(self, clients_weights: dict, federated_model) -> list:
        """
        Abstract method to aggregate client weights.

        Args:
            clients_weights (dict): Dictionary of client weights.
            federated_model: Current federated model weights (list or np.ndarray).

        Returns:
            list[numpy.ndarray]: Aggregated weights (layer-wise list).
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

    def aggregate_weights(self, clients_weights: dict, federated_model) -> list:
        """
        Aggregates client weights using the FedAvg algorithm.

        Returns list[numpy.ndarray].
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

    def aggregate_weights(self, clients_weights: dict, federated_model) -> list:
        """
        Aggregates client weights using the FedMiddleAvg algorithm.

        Returns list[numpy.ndarray].
        """
        if federated_model is None:
            raise ValueError("federated_model must be provided")

        fed_avg = self.compute_avg_weights(clients_weights, self.weighted)
        fed_model_layers = self._to_layer_list(federated_model)

        if len(fed_avg) != len(fed_model_layers):
            raise ValueError("Mismatch in number of layers between clients and federated model")

        return [(fed_avg[i] + fed_model_layers[i]) / 2.0 for i in range(len(fed_avg))]


class FedAvgMomentum(AggregationAlgorithm):
    """
    Implements the server momentum aggregation algorithm for Federated Learning.
    Combines momentum with FedAvg to stabilize and accelerate the training process.
    """

    def __init__(self, beta=0.9, learning_rate=0.01, weighted=True):
        self.beta = beta
        self.learning_rate = learning_rate
        self.weighted = weighted
        self.momentum = None  # will be list of arrays

    def aggregate_weights(self, clients_weights: dict, federated_model) -> list:
        """
        Aggregates client weights using server momentum.

        Returns list[numpy.ndarray].
        """
        if federated_model is None:
            raise ValueError("federated_model must be provided")

        fed_model_layers = self._to_layer_list(federated_model)

        if self.momentum is None:
            self.momentum = [np.zeros_like(layer, dtype=float) for layer in fed_model_layers]

        avg = self.compute_avg_weights(clients_weights, self.weighted)

        if len(avg) != len(fed_model_layers):
            raise ValueError("Mismatch in number of layers between clients and federated model")

        # delta = avg - federated_model (per-layer)
        deltas = [avg[i] - fed_model_layers[i] for i in range(len(avg))]

        # update momentum and compute new weights using a damped update
        new_weights = []
        for i, delta in enumerate(deltas):
            # use (1-beta) factor to avoid unbounded accumulation
            self.momentum[i] = self.beta * self.momentum[i] + (1.0 - self.beta) * delta
            new_w = fed_model_layers[i] + self.learning_rate * self.momentum[i]
            new_weights.append(new_w)

        return new_weights


class FedAdam(AggregationAlgorithm):
    """
    Implements the Adam (Adaptive Moment Estimation) aggregation algorithm for Federated Learning.
    Operates layer-wise and stores m/v as lists of arrays.
    """

    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-7, learning_rate=0.001):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.m = None  # list of first moments (per-layer)
        self.v = None  # list of second moments (per-layer)
        self.t = 0  # timestep

    def aggregate_weights(self, clients_weights: dict, federated_model) -> list:
        """
        Aggregates client weights using an Adam-like server optimizer.
        clients_weights is expected to contain 'weights'.
        Returns list[numpy.ndarray].
        """
        if federated_model is None:
            raise ValueError("federated_model must be provided")

        fed_model_layers = self._to_layer_list(federated_model)

        if self.m is None:
            self.m = [np.zeros_like(layer, dtype=float) for layer in fed_model_layers]
        if self.v is None:
            self.v = [np.zeros_like(layer, dtype=float) for layer in fed_model_layers]

        self.t += 1

        # Use FedAvg as base update (weighted)
        avg = self.compute_avg_weights(clients_weights, weighted=True)

        if len(avg) != len(fed_model_layers):
            raise ValueError("Mismatch in number of layers between clients and federated model")

        new_params = []
        for i in range(len(avg)):
            delta_w = avg[i] - fed_model_layers[i]
            # update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * delta_w
            # update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (delta_w ** 2)

            # compute bias-corrected estimates
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # parameter update
            denom = np.sqrt(v_hat) + self.epsilon
            new_param = fed_model_layers[i] + self.learning_rate * (m_hat / denom)
            new_params.append(new_param)

        return new_params

class FedAdagrad(AggregationAlgorithm):
    """
    Implements the FedAdagrad algorithm.
    
    This is an Adam-like optimizer that uses the Adagrad rule for the 
    second moment (v_t).
    """

    def __init__(self, beta1=0.9, epsilon=1e-7, learning_rate=0.001):
        self.beta1 = beta1
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.m = None  # list of first moments (per-layer)
        self.v = None  # list of second moments (per-layer)
        self.t = 0     # timestep

    def aggregate_weights(self, clients_weights: dict, federated_model) -> list:
        """
        Aggregates client weights using the FedAdagrad server optimizer.
        """
        if federated_model is None:
            raise ValueError("federated_model must be provided")

        fed_model_layers = self._to_layer_list(federated_model)

        if self.m is None:
            self.m = [np.zeros_like(layer, dtype=float) for layer in fed_model_layers]
        if self.v is None:
            # v_t is an accumulator, initialize to zeros
            self.v = [np.zeros_like(layer, dtype=float) for layer in fed_model_layers]

        self.t += 1

        # Use FedAvg as base update (weighted)
        avg = self.compute_avg_weights(clients_weights, weighted=True)

        if len(avg) != len(fed_model_layers):
            raise ValueError("Mismatch in number of layers between clients and federated model")

        new_params = []
        for i in range(len(avg)):
            delta_w = avg[i] - fed_model_layers[i] 

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * delta_w
            
            # v_t = v_{t-1} + Delta_t^2
            self.v[i] = self.v[i] + (delta_w ** 2)

            # Compute bias-corrected first moment (m_hat)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            denom = np.sqrt(self.v[i]) + self.epsilon
            new_param = fed_model_layers[i] + self.learning_rate * (m_hat / denom)
            new_params.append(new_param)

        return new_params


class FedYogi(AggregationAlgorithm):
    """
    Implements the FedYogi algorithm.
    
    This is an Adam-variant optimizer that uses a different rule for 
    the second moment (v_t) to control its increase.
    """

    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-7, learning_rate=0.001):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.m = None  # list of first moments (per-layer)
        self.v = None  # list of second moments (per-layer)
        self.t = 0     # timestep

    def aggregate_weights(self, clients_weights: dict, federated_model) -> list:
        """
        Aggregates client weights using the FedYogi server optimizer.
        """
        if federated_model is None:
            raise ValueError("federated_model must be provided")

        fed_model_layers = self._to_layer_list(federated_model)

        if self.m is None:
            self.m = [np.zeros_like(layer, dtype=float) for layer in fed_model_layers]
        if self.v is None:
            self.v = [np.zeros_like(layer, dtype=float) for layer in fed_model_layers]

        self.t += 1

        # Use FedAvg as base update (weighted)
        avg = self.compute_avg_weights(clients_weights, weighted=True)

        if len(avg) != len(fed_model_layers):
            raise ValueError("Mismatch in number of layers between clients and federated model")

        new_params = []
        for i in range(len(avg)):
            delta_w = avg[i] - fed_model_layers[i] 
            delta_w_sq = delta_w ** 2

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * delta_w
            
            v_prev = self.v[i]
            sign_diff = np.sign(v_prev - delta_w_sq)
            self.v[i] = v_prev - (1 - self.beta2) * delta_w_sq * sign_diff

            # Compute bias-corrected estimates (same as FedAdam)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            denom = np.sqrt(v_hat) + self.epsilon
            new_param = fed_model_layers[i] + self.learning_rate * (m_hat / denom)
            new_params.append(new_param)

        return new_params


class FedSGD(AggregationAlgorithm):
    """
    Implements Federated SGD (Stochastic Gradient Descent).
    Aggregates gradients directly and updates the model.
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def aggregate_weights(self, clients_gradients: dict, federated_model) -> list:
        if federated_model is None:
            raise ValueError("federated_model must be provided")

        fed_model_layers = self._to_layer_list(federated_model)
        avg_gradient = self.compute_avg_gradients(clients_gradients, True)

        if len(avg_gradient) != len(fed_model_layers):
            raise ValueError("Mismatch in number of layers between gradients and federated model")

        new_weights = [fed_model_layers[i] - self.learning_rate * avg_gradient[i] for i in range(len(fed_model_layers))]

        return new_weights