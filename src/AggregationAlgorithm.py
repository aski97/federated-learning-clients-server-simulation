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



# class FedAvgWS(AggregationAlgorithm):
#     """
#
#     """
#     def aggregate_weights(self, clients_weights: dict, federated_model: np.ndarray) -> np.ndarray:
#         fed_avg = FedAvg().aggregate_weights(clients_weights, federated_model)
#
#         total_data_size = sum(data_sizes)
#
#
#         # Sum the scaled parameters of all models
#         for model, size in zip(models, data_sizes):
#             for k in aggregated_model.keys():
#                 aggregated_model[k] += model[k] * (size / total_data_size)
#
#         return aggregated_model
#
#         return fed_middle_avg_weights