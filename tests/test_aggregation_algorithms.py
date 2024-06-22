import unittest
import numpy as np

from src.AggregationAlgorithm import FedAvg, FedMiddleAvg, FedAvgMomentum, FedAdam, FedSGD, FedProx


class TestAggregationAlgorithms(unittest.TestCase):

    def setUp(self):
        """
        This method will be run before each test. It sets up some common variables.
        """
        self.clients_weights = {
            'client1': {
                'weights': np.array([1, 2, 3]),
                'gradients': np.array([0.1, 0.1, 0.1]),
                'n_training_samples': 10
            },
            'client2': {
                'weights': np.array([4, 5, 6]),
                'gradients': np.array([0.2, 0.2, 0.2]),
                'n_training_samples': 20
            }
        }
        self.federated_model = np.array([2, 3, 4])

    def calculate_weighted_average(self, clients_values, subject="weights"):
        sum_values = None
        total_samples = 0
        for client_data in clients_values.values():
            value = client_data[subject]
            n_samples = client_data["n_training_samples"]
            total_samples += n_samples

            if sum_values is None:
                sum_values = value * n_samples
            else:
                sum_values += value * n_samples

        return sum_values / total_samples if total_samples > 0 else sum_values

    def test_fedavg(self):
        aggregator = FedAvg(weighted=True)
        new_weights = aggregator.aggregate_weights(self.clients_weights, self.federated_model)
        expected_weights = self.calculate_weighted_average(self.clients_weights)
        np.testing.assert_array_almost_equal(new_weights, expected_weights)

    def test_fedmiddleavg(self):
        aggregator = FedMiddleAvg(weighted=True)
        new_weights = aggregator.aggregate_weights(self.clients_weights, self.federated_model)
        avg_weights = self.calculate_weighted_average(self.clients_weights)
        expected_weights = (avg_weights + self.federated_model) / 2
        np.testing.assert_array_almost_equal(new_weights, expected_weights)

    def test_fedavgmomentum(self):
        aggregator = FedAvgMomentum(beta=0.9, learning_rate=0.1, weighted=True)
        avg_weights = self.calculate_weighted_average(self.clients_weights)
        delta = avg_weights - self.federated_model
        momentum = delta
        expected_weights = self.federated_model + 0.1 * momentum
        new_weights = aggregator.aggregate_weights(self.clients_weights, self.federated_model)
        np.testing.assert_array_almost_equal(new_weights, expected_weights)

    def test_fedadam(self):
        aggregator = FedAdam(beta1=0.9, beta2=0.999, epsilon=1e-7, learning_rate=0.001)
        avg_gradient = self.calculate_weighted_average(self.clients_weights, subject="gradients")

        m = np.zeros_like(self.federated_model)
        v = np.zeros_like(self.federated_model)
        t = 1

        m = 0.9 * m + (1 - 0.9) * avg_gradient
        v = 0.999 * v + (1 - 0.999) * (avg_gradient ** 2)

        m_hat = m / (1 - 0.9 ** t)
        v_hat = v / (1 - 0.999 ** t)

        expected_weights = self.federated_model - 0.001 * m_hat / (np.sqrt(v_hat) + 1e-7)
        new_weights = aggregator.aggregate_weights(self.clients_weights, self.federated_model)
        np.testing.assert_array_almost_equal(new_weights, expected_weights)

    def test_fedprox(self):
        aggregator = FedProx(learning_rate=0.01, mu=0.1)
        params = {}
        for key, client_data in self.clients_weights.items():
            gradient = client_data["gradients"]
            weights = client_data["weights"]
            params[key] = self.federated_model - 0.01 * (gradient + 0.1 * (weights - self.federated_model))

        sum_values = None
        total_samples = 0
        for key, param in params.items():
            n_samples = self.clients_weights[key]["n_training_samples"]
            total_samples += n_samples
            if sum_values is None:
                sum_values = param * n_samples
            else:
                sum_values += param * n_samples

        expected_weights = sum_values / total_samples
        new_weights = aggregator.aggregate_weights(self.clients_weights, self.federated_model)
        np.testing.assert_array_almost_equal(new_weights, expected_weights)

    def test_fedsgd(self):
        aggregator = FedSGD(learning_rate=0.01)
        avg_gradient = self.calculate_weighted_average(self.clients_weights, subject="gradients")
        expected_weights = self.federated_model - 0.01 * avg_gradient
        new_weights = aggregator.aggregate_weights(self.clients_weights, self.federated_model)
        np.testing.assert_array_almost_equal(new_weights, expected_weights)

if __name__ == '__main__':
    unittest.main()
