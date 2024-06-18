import unittest
import numpy as np

from src.AggregationAlgorithm import FedAvg, FedMiddleAvg, FedAvgMomentum, FedAdamW, FedAdam


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
                'weights': np.array([7, 8, 9]),
                'gradients': np.array([0.2, 0.2, 0.2]),
                'n_training_samples': 20
            }
        }
        self.federated_model = np.array([0.5, 0.5, 0.5])

    def test_fedavg_weighted(self):
        """
        Test the FedAvg algorithm with weighted averaging.
        """
        fedavg = FedAvg(weighted=True)
        aggregated_weights = fedavg.aggregate_weights(self.clients_weights, self.federated_model)
        expected_weights = (self.clients_weights['client1']['weights'] * 10 +
                            self.clients_weights['client2']['weights'] * 20) / 30
        np.testing.assert_array_almost_equal(aggregated_weights, expected_weights)

    def test_fedavg_unweighted(self):
        """
        Test the FedAvg algorithm with unweighted averaging.
        """
        fedavg = FedAvg(weighted=False)
        aggregated_weights = fedavg.aggregate_weights(self.clients_weights, self.federated_model)
        expected_weights = (self.clients_weights['client1']['weights'] +
                            self.clients_weights['client2']['weights']) / 2
        np.testing.assert_array_almost_equal(aggregated_weights, expected_weights)

    def test_fedmiddleavg_weighted(self):
        """
        Test the FedMiddleAvg algorithm with weighted averaging.
        """
        fedmiddleavg = FedMiddleAvg(weighted=True)
        fed_avg = (self.clients_weights['client1']['weights'] * 10 +
                   self.clients_weights['client2']['weights'] * 20) / 30
        expected_weights = (fed_avg + self.federated_model) / 2
        aggregated_weights = fedmiddleavg.aggregate_weights(self.clients_weights, self.federated_model)
        np.testing.assert_array_almost_equal(aggregated_weights, expected_weights)

    def test_fedmiddleavg_unweighted(self):
        """
        Test the FedMiddleAvg algorithm with unweighted averaging.
        """
        fedmiddleavg = FedMiddleAvg(weighted=False)
        fed_avg = (self.clients_weights['client1']['weights'] +
                   self.clients_weights['client2']['weights']) / 2
        expected_weights = (fed_avg + self.federated_model) / 2
        aggregated_weights = fedmiddleavg.aggregate_weights(self.clients_weights, self.federated_model)
        np.testing.assert_array_almost_equal(aggregated_weights, expected_weights)

    def test_fedavg_momentum(self):
        """
        Test the FedAvgMomentum algorithm.
        """
        fedavg_momentum = FedAvgMomentum(beta=0.9, learning_rate=0.1, weighted=True)
        federated_model = self.federated_model.copy()

        # Initialize momentum
        momentum = np.zeros_like(federated_model)

        # Simulate two rounds of updates
        for _ in range(2):
            fed_avg = (self.clients_weights['client1']['weights'] * 10 +
                       self.clients_weights['client2']['weights'] * 20) / 30
            delta = fed_avg - federated_model

            # Update momentum
            momentum = 0.9 * momentum + delta

            # Compute new weights
            federated_model = federated_model + 0.1 * momentum

        expected_weights = federated_model
        aggregated_weights = fedavg_momentum.aggregate_weights(self.clients_weights, self.federated_model)
        np.testing.assert_array_almost_equal(aggregated_weights, expected_weights, decimal=3)

    def test_fedadam_with_gradients(self):
        """
        Test the FedAdam algorithm with gradients.
        """
        fedadam = FedAdam(beta1=0.9, beta2=0.999, epsilon=1e-7, learning_rate=0.001, weighted=True)
        federated_model = self.federated_model.copy()

        # Initialize moment vectors
        m = np.zeros_like(federated_model)
        v = np.zeros_like(federated_model)

        # Simulate two rounds of updates
        for t in range(1, 3):
            total_samples = sum(client_info["n_training_samples"] for client_info in self.clients_weights.values())
            avg_gradient = sum(
                client_info["gradients"] * client_info["n_training_samples"] / total_samples for client_info in
                self.clients_weights.values())

            # Update biased first moment estimate
            m = 0.9 * m + (1 - 0.9) * avg_gradient

            # Update biased second raw moment estimate
            v = 0.999 * v + (1 - 0.999) * (avg_gradient ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = m / (1 - 0.9 ** t)

            # Compute bias-corrected second raw moment estimate
            v_hat = v / (1 - 0.999 ** t)

            # Compute new weights
            federated_model = federated_model - 0.001 * m_hat / (np.sqrt(v_hat) + 1e-7)

        expected_weights = federated_model
        aggregated_weights = fedadam.aggregate_weights(self.clients_weights, self.federated_model)
        np.testing.assert_array_almost_equal(aggregated_weights, expected_weights, decimal=3)

    def test_fedadam_with_weights(self):
        """
        Test the FedAdam algorithm with weights.
        """
        fedadam_w = FedAdamW(beta1=0.9, beta2=0.999, epsilon=1e-7, learning_rate=0.001, weighted=True)
        federated_model = self.federated_model.copy()

        # Initialize moment vectors
        m = np.zeros_like(federated_model)
        v = np.zeros_like(federated_model)

        # Simulate two rounds of updates
        for t in range(1, 3):
            avg = (self.clients_weights['client1']['weights'] * 10 +
                   self.clients_weights['client2']['weights'] * 20) / 30
            delta = avg - federated_model

            # Update biased first moment estimate
            m = 0.9 * m + (1 - 0.9) * delta

            # Update biased second raw moment estimate
            v = 0.999 * v + (1 - 0.999) * (delta ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = m / (1 - 0.9 ** t)

            # Compute bias-corrected second raw moment estimate
            v_hat = v / (1 - 0.999 ** t)

            # Compute new weights
            federated_model = federated_model + 0.001 * m_hat / (np.sqrt(v_hat) + 1e-7)

        expected_weights = federated_model
        aggregated_weights = fedadam_w.aggregate_weights(self.clients_weights, self.federated_model)
        np.testing.assert_array_almost_equal(aggregated_weights, expected_weights, decimal=3)

if __name__ == '__main__':
    unittest.main()
