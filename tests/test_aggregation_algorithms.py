import unittest
import numpy as np

from src.AggregationAlgorithm import FedAvg, FedMiddleAvg, FedAvgServerMomentum


class TestAggregationAlgorithms(unittest.TestCase):

    def setUp(self):
        """
        This method will be run before each test. It sets up some common variables.
        """
        self.clients_weights = {
            'client1': {
                'weights': np.array([1, 2, 3]),
                'n_training_samples': 10
            },
            'client2': {
                'weights': np.array([7, 8, 9]),
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

    def test_fedavg_server_momentum(self):
        """
        Test the FedAvgServerMomentum algorithm.
        """
        fedavg_server_momentum = FedAvgServerMomentum(beta=0.9, learning_rate=0.1, weighted=True)
        fed_avg = (self.clients_weights['client1']['weights'] * 10 +
                   self.clients_weights['client2']['weights'] * 20) / 30
        delta = fed_avg - self.federated_model
        expected_weights = self.federated_model + 0.1 * delta
        aggregated_weights = fedavg_server_momentum.aggregate_weights(self.clients_weights, self.federated_model)
        np.testing.assert_array_almost_equal(aggregated_weights, expected_weights)


if __name__ == '__main__':
    unittest.main()