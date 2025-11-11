import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """
    Calculates aggregate metrics from raw client evaluation data.
    """
    def __init__(self, aggregation_name: str, num_rounds: int, classes_name: list[str]):
        self.aggregation_name = aggregation_name
        self.num_rounds = num_rounds
        self.classes_name = classes_name
        self.n_classes = len(classes_name)

    def process(self, client_evaluations: dict) -> dict:
        """
        Processes raw client evaluations into aggregate metrics.
        
        Returns:
            A dictionary with 'accuracy_avg', 'loss_avg', and 'cm_mean'.
        """
        logger.debug("Calculating average metrics...")
        accuracy_avg = self._calculate_average_metrics('accuracy', client_evaluations)
        loss_avg = self._calculate_average_metrics('loss', client_evaluations)
        
        logger.debug("Calculating average confusion matrix...")
        cm_mean = self._calculate_average_confusion_matrix(client_evaluations)
        
        return {
            "accuracy_avg": accuracy_avg,
            "loss_avg": loss_avg,
            "cm_mean": cm_mean
        }

    def _calculate_average_metrics(self, metric_type: str, values: dict) -> np.ndarray:
        """
        Calculates average metrics across all clients and saves per-node data.
        """
        data_per_client = []
        for key, value in values.items():
            eval_federated = value['evaluation_federated']
            # el[0] is accuracy, el[1] is loss
            metric_values = [el[0 if metric_type == "accuracy" else 1] for el in eval_federated]
            data_per_client.append(metric_values)

        values_per_client_np = np.array(data_per_client)
        
        # Save the per-node values to a NumPy file
        directory = 'evaluations/nodes/'
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, f'{metric_type}_{self.aggregation_name}_{self.num_rounds}rounds.npy')
        np.save(file_path, values_per_client_np)
        logger.debug("Saved per-node %s data to %s", metric_type, file_path)

        # Return the mean across clients
        return np.mean(values_per_client_np, axis=0)

    def _calculate_average_confusion_matrix(self, values: dict) -> np.ndarray:
        """
        Calculates the mean confusion matrix across all clients.
        """
        # Get the *last* confusion matrix from each client
        final_cm_per_client = [value['cm_federated'][-1] for value in values.values()]
        
        sum_rows = np.zeros((self.n_classes, self.n_classes))
        count_rows = np.zeros(self.n_classes)

        for matrix in final_cm_per_client:
            for i in range(self.n_classes):
                # if the row (true class) has at least one prediction
                if np.any(matrix[i]):
                    sum_rows[i] += matrix[i]
                    count_rows[i] += 1

        average_rows = np.zeros((self.n_classes, self.n_classes))
        for i in range(self.n_classes):
            if count_rows[i] > 0:
                # Average the predictions for that true class
                average_rows[i] = sum_rows[i] / count_rows[i]

        return np.round(average_rows, 2)