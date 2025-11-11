import numpy as np
from matplotlib import pyplot as plt
import itertools
import os
import logging

# Module-level logger
logger = logging.getLogger(__name__)

class PlotGenerator:
    """
    Generates and saves all plots for federated learning evaluation.
    """
    def __init__(self, aggregation_name: str, classes_name: list[str], num_rounds: int):
        self.aggregation_name = aggregation_name
        self.classes_name = classes_name
        self.num_rounds = num_rounds

    def plot_all(self, processed_data: dict, raw_evaluations: dict, profiling_enabled: bool):
        """Orchestrator method to generate all plots."""
        logger.info("Generating evaluation plots...")
        try:
            self._plot_metric_per_client('accuracy', raw_evaluations)
            self._plot_metric_per_client('loss', raw_evaluations)
            
            self._plot_average_metric("accuracy", processed_data['accuracy_avg'])
            self._plot_average_metric("loss", processed_data['loss_avg'])
            
            self._plot_confusion_matrix(processed_data['cm_mean'], self.classes_name)

            if profiling_enabled:
                self._plot_profiling_data('training_n_instructions',
                                            "Total instructions during training",
                                            "# instructions", raw_evaluations)
                self._plot_profiling_data('training_execution_time',
                                            "Total training execution time",
                                            "seconds", raw_evaluations)
                self._plot_profiling_data('max_ram_used',
                                            "Max memory used during training",
                                            "GB", raw_evaluations)
            
            logger.info("All plots generated.")
            
        except ImportError:
            logger.warning("matplotlib not installed. Skipping plot generation.")
        except Exception as e:
            logger.error("An error occurred during plot generation: %s", e)

    def _plot_metric_per_client(self, metric_type: str, raw_evaluations: dict):
        """Plots a given metric (accuracy/loss) over rounds for each client."""
        fig, ax = plt.subplots()
        fig.suptitle(f'{metric_type.capitalize()} on Test Samples (Per-Client)')

        for key, value in raw_evaluations.items():
            client_id = key
            eval_federated = value['evaluation_federated']
            metric_values = [el[0 if metric_type == "accuracy" else 1] for el in eval_federated]
            round_numbers = list(range(len(eval_federated)))
            ax.plot(round_numbers, metric_values, label=f'Client {client_id}', marker='o')

        ax.set_xlabel('Rounds')
        ax.set_ylabel(metric_type.capitalize())
        ax.legend()
        plt.show()

    def _plot_average_metric(self, metric_type: str, values: np.ndarray):
        """Plots the federated average of a metric over rounds."""
        rounds = list(range(len(values)))
        fig, ax = plt.subplots()
        fig.suptitle(f'Average {metric_type.capitalize()} of Federated Model per Round')
        ax.plot(rounds, values, label=f'Federated Model (Avg)', marker='o')
        ax.set_xlabel('Rounds')
        ax.set_ylabel(f'{metric_type.capitalize()}')
        ax.legend()
        plt.show()

        # Save the average values to a NumPy file
        directory = 'evaluations'
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, f'{metric_type}_{self.aggregation_name}_{self.num_rounds}rounds.npy')
        np.save(file_path, values)
        logger.debug("Saved average %s data to %s", metric_type, file_path)

    def _plot_confusion_matrix(self, values: np.ndarray, classes: list):
        """Plots the confusion matrix."""
        plt.figure()
        plt.imshow(values, interpolation='nearest', cmap=plt.colormaps.get("Reds"))
        plt.title('Average Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        thresh = values.max() / 2.
        for i, j in itertools.product(range(values.shape[0]), range(values.shape[1])):
            color = "white" if values[i, j] > thresh else "black"
            plt.text(j, i, str(values[i, j]), horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def _plot_profiling_data(self, data_type: str, title: str, y_label: str, raw_evaluations: dict):
        """Generates a stem plot for a given profiling metric."""
        clients = []
        data = []
        
        plt.figure()
        plt.title(title)
        
        for key, value in raw_evaluations.items():
            client_id = key
            clients.append(f"C{client_id}")

            if data_type == "training_execution_time":
                val = value.get("training_execution_time", 0)
            else:
                profiling_data = value.get('info_profiling', {})
                val = profiling_data.get(data_type, 0)

            if data_type == "max_ram_used":
                val = val / (1024.0 * 1024.0)  # convert from KB to GB

            data.append(val)

        plt.stem(clients, data)
        plt.ylabel(y_label)
        plt.show()