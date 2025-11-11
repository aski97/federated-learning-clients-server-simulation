import logging

class ResultsLogger:
    """
    Logs final evaluation summaries and client profiling data.
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_summary(self, processed_data: dict):
        """Logs the final federated model performance summary."""
        try:
            self.logger.info("Average accuracy of final federated model: %s", processed_data['accuracy_avg'][-1])
            self.logger.info("Average loss of final federated model: %s", processed_data['loss_avg'][-1])
            self.logger.info("Average Confusion Matrix of final federated model (Percentage):\n%s", processed_data['cm_mean'])
        except (KeyError, IndexError) as e:
            self.logger.error("Error logging summary, data may be incomplete: %s", e)

    def log_client_profiling(self, client_evaluations: dict, output_bytes: dict, profiling_enabled: bool):
        """Logs profiling information for each client."""
        self.logger.info("--- Client Profiling Data ---")
        for key, value in client_evaluations.items():
            client_id = key
            try:
                if profiling_enabled:
                    profiling_data = value.get('info_profiling', {})
                    b_out = output_bytes.get(client_id, 0)
                    b_in = profiling_data.get('bytes_input', 0)
                    train_samples = profiling_data.get('train_samples', 0)
                    test_samples = profiling_data.get('test_samples', 0)
                    n_i = profiling_data.get('training_n_instructions', 0)
                    e_t = value.get('training_execution_time', 0)
                    ram_used = profiling_data.get('max_ram_used', 0)

                    self.logger.info(
                        "Profiling Client %s -> input_bytes=%s B | output_bytes=%s B | #instructions=%s | "
                        "execution_time=%s s | max_ram_used=%s B | #train_samples=%s | #test_samples=%s",
                        client_id, b_in, b_out, n_i, e_t, ram_used, train_samples, test_samples
                    )
                else:
                    e_t = value.get('training_execution_time', 0)
                    self.logger.info("Client %s execution_time=%s s", client_id, e_t)
            except Exception as e:
                self.logger.error("Error logging profiling for client %s: %s", client_id, e)
        self.logger.info("-----------------------------")