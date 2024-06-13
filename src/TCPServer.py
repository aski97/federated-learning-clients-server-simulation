import os

from src.AggregationAlgorithm import AggregationAlgorithm, FedAvg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from abc import ABC, abstractmethod
import numpy as np
import socket
import threading
import struct
from src.CSUtils import MessageType, build_message, unpack_message
from tensorflow.keras.models import Model


class TCPServer(ABC):

    def __init__(self, address, number_clients: int, number_rounds: int, save_weights_path: str = None):
        self._server_address = address
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._client_sockets = []  # list of client sockets
        self._clients_profiling_enabled = False
        self._evaluation_plots_enabled = True
        self._save_weights_path = save_weights_path
        # FL
        self.aggregation_algorithm = FedAvg()
        self.client_weights = {}  # shared variable
        self.clients_evaluations = {}  # shared variable
        self.weights = self._initialize_federated_model()
        self.actual_round = 0
        self.number_clients = number_clients
        self.number_rounds = number_rounds
        self._output_bytes_clients = {}
        # threads
        self.client_threads = []  # list of client threads connected to the server
        self.thread_client_connections = None
        self.thread_fl_algorithms = None
        self.thread_final_evaluations = None
        # conditions to access to shared variables
        self.condition_add_weights = threading.Condition()
        self.condition_add_client_evaluation = threading.Condition()

    # PROPERTIES

    @property
    def server_address(self):
        return self._server_address

    # METHODS

    @staticmethod
    def _send_message(recipient: socket.socket, msg_type: MessageType, body: object) -> None:
        """
        Send a message to a client in {'type': '', 'body': ''} format
        :param recipient: recipient socket.
        :param msg_type: type of the message.
        :param  body: body of the message.
        """
        msg_serialized = build_message(msg_type, body)
        recipient.sendall(msg_serialized)

    @staticmethod
    def _receive_message(client_socket: socket.socket) -> tuple:
        """
        It waits until a message from a client is received.
        :param client_socket: client socket.
        :return: unpacked message and length (msg_type, msg_body, msg_length)
        """
        # Read the message length by first 4 bytes
        msg_len_bytes = client_socket.recv(4)
        if not msg_len_bytes:  # EOF
            return None, None, None
        msg_len = struct.unpack('!I', msg_len_bytes)[0]
        # Read the message data
        data = b''
        while len(data) < msg_len:
            packet = client_socket.recv(msg_len - len(data))
            if not packet:  # EOF
                break
            data += packet

        msg_body, msg_type = unpack_message(data)
        return msg_body, msg_type, msg_len

    @staticmethod
    def _is_client_active(client_socket: socket.socket) -> bool:
        """
        Check if the client socket is active.
        :param client_socket: socket to check.
        :return: True if is active, False otherwise.
        """
        try:
            # Get SO_ERROR
            error_code = client_socket.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)

            # If no errors the socket is active
            if error_code == 0:
                return True
            else:
                return False

        except socket.error as e:
            # For any exception socket is not active
            print(f"Error during the check of the socket: {e}")
            return False

    def _initialize_server(self) -> None:
        """
        It configures and starts the server
        :return:
        """
        self.socket.bind(self.server_address)

    def _wait_for_clients(self) -> None:
        """It waits client connections"""
        self.socket.listen(self.number_clients)
        print("Server active on {}:{}".format(*self.server_address))

    def _create_server_threads(self) -> None:
        """
        It creates three threads needed by the server:
        - A thread to manage connections with clients
        - A thread to execute Federated algorithms
        - A thread to manage evaluations of the Federated Learning
        """
        # Thread that accepts connections with clients
        self.thread_client_connections = threading.Thread(target=self._handle_accept_connections)
        self.thread_client_connections.start()

        self.thread_fl_algorithms = threading.Thread(target=self._handle_round_fl)
        self.thread_fl_algorithms.start()

        self.thread_final_evaluations = threading.Thread(target=self._handle_final_evaluations())
        self.thread_final_evaluations.start()

    def _join_server_threads(self) -> None:
        """ It waits the end of the threads"""
        for client_thread in self.client_threads:
            client_thread.join()

        self.thread_client_connections.join()
        self.thread_fl_algorithms.join()
        self.thread_final_evaluations.join()

    def _handle_client(self, client_socket: socket.socket, client_address: tuple) -> None:
        """
        It manages client communications.
        :param client_socket: socket of the client
        :param client_address: address of the client
        """
        try:
            # send updated weights to the client
            self._send_fl_model_to_client(client_socket)

            while True:
                # Wait for client message
                m_body, m_type, m_len = self._receive_message(client_socket)

                # check if client closed the connection
                if m_body is None or m_type is None:
                    break

                # specify the type of the received message
                m_body: dict

                # extrapolate client id from the message
                if "client_id" in m_body:
                    client_id = m_body.pop("client_id")
                else:
                    break

                if client_id in self._output_bytes_clients:
                    self._output_bytes_clients[client_id] += m_len
                else:
                    self._output_bytes_clients[client_id] = m_len

                # behave differently with respect to the type of message received
                match m_type:
                    case MessageType.CLIENT_TRAINED_WEIGHTS:
                        # Received trained weights from the client
                        # print("Received trained weights")
                        # Add weights to the shared variable
                        with self.condition_add_weights:
                            weights = m_body["weights"]
                            n_training_samples = m_body["n_training_samples"]
                            self.client_weights[client_id] = {"weights": weights,
                                                              "n_training_samples": n_training_samples}
                            # Notify the server thread
                            self.condition_add_weights.notify()
                    case MessageType.CLIENT_EVALUATION:
                        # print("Received evaluation")
                        with self.condition_add_client_evaluation:
                            self.clients_evaluations[client_id] = m_body
                            # Notify the server thread
                            self.condition_add_client_evaluation.notify()
                    case _:
                        continue

        except (socket.error, BrokenPipeError) as e:
            print(f"Error connection to the client {client_socket}: {e}")
        finally:
            # Close connection
            self._close_client_socket(client_socket, threading.current_thread())

    def _close_client_socket(self, client_socket: socket.socket, client_thread: threading.Thread) -> None:
        """
        Close the client connection.
        :param client_socket: the socket of the client.
        :param client_thread: thread that handles the client communication.
        """
        client_socket.close()
        # print("Close connection with {}".format(client_address))

        # remove thread from the list
        self.client_threads.remove(client_thread)

        # remove socket from the list
        self._client_sockets.remove(client_socket)

    def _handle_accept_connections(self) -> None:
        """
        It accepts connections from clients. For each client it creates a new thread
        to manage the communication client<-->server.
        """
        n_client = 0
        while n_client < self.number_clients:
            # Client connection
            client_socket, client_address = self.socket.accept()

            # Create a thread to handle the client
            client_thread = threading.Thread(target=self._handle_client,
                                             args=(client_socket, client_address))
            client_thread.start()

            print(f"{client_address} connected")
            # Add the thread to the list
            self.client_threads.append(client_thread)
            # Add socket to the list
            self._client_sockets.append(client_socket)
            n_client += 1

    def _handle_round_fl(self) -> None:
        """
        It manages rounds for the federated learning:
        it aggregates weights and send the new model to the clients.
        """
        with self.condition_add_weights:
            while True:
                self.condition_add_weights.wait()
                # Check if all clients involved have sent trained weights
                if len(self.client_weights) >= self.number_clients:
                    # increase round
                    self.actual_round += 1
                    # aggregate weights
                    self._aggregate_weights()
                    # send new model to clients
                    self._send_fl_model_to_clients()
                    # check if it's finished
                    if self.actual_round >= self.number_rounds:
                        # FL ENDED
                        if self._save_weights_path is not None:
                            self.save_federated_weights(self._save_weights_path)

    def _handle_final_evaluations(self) -> None:
        """
        It manages the final evaluations of Federated Learning,
        displaying the results and generating any graphs if necessary.
        """
        with self.condition_add_client_evaluation:
            while True:
                self.condition_add_client_evaluation.wait()

                # Check if all clients involved have sent the evaluation
                if len(self.clients_evaluations) == self.number_clients:

                    def get_federated_average_metrics(metric_type, values):
                        data_per_client = []

                        for key, value in values.items():
                            eval_federated = value['evaluation_federated']
                            metric_values = [el[0 if metric_type == "accuracy" else 1] for el in eval_federated]
                            data_per_client.append(metric_values)

                        values_per_client_np = np.array(data_per_client)
                        mean = np.mean(values_per_client_np, axis=0)
                        return mean

                    def print_clients_profiling_data():
                        for key, value in self.clients_evaluations.items():
                            client_id = key
                            profiling_data = value['info_profiling']

                            output_bytes = self._output_bytes_clients[client_id]
                            input_bytes = profiling_data['bytes_input']
                            train_samples = profiling_data['train_samples']
                            test_samples = profiling_data['test_samples']
                            n_i = profiling_data['training_n_instructions']
                            e_t = profiling_data['training_execution_time']
                            ram_used = profiling_data['max_ram_used']

                            print(f"Profiling Client {client_id} -> "
                                  f"input_bytes = {input_bytes} B"
                                  f"|| output_bytes = {output_bytes} B"
                                  f"|| #instructions = {n_i} "
                                  f"|| execution_time = {e_t} s "
                                  f"|| max_ram_used = {ram_used / (1024.0 * 1024.0)} GB "
                                  f"|| #train_samples = {train_samples} "
                                  f"!! #test_samples = {test_samples} ")

                    def plot_profiling_data(data_type, title, y_label):
                        from matplotlib import pyplot as plt

                        clients = []
                        data = []
                        plt.title(title)
                        for key, value in self.clients_evaluations.items():
                            client_id = key
                            profiling_data = value['info_profiling']

                            clients.append(f"C{client_id}")
                            val = profiling_data[data_type]

                            if data_type == "max_ram_used":
                                val = val / (1024.0 * 1024.0)  # convert from KB to GB

                            data.append(val)

                        plt.stem(clients, data)

                        plt.ylabel(y_label)

                        plt.show()

                    def plot_metric_per_client(metric_type):
                        from matplotlib import pyplot as plt

                        fig, ax = plt.subplots()
                        fig.suptitle(f'{metric_type.capitalize()} on test samples after receiving Federated Model')

                        for key, value in self.clients_evaluations.items():
                            client_id = key
                            eval_federated = value['evaluation_federated']
                            metric_values = [el[0 if metric_type == "accuracy" else 1] for el in eval_federated]
                            round_numbers = list(range(len(eval_federated)))

                            ax.plot(round_numbers, metric_values, label=f'Client {client_id}', marker='o')

                        ax.set_xlabel('Rounds')
                        ax.set_ylabel(metric_type.capitalize())
                        ax.legend()

                        # plt.savefig(f'plots/{metric_type}_per_client.png')

                        plt.show()

                    def plot_average_metric(metric_type, values):
                        from matplotlib import pyplot as plt

                        rounds = list(range(len(values)))

                        fig, ax = plt.subplots()
                        fig.suptitle(f'Average {metric_type} of Federated Model per round')

                        ax.plot(rounds, values, label=f'Federated Model', marker='o')

                        ax.set_xlabel('Rounds')
                        ax.set_ylabel(f'{metric_type.capitalize()}')
                        ax.legend()

                        # plt.savefig(f'plots/{metric_type}_federated_model.png')
                        plt.show()

                    def plot_confusion_matrix(values, classes):
                        from matplotlib import pyplot as plt
                        import itertools

                        plt.imshow(values, interpolation='nearest', cmap=plt.colormaps["Reds"])
                        plt.title('Confusion matrix')
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

                    # compute federated average metrics
                    accuracy_avg = get_federated_average_metrics('accuracy', self.clients_evaluations)
                    loss_avg = get_federated_average_metrics('loss', self.clients_evaluations)

                    # confusion matrix
                    final_cm_per_client = [value['cm_federated'][-1] for value in self.clients_evaluations.values()]
                    cm_mean = np.round(np.mean(final_cm_per_client, axis=0), 2)

                    print(f"Average accuracy of final federated model: {accuracy_avg[-1]}\n")
                    print(f"Average loss of final federated model: {loss_avg[-1]}\n")
                    print("Average Confusion Matrix of final federated model (Percentage):")
                    print(cm_mean)
                    if self._clients_profiling_enabled:
                        import resource
                        max_m_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        print(f"Server max memory used: {max_m_used / (1024.0 * 1024.0)} GB")

                    print_clients_profiling_data()

                    if self._evaluation_plots_enabled:
                        plot_metric_per_client('accuracy')
                        plot_metric_per_client('loss')
                        plot_average_metric("accuracy", accuracy_avg)
                        plot_average_metric("loss", loss_avg)
                        plot_confusion_matrix(cm_mean, self.get_classes_name())

                        if self._clients_profiling_enabled:
                            plot_profiling_data('training_n_instructions',
                                                "Total number of instructions during all the training process",
                                                "# instructions")
                            plot_profiling_data('training_execution_time',
                                                "Total execution time of the training",
                                                "seconds")
                            plot_profiling_data('max_ram_used',
                                                "Max memory used during all the training process",
                                                "GB")

    def _send_fl_model_to_client(self, client_socket: socket.socket) -> None:
        """
        Send the federated weights to the client,
        informing either a round is running (so the client has to train the model with the new weights)
        or the federated learning has finished (so the client just evaluate the model with the new weights).
        :param client_socket: socket that handles the communication to the client
        """
        msg_type = MessageType.END_FL_TRAINING

        msg = {'weights': self.weights}

        if self.actual_round < self.number_rounds:
            msg_type = MessageType.FEDERATED_WEIGHTS

        if self.actual_round == 0 and self._clients_profiling_enabled:
            # the first message of the server contains some configurations
            msg['configurations'] = {'profiling': True}

        self._send_message(client_socket, msg_type, msg)
        # print("Sent updated weights to client")

    def _send_fl_model_to_clients(self) -> None:
        """Send the federated model (weights) to all clients"""
        for client_socket in self._client_sockets:
            # Send only if the client is connected
            if self._is_client_active(client_socket):
                self._send_fl_model_to_client(client_socket)
            else:
                self._client_sockets.remove(client_socket)

    def _initialize_federated_model(self) -> np.ndarray:
        """
        Initialize weights of the model to send to clients. The result
        is affected by "kernel_initializer" of the keras model layers.
        :return: weights
        :rtype: np.ndarray
        """
        model = self.get_skeleton_model()

        # initialize weights
        return np.array(model.get_weights(), dtype='object')

    def _aggregate_weights(self) -> None:
        """
        It aggregates weights from clients computing the mean of the weights.
        """

        self.weights = self.aggregation_algorithm.aggregate_weights(self.client_weights, self.weights)

        self.client_weights.clear()

    def run(self) -> None:
        """
        Execute server tasks
        """
        try:
            # bind server
            self._initialize_server()
            # listen to clients connections
            self._wait_for_clients()
            # Create server threads
            self._create_server_threads()
        finally:
            self._join_server_threads()
            # Close server socket
            self.socket.close()

    def enable_clients_profiling(self, value: bool) -> None:
        """
        It enables the profiling of the clients in order to receive KPIs from clients
        """
        self._clients_profiling_enabled = value

    def enable_evaluations_plots(self, value: bool) -> None:
        """
        If enabled it plots graphs of the evaluation data and KPI
        """
        self._evaluation_plots_enabled = value

    def set_aggregation_algorithm(self, aggregation_algorithm: AggregationAlgorithm) -> None:
        """
        It sets the aggregation algorithm to use in the federated learning process.
        """
        self.aggregation_algorithm = aggregation_algorithm

    def save_federated_weights(self, file_path) -> None:
        """
        It saves the federated weights at the specified path.
        :param file_path: path where to save the weights
        """
        directory = os.path.dirname(file_path)

        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save(file_path, self.weights)

    def load_initial_weights(self, file_path) -> None:
        """
        It initializes the federated model with the loaded weights at the specified path.
        :param file_path: path of the weights.
        """
        weights = np.load(file_path, allow_pickle=True)

        if weights.shape != self.weights.shape:
            raise ValueError("Your model doesn't accept this weights. The shapes are not matching.")

        self.weights = weights

    @abstractmethod
    def get_skeleton_model(self) -> Model:
        """
        Get the skeleton of the model
        :return: keras Model
        """
        pass

    @abstractmethod
    def get_classes_name(self) -> list[str]:
        """ Get the list of names of the classes to predict"""
        pass
