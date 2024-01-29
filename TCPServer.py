import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import socket
import threading
import struct
import CSUtils
from matplotlib import pyplot as plt


class TCPServer:
    NUMBER_OF_CLIENTS = 10
    ROUNDS = 10

    def __init__(self, address):
        self.server_address = address
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_sockets = []  # list of client sockets
        # FL
        self.client_weights = {}  # shared variable
        self.clients_evaluations = {}  # shared variable
        self.weights = self.initialize_federated_model()
        self.round = 0
        # threads
        self.client_threads = []  # list of client threads connected to the server
        self.thread_client_connections = None
        self.thread_fl_algorithms = None
        self.thread_final_evaluations = None
        # conditions to access to shared variables
        self.condition_add_weights = threading.Condition()
        self.condition_add_client_evaluation = threading.Condition()

    def bind_and_listen(self) -> None:
        """It makes the server listening to server_address"""
        self.socket.bind(self.server_address)
        self.socket.listen(self.NUMBER_OF_CLIENTS)
        print("Server active on {}:{}".format(*self.server_address))

    @staticmethod
    def send_message(recipient: socket.socket, msg_type: CSUtils.MessageType, body: object) -> None:
        """
        Send a message to a client in {'type': '', 'body': ''} format
        :param recipient: recipient socket.
        :param msg_type: type of the message.
        :param  body: body of the message.
        """
        msg_serialized = CSUtils.build_message(msg_type, body)
        recipient.sendall(msg_serialized)

    @staticmethod
    def receive_message(client_socket: socket.socket) -> tuple:
        """
        It waits until a message from a client is received.
        :param client_socket: client socket.
        :return: unpacked message {msg_type, msg_body}.
        """
        # Read the message length by first 4 bytes
        msg_len_bytes = client_socket.recv(4)
        if not msg_len_bytes:  # EOF
            return None, None
        msg_len = struct.unpack('!I', msg_len_bytes)[0]
        # Read the message data
        data = b''
        while len(data) < msg_len:
            packet = client_socket.recv(msg_len - len(data))
            if not packet:  # EOF
                break
            data += packet

        return CSUtils.unpack_message(data)

    def create_server_threads(self) -> None:
        """
        It creates three threads needed by the server:
        - A thread to manage connections with clients
        - A thread to execute Federated algorithms
        - A thread to manage evaluations of the Federated Learning
        """
        # Thread that accepts connections with clients
        self.thread_client_connections = threading.Thread(target=self.handle_accept_connections)
        self.thread_client_connections.start()

        self.thread_fl_algorithms = threading.Thread(target=self.handle_round_fl)
        self.thread_fl_algorithms.start()

        self.thread_final_evaluations = threading.Thread(target=self.handle_final_evaluations())
        self.thread_final_evaluations.start()

    def join_server_threads(self) -> None:
        """ It joins the threads with the main thread """
        for client_thread in self.client_threads:
            client_thread.join()

        self.thread_client_connections.join()
        self.thread_fl_algorithms.join()
        self.thread_final_evaluations.join()

    @staticmethod
    def is_client_active(client_socket: socket.socket) -> bool:
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

    def handle_client(self, client_socket: socket.socket, client_address: tuple) -> None:
        """
        It manages client communications.
        :param client_socket: socket of the client
        :param client_address: address of the client
        """
        try:
            # send updated weights to the client
            self.send_fl_model_to_client(client_socket)

            while True:
                # Wait for client message
                m_body, m_type = self.receive_message(client_socket)

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

                # behave differently with respect to the type of message received
                match m_type:
                    case CSUtils.MessageType.CLIENT_TRAINED_WEIGHTS:
                        # Received trained weights from the client
                        # print("Received trained weights")
                        # Add weights to the shared variable
                        with self.condition_add_weights:
                            weights = m_body["weights"]
                            self.client_weights[client_id] = weights
                            # Notify the server thread
                            self.condition_add_weights.notify()
                    case CSUtils.MessageType.CLIENT_EVALUATION:
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
            self.close_client_socket(client_socket, client_address, threading.current_thread())

    def close_client_socket(self, client_socket: socket.socket, client_address: tuple,
                            client_thread: threading.Thread) -> None:
        """
        Close the client connection.
        :param client_address: client address.
        :param client_socket: the socket of the client.
        :param client_thread: thread that handles the client communication.
        """
        client_socket.close()
        # print("Close connection with {}".format(client_address))

        # remove thread from the list
        self.client_threads.remove(client_thread)

        # remove socket from the list
        self.client_sockets.remove(client_socket)

    def handle_accept_connections(self) -> None:
        """
        It accepts connections from clients. For each client it creates a new thread
        to manage the communication client<-->server.
        """
        n_client = 0
        while n_client < self.NUMBER_OF_CLIENTS:
            # Client connection
            client_socket, client_address = self.socket.accept()

            # Create a thread to handle the client
            client_thread = threading.Thread(target=self.handle_client,
                                             args=(client_socket, client_address))
            client_thread.start()

            # print(f"{client_address} connected")
            # Add the thread to the list
            self.client_threads.append(client_thread)
            # Add socket to the list
            self.client_sockets.append(client_socket)
            n_client += 1

    def handle_round_fl(self) -> None:
        """
        It starts a round for the federated learning:
        it aggregates weights and send the new model to the clients.
        """
        with self.condition_add_weights:
            while True:
                self.condition_add_weights.wait()
                # Check if all clients involved have sent trained weights
                if len(self.client_weights) >= self.NUMBER_OF_CLIENTS:
                    # aggregate weights
                    self.aggregate_weights()
                    # send new model to clients
                    self.send_fl_model_to_clients()
                    # increase round
                    self.round += 1

    def handle_final_evaluations(self) -> None:
        """
        When federated learning finishes it plots evaluations
        """
        with self.condition_add_client_evaluation:
            while True:
                self.condition_add_client_evaluation.wait()
                i = 4
                # Check if all clients involved have sent the evaluation
                if len(self.clients_evaluations) == self.NUMBER_OF_CLIENTS:

                    def plot_metric_per_client(metric_type, values):
                        fig, ax = plt.subplots()
                        fig.suptitle(f'{metric_type.capitalize()} on test samples after receiving Federated Model')

                        values_per_client = []

                        for key, value in values.items():
                            client_id = key
                            eval_federated = value['evaluation_federated']

                            metric_values = [el[0 if metric_type == "accuracy" else 1] for el in eval_federated]
                            values_per_client.append(metric_values)
                            round_numbers = list(range(len(eval_federated)))

                            ax.plot(round_numbers, metric_values, label=f'Client {client_id}', marker='o')

                        ax.set_xlabel('Rounds')
                        ax.set_ylabel(metric_type.capitalize())
                        ax.legend()

                        # plt.savefig(f'plots/{metric_type}_per_client.png')

                        plt.show()
                        values_per_client_np = np.array(values_per_client)
                        mean = np.mean(values_per_client_np, axis=0)
                        return mean

                    # Accuracy plot
                    ac_mean = plot_metric_per_client('accuracy', self.clients_evaluations)

                    # Loss plot
                    loss_mean = plot_metric_per_client('loss', self.clients_evaluations)

                    print(f"Average accuracy final federated model: {ac_mean[-1]}\n")
                    print(f"Average loss final federated model: {loss_mean[-1]}\n")

                    def plot_average(metric_type, values):
                        rounds = list(range(len(values)))

                        fig, ax = plt.subplots()
                        fig.suptitle(f'Average {metric_type} of Federated Model per round')

                        ax.plot(rounds, values, label=f'Federated Model', marker='o')

                        ax.set_xlabel('Rounds')
                        ax.set_ylabel(f'{metric_type.capitalize()}')
                        ax.legend()

                        # plt.savefig(f'plots/{metric_type}_federated_model.png')
                        plt.show()

                    # plot_average("accuracy", ac_mean)
                    # plot_average("loss", loss_mean)

    def send_fl_model_to_client(self, client_socket: socket.socket) -> None:
        """
        Send the federated weights to the client,
        informing either a round is running (so the client has to train the model with the new weights)
        or the federated learning has finished (so the client just evaluate the model with the new weights).
        :param client_socket: socket that handles the communication to the client
        """
        msg_type = CSUtils.MessageType.END_FL_TRAINING

        if self.round < self.ROUNDS - 1:
            msg_type = CSUtils.MessageType.FEDERATED_WEIGHTS

        self.send_message(client_socket, msg_type, self.weights)
        # print("Sent updated weights to client")

    def send_fl_model_to_clients(self) -> None:
        """Send the federated model (weights) to all clients"""
        for client_socket in self.client_sockets:
            # Send only if the client is connected
            if self.is_client_active(client_socket):
                self.send_fl_model_to_client(client_socket)
            else:
                self.client_sockets.remove(client_socket)

    @staticmethod
    def initialize_federated_model() -> np.ndarray:
        """
        Initialize weights of the model to send to clients.
        :return: weights
        :rtype: np.ndarray
        """

        model = CSUtils.get_skeleton_model()

        model_weights = model.get_weights()

        # initialize zero weights
        return np.array(model_weights, dtype='object') * 0

    def aggregate_weights(self) -> None:
        """
        It aggregates weights from clients computing the mean of the weights.
        """
        weights = self.client_weights[0]
        for i in range(1, len(self.client_weights)):
            weights = weights + np.array(self.client_weights[i], dtype='object')

        # aggregate weights, computing the mean
        self.weights = weights / len(self.client_weights)

        self.client_weights.clear()


if __name__ == "__main__":
    server_address = ('localhost', 12345)

    # Server creation and initialization
    server = TCPServer(server_address)

    try:
        # Binding and ready to listen
        server.bind_and_listen()
        # Create server threads
        server.create_server_threads()
    finally:
        server.join_server_threads()
        # Close server socket
        server.socket.close()
