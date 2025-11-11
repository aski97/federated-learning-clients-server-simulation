import os
import logging

import time 

try:
    from src.federated_sim.utils.MetricsCalculator import MetricsCalculator
    from src.federated_sim.utils.ResultsLogger import ResultsLogger
    from src.federated_sim.utils.PlotGenerator import PlotGenerator
except ImportError:
    print("Could not import custom utils. Please ensure they are in the correct path.")
    exit(1)

from src.federated_sim.AggregationAlgorithm import AggregationAlgorithm, FedAvg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from abc import ABC, abstractmethod
import numpy as np
import socket
import threading
import struct
from src.federated_sim.CSUtils import MessageType, build_message, unpack_message
import tensorflow as tf
from tensorflow.keras.models import Model


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

# Module-level logger
logger = logging.getLogger(__name__)


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
        self.weights = None
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

        # graceful shutdown event
        self._stop_event = threading.Event()
        # flag that indicates FL rounds finished (used to trigger final evaluations)
        self._rounds_finished = False

        # per-server logger
        self.logger = logging.getLogger(f"{__name__}.server")
        self.logger.info("TCPServer initialized on %s expecting %d clients for %d rounds", self._server_address, self.number_clients, self.number_rounds)

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
        try:
            recipient.sendall(msg_serialized)
            logger.debug("Sent message type=%s bytes=%d to %s", msg_type, len(msg_serialized), recipient)
        except Exception as e:
            logger.error("Error sending message type=%s to %s: %s", msg_type, recipient, e)
            raise

    @staticmethod
    def _receive_message(client_socket: socket.socket) -> tuple:
        """
        It waits until a message from a client is received.
        :param client_socket: client socket.
        :return: unpacked message and length (msg_body, msg_type, msg_length)
        """
        try:
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
        except socket.timeout:
            # Caller can treat None as no-message; timeout is expected when using settimeout for graceful shutdown
            logger.debug("Socket recv timed out (no data available)")
            return None, None, None
        except Exception as e:
            logger.error("Error receiving message from client socket %s: %s", client_socket, e)
            return None, None, None

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
                logger.debug("Client socket %s is active", client_socket)
                return True
            else:
                logger.debug("Client socket %s reported error code %s", client_socket, error_code)
                return False

        except socket.error as e:
            # For any exception socket is not active
            logger.error("Error during the check of the socket: %s", e)
            return False

    def _initialize_server(self) -> None:
        """
        It configures and starts the server
        :return:
        """
        self.socket.bind(self.server_address)
        self.logger.info("Bound server socket to %s", self.server_address)

    def _wait_for_clients(self) -> None:
        """It waits client connections"""
        self.socket.listen(self.number_clients)
        # short timeout to allow clean shutdown checks in accept loop
        self.socket.settimeout(1.0)
        self.logger.info("Server listening on %s:%d (max clients=%d)", *self.server_address, self.number_clients)

    def _create_server_threads(self) -> None:
        """
        It creates three threads needed by the server:
        - A thread to manage connections with clients
        - A thread to execute Federated algorithms
        - A thread to manage evaluations of the Federated Learning
        """
        # Thread that accepts connections with clients
        self.thread_client_connections = threading.Thread(target=self._handle_accept_connections, daemon=True, name="accept-connections")
        self.thread_client_connections.start()
        self.logger.debug("Started thread to accept client connections")

        self.thread_fl_algorithms = threading.Thread(target=self._handle_round_fl, daemon=True, name="fl-algorithms")
        self.thread_fl_algorithms.start()
        self.logger.debug("Started thread to handle federated rounds")

        # BUGFIX: do not call the function here, pass it as target
        self.thread_final_evaluations = threading.Thread(target=self._handle_final_evaluations, daemon=True, name="final-evaluations")
        self.thread_final_evaluations.start()
        self.logger.debug("Started thread to handle final evaluations")

    def _join_server_threads(self) -> None:
        """ It waits the end of the threads"""
        self.logger.debug("Waiting for %d client threads to finish", len(self.client_threads))
        for client_thread in self.client_threads:
            client_thread.join()
            self.logger.debug("Client thread %s joined", client_thread.name)

        if self.thread_client_connections:
            self.thread_client_connections.join()
            self.logger.debug("Client connections thread joined")
        if self.thread_fl_algorithms:
            self.thread_fl_algorithms.join()
            self.logger.debug("FL algorithms thread joined")
        if self.thread_final_evaluations:
            self.thread_final_evaluations.join()
            self.logger.debug("Final evaluations thread joined")

    def _handle_client(self, client_socket: socket.socket, client_address: tuple) -> None:
        """
        It manages client communications.
        :param client_socket: socket of the client
        :param client_address: address of the client
        """
        try:
            self.logger.info("Handling client %s", client_address)
            # send updated weights to the client
            self._send_fl_model_to_client(client_socket)

            while True:
                # Wait for client message
                m_body, m_type, m_len = self._receive_message(client_socket)

                # check if client closed the connection
                if m_body is None or m_type is None:
                    self.logger.info("Client %s closed connection or sent empty message", client_address)
                    break

                # specify the type of the received message
                m_body: dict

                # extrapolate client id from the message
                if "client_id" in m_body:
                    client_id = m_body.pop("client_id")
                else:
                    self.logger.warning("Received message without client_id from %s, ignoring", client_address)
                    break

                self.logger.debug("Received message type=%s from client_id=%s bytes=%d", m_type, client_id, m_len)

                if client_id in self._output_bytes_clients:
                    self._output_bytes_clients[client_id] += m_len
                else:
                    self._output_bytes_clients[client_id] = m_len

                # behave differently with respect to the type of message received
                match m_type:
                    case MessageType.CLIENT_MODEL:
                        # Received trained weights from the client
                        with self.condition_add_weights:
                            weights = m_body["weights"]
                            n_training_samples = m_body["n_training_samples"]
                            local_gradients = m_body["gradients"]
                            self.client_weights[client_id] = {"weights": weights,
                                                              "gradients": local_gradients,
                                                              "n_training_samples": n_training_samples}
                            self.logger.info("Received CLIENT_MODEL from client %s (#samples=%d)", client_id, n_training_samples)
                            # Notify the server thread
                            self.condition_add_weights.notify()
                    case MessageType.CLIENT_EVALUATION:
                        with self.condition_add_client_evaluation:
                            self.clients_evaluations[client_id] = m_body
                            self.logger.info("Received CLIENT_EVALUATION from client %s", client_id)
                            # Notify the server thread
                            self.condition_add_client_evaluation.notify()
                    case _:
                        self.logger.debug("Unhandled message type %s from client %s", m_type, client_id)
                        continue

        except (socket.error, BrokenPipeError) as e:
            self.logger.error("Connection error with client %s: %s", client_address, e)
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
        self.logger.info("Closed connection for client thread %s", client_thread.name)

        # remove thread from the list
        try:
            self.client_threads.remove(client_thread)
        except ValueError:
            self.logger.debug("Client thread %s not found in list when removing", client_thread.name)

        # remove socket from the list
        try:
            self._client_sockets.remove(client_socket)
        except ValueError:
            self.logger.debug("Client socket not found in list when removing")

    def _handle_accept_connections(self) -> None:
        """
        It accepts connections from clients. For each client it creates a new thread
        to manage the communication client<-->server.
        """
        n_client = 0
        while n_client < self.number_clients and not self._stop_event.is_set():
            try:
                # Client connection (may timeout to allow stop checks)
                client_socket, client_address = self.socket.accept()
            except socket.timeout:
                continue
            except Exception as e:
                self.logger.error("Error accepting client connection: %s", e)
                continue

            # Create a thread to handle the client
            client_thread = threading.Thread(target=self._handle_client,
                                             args=(client_socket, client_address),
                                             daemon=True,
                                             name=f"client-{n_client}")
            client_thread.start()

            self.logger.info("Client connected: %s", client_address)
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
            while not self._stop_event.is_set():
                self.logger.debug("Waiting for client weights (received %d/%d)", len(self.client_weights), self.number_clients)
                self.condition_add_weights.wait(timeout=1.0)
                # Check if all clients involved have sent trained weights
                if len(self.client_weights) >= self.number_clients:
                    # increase round
                    self.actual_round += 1
                    self.logger.info("Starting aggregation for round %d", self.actual_round)
                    # aggregate weights
                    try:
                        self._aggregate_weights()
                    except Exception as e:
                        self.logger.error("Error during aggregation: %s", e)
                        continue
                    self.logger.info("Aggregation for round %d completed", self.actual_round)
                    # send new model to clients
                    self._send_fl_model_to_clients()
                    self.logger.info("Sent federated model to clients for round %d", self.actual_round)
                    # check if it's finished
                    if self.actual_round >= self.number_rounds:
                        # FL ENDED
                        self.logger.info("Reached configured number of rounds (%d). Federated Learning finished.", self.number_rounds)
                        if self._save_weights_path is not None:
                            self.save_federated_weights(self._save_weights_path)
                            self.logger.info("Saved federated weights to %s", self._save_weights_path)
                        # mark rounds finished and notify final evaluation thread (do NOT set _stop_event here)
                        self._rounds_finished = True
                        with self.condition_add_client_evaluation:
                            self.condition_add_client_evaluation.notify_all()
                        break

    def _handle_final_evaluations(self) -> None:
        """
        It manages the final evaluations of Federated Learning,
        delegating processing, logging, and plotting to specialized classes.
        """
        with self.condition_add_client_evaluation:
            while True:
                self.logger.debug("Waiting for client evaluations (received %d/%d)", 
                                 len(self.clients_evaluations), self.number_clients)
                
                # Wait with a timeout to allow _stop_event checks
                self.condition_add_client_evaluation.wait(timeout=1.0)

                # If all evaluations received -> process and then request shutdown
                if len(self.clients_evaluations) == self.number_clients:
                    self.logger.info("All %d client evaluations received. Processing final results...", 
                                     self.number_clients)
                    
                    try:
                        # 1. Initialize helper classes
                        agg_name = self.aggregation_algorithm.__class__.__name__
                        class_names = self.get_classes_name()

                        calculator = MetricsCalculator(
                            aggregation_name=agg_name,
                            num_rounds=self.number_rounds,
                            classes_name=class_names
                        )
                        reporter = ResultsLogger(logger=self.logger)
                        plotter = PlotGenerator(
                            aggregation_name=agg_name,
                            classes_name=class_names,
                            num_rounds=self.number_rounds
                        )

                        # 2. Process: Delegate calculation
                        processed_data = calculator.process(self.clients_evaluations)

                        # 3. Report: Delegate logging
                        reporter.log_summary(processed_data)
                        reporter.log_client_profiling(
                            self.clients_evaluations,
                            self._output_bytes_clients,
                            self._clients_profiling_enabled
                        )

                        # 4. Plot: Delegate visualization
                        if self._evaluation_plots_enabled:
                            plotter.plot_all(
                                processed_data,
                                self.clients_evaluations,
                                self._clients_profiling_enabled
                            )
                        else:
                            self.logger.info("Evaluation plots are disabled. Skipping plot generation.")

                    except Exception as e:
                        self.logger.critical("A critical error occurred during final evaluation processing: %s", e, exc_info=True)
                    
                    finally:
                        # 5. Signal server shutdown
                        self.logger.info("Final evaluations finished. Signalling server to stop.")
                        self._stop_event.set()
                        break # Exit the wait loop
                # Handle case where rounds finished but not all clients reported
                elif self._rounds_finished:
                    if len(self.clients_evaluations) > 0:
                        self.logger.warning("Rounds finished but received only %d/%d client evaluations. Proceeding with available data in 1s...",
                                            len(self.clients_evaluations), self.number_clients)
                        time.sleep(1.0) # Give a short grace period
                        
                    else:
                        self.logger.warning("Rounds finished but no evaluations received. Shutting down.")
                        time.sleep(1.0) # Give a short grace period

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
        self.logger.debug("Sent %s to client socket %s (round=%d)", msg_type, client_socket, self.actual_round)
        # print("Sent updated weights to client")

    def _send_fl_model_to_clients(self) -> None:
        """Send the federated model (weights) to all clients"""
        removed = []
        for client_socket in list(self._client_sockets):
            # Send only if the client is connected
            if self._is_client_active(client_socket):
                self._send_fl_model_to_client(client_socket)
            else:
                removed.append(client_socket)
                try:
                    self._client_sockets.remove(client_socket)
                except ValueError:
                    self.logger.debug("Tried to remove non-existing client socket %s", client_socket)
        if removed:
            self.logger.warning("Removed %d inactive client sockets", len(removed))

    def _initialize_federated_model(self) -> list:
        """
        Initialize weights of the model to send to clients. The result
        is affected by "kernel_initializer" of the keras model layers.
        :return: weights as list of numpy arrays
        :rtype: list[numpy.ndarray]
        """
        model = self.get_skeleton_model()

        # initialize weights as list (keras returns list)
        weights = model.get_weights()
        self.logger.info("Initialized federated model with %d weight tensors", len(weights))
        return weights

    def _aggregate_weights(self) -> None:
        """
        It aggregates weights from clients computing the mean of the weights.
        """
        self.logger.debug("Aggregating weights from %d clients for round %d", len(self.client_weights), self.actual_round + 1)
        aggregated = self.aggregation_algorithm.aggregate_weights(self.client_weights, self.weights)
        # ensure stored representation is a plain list[numpy.ndarray]
        self.weights = list(aggregated)
        self.client_weights.clear()
        self.logger.info("Aggregation completed for round %d", self.actual_round)

    def run(self) -> None:
        """
        Execute server tasks
        """
        if self.weights is None:
            self.weights = self._initialize_federated_model()

        try:
            # bind server
            self._initialize_server()
            # listen to clients connections
            self._wait_for_clients()
            # Create server threads
            self._create_server_threads()
            self.logger.info("Server run sequence started, waiting for stop signal")
            # Block until stop event is set by FL thread or by external interrupt
            self._stop_event.wait()
        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt received. Shutting down server.")
            self._stop_event.set()
        finally:
            # request stop for threads and join
            self._stop_event.set()
            self._join_server_threads()
            # Close server socket
            try:
                self.socket.close()
            except Exception as e:
                self.logger.debug("Error closing server socket: %s", e)
            self.logger.info("Server socket closed")

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

        # save as object-array (allow_pickle=True) to preserve a list of arrays
        np.save(file_path, np.array(self.weights, dtype=object), allow_pickle=True)

    def load_initial_weights(self, file_path) -> None:
        """
        It initializes the federated model with the loaded weights at the specified path.
        :param file_path: path of the weights.
        """
        weights = np.load(file_path, allow_pickle=True)

        # if weights.shape != self.weights.shape:
        #     raise ValueError("Your model doesn't accept this weights. The shapes are not matching.")

        self.weights = weights

    @staticmethod
    def enable_op_determinism() -> None:
        """ Used to have the same initialization of the federated Model"""
        tf.keras.utils.set_random_seed(1)  # sets seeds for base-python, numpy and tf
        tf.config.experimental.enable_op_determinism()

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
