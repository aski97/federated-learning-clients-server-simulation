import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from abc import ABC, abstractmethod
import struct
import numpy as np
import socket
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from src.CSUtils import MessageType, build_message, unpack_message


class TCPClient(ABC):

    def __init__(self, server_address, client_id: int):
        self.server_address = server_address
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.id = client_id
        self._is_profiling = False
        self._info_profiling = {
            'train_samples': 0,
            'test_samples': 0,
            'bytes_input': 0,
            'training_n_instructions': 0,
            'training_execution_time': 0,
            'max_ram_used': 0}
        self.weights = None

        self._shuffle_dataset_before_training = False

        self.evaluation_data_federated_model = np.empty((0, 2), dtype=float)
        self.evaluation_data_training_model = np.empty((0, 2), dtype=float)
        _n_classes = self.get_num_classes()
        self.confusion_matrix_federated_model = np.empty((0, _n_classes, _n_classes), dtype=float)
        self.confusion_matrix_training_model = np.empty((0, _n_classes, _n_classes), dtype=float)
        # populate dataset
        self.x_train, self.x_test, self.y_train, self.y_test = self.load_dataset()

    # METHODS

    def _connect(self) -> None:
        """Connect to the server"""
        self.socket.connect(self.server_address)

    def _manage_communication(self) -> None:
        """ It manages the message communication with the server"""
        while True:
            # Wait for Server message
            m_body, m_type, m_len = self._receive_message()

            # check if server closed the connection
            if m_body is None or m_type is None:
                break

            self._info_profiling['bytes_input'] += m_len

            # behave differently with respect to the type of message received
            match m_type:
                case MessageType.FEDERATED_WEIGHTS:
                    print("Received federated weights")
                    # get weights from server
                    weights = m_body["weights"]

                    # check for configurations (round 0)
                    if "configurations" in m_body:
                        self._is_profiling = m_body['configurations']['profiling']

                    # update model
                    self._update_weights(weights)
                    # evaluate model with new weights
                    self._evaluate_model()

                    # train model with new weights
                    if self._is_profiling:
                        import resource
                        import trace
                        import time

                        print("Tracing active")

                        start_time = time.time()

                        tracer = trace.Trace(
                            count=True,
                            trace=False,
                            timing=True)

                        tracer.runfunc(self._train_model)

                        stats = tracer.results()

                        execution_time = time.time() - start_time

                        n_instructions = sum(stats.counts.values())

                        used_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

                        self._info_profiling['training_n_instructions'] += n_instructions
                        self._info_profiling['training_execution_time'] += execution_time
                        self._info_profiling['max_ram_used'] = max(used_memory, self._info_profiling['max_ram_used'])

                    else:
                        self._train_model()
                    # send trained weights to the server
                    self._send_trained_weights()
                case MessageType.END_FL_TRAINING:
                    print("Received final federated weights. Federated training has finished.")
                    # get weights from server
                    weights = m_body["weights"]
                    # Update model
                    self._update_weights(weights)
                    # evaluate model with new weights
                    self._evaluate_model()
                    # send evaluation data to the server
                    self._send_kpi_data()
                    # Client disconnect to the server
                    break
                case _:
                    continue

    def _send_message(self, msg_type: MessageType, body: object) -> None:
        """
        Send a message to the server in {'type': '', 'body': ''} format
        :param msg_type: type of the message.
        :param body: body of the message.
        """
        msg_serialized = build_message(msg_type, body)
        self.socket.sendall(msg_serialized)

    def _receive_message(self) -> tuple:
        """
        It waits until a message from a server is received.
        :return: unpacked message {msg_type, msg_body} and message length
        """
        # Read message length by first 4 bytes
        msg_len_bytes = self.socket.recv(4)
        if not msg_len_bytes:
            return None, None, None
        msg_len = struct.unpack('!I', msg_len_bytes)[0]
        # Read the message data
        data = b''
        while len(data) < msg_len:
            packet = self.socket.recv(msg_len - len(data))
            if not packet:  # EOF
                break
            data += packet

        msg_body, msg_type = unpack_message(data)
        return msg_body, msg_type, msg_len

    def _close_connection(self) -> None:
        """It closes connection with the server"""
        self.socket.close()

    def _load_model_and_weights(self) -> keras.Model:
        """ It loads the local model to be trained"""
        model = self.get_skeleton_model()

        if self.weights is not None:
            model.set_weights(self.weights)

        optimizer = self.get_optimizer()
        loss = self.get_loss_function()
        metric = self.get_metric()

        model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

        return model

    def _update_weights(self, weights) -> None:
        """ It updates the weights of the local model"""
        self.weights = weights

    def _train_model(self) -> None:
        """
        It trains the model.
        """
        model = self._load_model_and_weights()

        if self._shuffle_dataset_before_training is True:
            # Shuffle training dataset
            indices = np.arange(self.x_train.shape[0])
            np.random.shuffle(indices)

            self.x_train = self.x_train[indices]
            self.y_train = self.y_train[indices]

        batch_size = self.get_batch_size()
        epochs = self.get_train_epochs()

        model.fit(x=self.x_train, y=self.y_train, batch_size=batch_size, epochs=epochs, verbose=1)

        # set trained weights
        self._update_weights(np.array(model.get_weights(), dtype='object'))
        # evaluate model
        self._evaluate_model(model)

    def _evaluate_model(self, model=None) -> np.ndarray:
        """
        It evaluates the model with test dataset.
        :param model: model to evaluate, otherwise it loads it.
        :return: numpy array of (accuracy, loss)
        """
        is_evaluating_fm = False

        if model is None:
            model = self._load_model_and_weights()
            is_evaluating_fm = True

        test_loss, test_acc = model.evaluate(self.x_test, self.y_test)

        # Confusion Matrix
        y_pred = model.predict(self.x_test)

        cm = confusion_matrix(np.argmax(self.y_test, axis=1), np.argmax(y_pred, axis=1), normalize="true",
                              labels=list(range(self.get_num_classes())))

        cm_percentage = np.round(cm, 2)

        print(f'Test accuracy: {test_acc}')
        print("Confusion Matrix (Percentage):")
        print(cm_percentage)

        evaluation_data = np.array([[test_acc, test_loss]])

        if is_evaluating_fm is True:
            # It's using federated weights
            self.evaluation_data_federated_model = np.append(self.evaluation_data_federated_model, evaluation_data,
                                                             axis=0)
            self.confusion_matrix_federated_model = np.append(self.confusion_matrix_federated_model, [cm_percentage],
                                                              axis=0)

        else:
            # It's using training weights
            self.evaluation_data_training_model = np.append(self.evaluation_data_training_model, evaluation_data,
                                                            axis=0)
            self.confusion_matrix_training_model = np.append(self.confusion_matrix_training_model, [cm_percentage],
                                                             axis=0)

        return evaluation_data

    def _send_trained_weights(self) -> None:
        """Send trained weights to the server"""
        msg_body = {'client_id': self.id, 'weights': self.weights}
        self._send_message(MessageType.CLIENT_TRAINED_WEIGHTS, msg_body)

    def _send_kpi_data(self) -> None:
        """Send evaluation data to the server"""
        msg_body = {'client_id': self.id,
                    'evaluation_federated': self.evaluation_data_federated_model,
                    'evaluation_training': self.evaluation_data_training_model,
                    'cm_federated': self.confusion_matrix_federated_model,
                    'cm_training': self.confusion_matrix_training_model}

        if self._is_profiling:
            self._info_profiling['train_samples'] = len(self.y_train)
            self._info_profiling['test_samples'] = len(self.y_test)
            msg_body['info_profiling'] = self._info_profiling

        self._send_message(MessageType.CLIENT_EVALUATION, msg_body)

    def run(self) -> None:
        """
        Execute client tasks
        """
        try:
            # Connection to the Server
            self._connect()
            # Manage communication with the Server
            self._manage_communication()

        except socket.error as e:
            print(f"Socket error: {e}")
        finally:
            # Close socket
            self._close_connection()

    @staticmethod
    def enable_op_determinism() -> None:
        """ The training process uses deterministic operation in order to have the same experimental results"""
        tf.keras.utils.set_random_seed(1)  # sets seeds for base-python, numpy and tf
        tf.config.experimental.enable_op_determinism()

    def shuffle_dataset_before_training(self, value: bool) -> None:
        """If True it shuffles the training dataset randomly before training the model."""
        self._shuffle_dataset_before_training = value

    @abstractmethod
    def load_dataset(self) -> tuple:
        """
        It loads client dataset
        :return: x_train, x_test, y_train, y_test
        """
        pass

    @abstractmethod
    def get_skeleton_model(self) -> keras.Model:
        """
        Get the skeleton of the model
        :return: keras model
        """
        pass

    @abstractmethod
    def get_optimizer(self) -> keras.optimizers.Optimizer | str:
        """
        Get the optimizer of the model
        :return: keras optimizer
        """
        pass

    @abstractmethod
    def get_loss_function(self) -> keras.losses.Loss | str:
        """
        Get the loss of the model
        :return: keras loss
        """
        pass

    @abstractmethod
    def get_metric(self) -> keras.metrics.Metric | str:
        """
        Get the metric for the evaluation
        :return: keras metric
        """
        pass

    @abstractmethod
    def get_batch_size(self) -> int:
        pass

    @abstractmethod
    def get_train_epochs(self) -> int:
        pass

    @abstractmethod
    def get_num_classes(self) -> int:
        """Get the number of classes managed by the dataset"""
        pass

