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
import time

class TCPClient(ABC):

    def __init__(self, server_address, client_id: int):
        self.server_address = server_address
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.id = client_id
        self._is_profiling = False
        self._training_execution_time = 0
        self._info_profiling = {
            'train_samples': 0,
            'test_samples': 0,
            'bytes_input': 0,
            'training_n_instructions': 0,
            'max_ram_used': 0}
        self.weights = None
        self.gradients = None

        self.evaluation_data_federated_model = np.empty((0, 2), dtype=float)
        self.evaluation_data_training_model = np.empty((0, 2), dtype=float)
        _n_classes = self.get_num_classes()
        self.confusion_matrix_federated_model = np.empty((0, _n_classes, _n_classes), dtype=float)
        self.confusion_matrix_training_model = np.empty((0, _n_classes, _n_classes), dtype=float)
        # populate dataset
        self.x_train, self.x_test, self.y_train, self.y_test = self.load_dataset()

        self._model = self._load_compiled_model()
        self._loss_fn = self.get_loss_function()
        self._batch_size = self.get_batch_size()
        self._epochs = self.get_train_epochs()
        self._shuffle_dataset_each_epoch = True

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

                    start_time = time.time()

                    # train model with new weights
                    if self._is_profiling:
                        import resource
                        import trace

                        print("Tracing active")

                        tracer = trace.Trace(
                            count=True,
                            trace=False,
                            timing=True)

                        tracer.runfunc(self._train_model)

                        stats = tracer.results()

                        n_instructions = sum(stats.counts.values())

                        used_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

                        self._info_profiling['training_n_instructions'] += n_instructions
                        self._info_profiling['max_ram_used'] = max(used_memory, self._info_profiling['max_ram_used'])

                    else:
                        self._train_model()

                    execution_time = time.time() - start_time
                    self._training_execution_time += execution_time

                    # send trained weights to the server
                    self._send_local_model()
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

    def _load_compiled_model(self) -> keras.Model:
        """ It loads the local model to be trained"""
        model = self.get_skeleton_model()

        optimizer = self.get_optimizer()
        loss = self.get_loss_function()
        metric = self.get_metric()

        model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

        return model

    def _update_weights(self, weights) -> None:
        """ It updates the weights of the local model"""
        self.weights = weights
        self._model.set_weights(weights)

    @tf.function
    def _train_step(self, x, y):
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            predictions = self._model(x, training=True)

            # Compute the loss value for this minibatch.
            loss_value = self._loss_fn(y, predictions)

        gradients = tape.gradient(loss_value, self._model.trainable_variables)
        self._model.optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))
        return loss_value, gradients

    @tf.function
    def _test_step(self, x, y):
        predictions = self._model(x, training=False)
        loss_value = self._loss_fn(y, predictions)
        return loss_value

    def _train_model(self) -> None:
        """
        It trains the model.
        """
        for epoch in range(self._epochs):
            print(f"Epoch {epoch + 1}/{self._epochs}")

            if self._shuffle_dataset_each_epoch:
                indices = np.arange(self.x_train.shape[0])
                np.random.shuffle(indices)
                self.x_train = self.x_train[indices]
                self.y_train = self.y_train[indices]

            epoch_loss_avg = tf.metrics.Mean()
            epoch_accuracy = self.get_metric()

            # Accumulator for gradients
            accumulated_gradients = [tf.zeros_like(var) for var in self._model.trainable_variables]
            num_batches = 0

            # Iterate over the batches of the dataset.
            for step in range(0, len(self.x_train), self._batch_size):
                x_batch = self.x_train[step:step + self._batch_size]
                y_batch = self.y_train[step:step + self._batch_size]

                loss_value, gradients = self._train_step(x_batch, y_batch)

                # Accumulate gradients
                accumulated_gradients = [accum_grad + grad for accum_grad, grad in
                                         zip(accumulated_gradients, gradients)]
                num_batches += 1

                epoch_loss_avg.update_state(loss_value)
                epoch_accuracy.update_state(y_batch, self._model(x_batch, training=True))

            # Average gradients over the number of batches
            averaged_gradients = [grad / num_batches for grad in accumulated_gradients]
            self.gradients = np.array([grad.numpy() for grad in averaged_gradients], dtype='object')

            print(f"Epoch {epoch + 1}: Loss: {epoch_loss_avg.result()}, Accuracy: {epoch_accuracy.result()}")

        # set trained weights
        self._update_weights(np.array(self._model.get_weights(), dtype='object'))
        # evaluate model
        self._evaluate_model(self._model)

    def _evaluate_model(self, model=None) -> np.ndarray:
        """
        It evaluates the model with test dataset.
        :param model: model to evaluate, otherwise it loads it.
        :return: numpy array of (accuracy, loss)
        """
        is_evaluating_fm = False

        if model is None:
            model = self._model
            is_evaluating_fm = True

        test_loss_avg = tf.metrics.Mean()
        test_accuracy = self.get_metric()

        for step in range(0, len(self.x_test), self._batch_size):
            x_batch = self.x_test[step:step + self._batch_size]
            y_batch = self.y_test[step:step + self._batch_size]

            loss_value = self._test_step(x_batch, y_batch)
            test_loss_avg.update_state(loss_value)
            test_accuracy.update_state(y_batch, model(x_batch, training=False))

        print(f"Test Loss: {test_loss_avg.result()}, Test Accuracy: {test_accuracy.result()}")

        # Confusion Matrix
        y_pred = model.predict(self.x_test)

        cm = confusion_matrix(np.argmax(self.y_test, axis=1), np.argmax(y_pred, axis=1), normalize="true",
                              labels=list(range(self.get_num_classes())))

        cm_percentage = np.round(cm, 2)

        print("Confusion Matrix (Percentage):")
        print(cm_percentage)

        evaluation_data = np.array([[test_accuracy.result(), test_loss_avg.result()]])

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

    def _send_local_model(self) -> None:
        """Send trained weights to the server"""
        msg_body = {'client_id': self.id, 'weights': self.weights, 'gradients': self.gradients, 'n_training_samples': len(self.y_train)}
        self._send_message(MessageType.CLIENT_MODEL, msg_body)

    def _send_kpi_data(self) -> None:
        """Send evaluation data to the server"""
        msg_body = {'client_id': self.id,
                    'evaluation_federated': self.evaluation_data_federated_model,
                    'evaluation_training': self.evaluation_data_training_model,
                    'cm_federated': self.confusion_matrix_federated_model,
                    'cm_training': self.confusion_matrix_training_model,
                    'training_execution_time': self._training_execution_time}

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

    def shuffle_dataset_each_epoch(self, value: bool) -> None:
        """If True it shuffles the training dataset randomly before each epoch. Enabled by default."""
        self._shuffle_dataset_each_epoch = value

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

