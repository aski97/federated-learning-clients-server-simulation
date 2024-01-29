import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import struct
import numpy as np
import socket
import argparse
import tensorflow as tf
import tensorflow_federated as tff
import CSUtils

np.random.seed(1)  # reproducibility of simulations


class TCPClient:
    EPOCHES = 10
    BATCH_SIZE = 20

    def __init__(self, server_address, client_id: int):
        self.server_address = server_address
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.id = client_id
        self.weights = None
        self.evaluation_data_federated_model = np.empty((0, 2), dtype=float)
        self.evaluation_data_training_model = np.empty((0, 2), dtype=float)
        # populate dataset
        self.x_train, self.x_test, self.y_train, self.y_test = self.load_client_dataset(client_id)

    def connect(self) -> None:
        """Connect to the server"""
        self.socket.connect(self.server_address)

    def run(self) -> None:
        """ It manages the message communication with the server"""
        while True:
            # Wait for Server message
            m_body, m_type = self.receive_message()

            # check if server closed the connection
            if m_body is None or m_type is None:
                break

            # behave differently with respect to the type of message received
            match m_type:
                case CSUtils.MessageType.FEDERATED_WEIGHTS:
                    print("Received federated weights")
                    # Update model
                    self.update_weights(m_body)
                    # evaluate model with new weights
                    self.evaluate_model()
                    # train model with new weights
                    self.train_model()
                    # send trained weights to the server
                    self.send_trained_weights()
                case CSUtils.MessageType.END_FL_TRAINING:
                    print("Received final federated weights. Federated training has finished.")
                    # Update model
                    self.update_weights(m_body)
                    # evaluate model with new weights
                    self.evaluate_model()
                    # send evaluation data to the server
                    self.send_evaluation_data()
                    # Client disconnect to the server
                    break
                case _:
                    continue

    def send_message(self, msg_type: CSUtils.MessageType, body: object) -> None:
        """
        Send a message to the server in {'type': '', 'body': ''} format
        :param msg_type: type of the message.
        :param body: body of the message.
        """
        msg_serialized = CSUtils.build_message(msg_type, body)
        self.socket.sendall(msg_serialized)

    def receive_message(self) -> tuple:
        """
        It waits until a message from a server is received.
        :return: unpacked message {msg_type, msg_body}.
        """
        # Read message length by first 4 bytes
        msg_len_bytes = self.socket.recv(4)
        if not msg_len_bytes:
            return None, None
        msg_len = struct.unpack('!I', msg_len_bytes)[0]
        # Read the message data
        data = b''
        while len(data) < msg_len:
            packet = self.socket.recv(msg_len - len(data))
            if not packet:  # EOF
                break
            data += packet

        return CSUtils.unpack_message(data)

    def close(self) -> None:
        """It closes connection with the server"""
        self.socket.close()

    @staticmethod
    def load_client_dataset(client_id) -> tuple:
        """
        It loads dataset given the id of the client.
        :param client_id: id of the client.
        :return: x_train, x_test, y_train, y_test
        """
        emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

        train_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[client_id])
        test_dataset = emnist_test.create_tf_dataset_for_client(emnist_test.client_ids[client_id])

        def get_x_y_set_reshaped(dataset):
            """ Reshape dataset in order to give it to the Dense layers"""
            x_set = np.empty((0, 28, 28))
            y_set = np.empty(0)

            for element in dataset.as_numpy_iterator():
                img = element['pixels']
                label = element['label']

                x_set = np.append(x_set, [img], axis=0)
                y_set = np.append(y_set, label)

            # reshape data from (value, 28, 28) to (value, 784)
            x_set_reshaped = x_set.reshape((x_set.shape[0], -1))

            return x_set_reshaped, y_set

        x_train, y_train = get_x_y_set_reshaped(train_dataset)
        x_test, y_test = get_x_y_set_reshaped(test_dataset)

        return x_train, x_test, y_train, y_test

    def load_model(self):
        """ It loads the local model to be trained"""
        model = CSUtils.get_skeleton_model()

        optimizer = tf.keras.optimizers.SGD(learning_rate=0.02)

        # Using sparse categorical crossentropy to not convert the labels in one-hot representation
        # ex: 5 ----> [0 0 0 0 0 1 0 0 0 0]
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

        if self.weights is not None:
            model.set_weights(self.weights)

        return model

    def update_weights(self, weights):
        """ It updates the weights of the local model"""
        self.weights = weights

    def train_model(self, shuffle=True) -> None:
        """
        It trains the model.
        :param shuffle: If True it shuffles the dataset randomly
        """
        model = self.load_model()

        if shuffle is True:
            # Shuffle dataset
            indices = np.arange(self.x_train.shape[0])
            np.random.shuffle(indices)

            self.x_train = self.x_train[indices]
            self.y_train = self.y_train[indices]

        model.fit(x=self.x_train, y=self.y_train,
                            batch_size=self.BATCH_SIZE, epochs=self.EPOCHES,
                            verbose=1)
        # set trained weights
        self.update_weights(np.array(model.get_weights(), dtype='object'))
        # evaluate model
        self.evaluate_model(model)

    def evaluate_model(self, model=None) -> np.ndarray:
        """
        It evaluates the model with test dataset.
        :param model: model to evaluate, otherwise it loads it.
        :return: numpy array of (accuracy, loss)
        """
        is_evaluating_fm = False

        if model is None:
            model = self.load_model()
            is_evaluating_fm = True

        test_loss, test_acc = model.evaluate(self.x_test, self.y_test)

        print(f'Test accuracy: {test_acc}')

        evaluation_data = np.array([[test_acc, test_loss]])

        if is_evaluating_fm is True:
            # It's using federated weights
            self.evaluation_data_federated_model = np.append(self.evaluation_data_federated_model, evaluation_data,
                                                             axis=0)
        else:
            # It's using train weights
            self.evaluation_data_training_model = np.append(self.evaluation_data_training_model, evaluation_data,
                                                            axis=0)

        return evaluation_data

    def send_trained_weights(self):
        """Send trained weights to the server"""
        msg_body = {'client_id': self.id, 'weights': self.weights}
        self.send_message(CSUtils.MessageType.CLIENT_TRAINED_WEIGHTS, msg_body)

    def send_evaluation_data(self):
        """Send evaluation data to the server"""
        msg_body = {'client_id': self.id, 'evaluation_federated': self.evaluation_data_federated_model,
                    'evaluation_training': self.evaluation_data_training_model}
        self.send_message(CSUtils.MessageType.CLIENT_EVALUATION, msg_body)


if __name__ == "__main__":
    # get arguments from the console
    parser = argparse.ArgumentParser()
    parser.add_argument('id', type=int, help='Client ID')
    args = parser.parse_args()

    server_address = ('localhost', 12345)

    # Create client
    client = TCPClient(server_address, args.id)

    try:
        # Connection to the Server
        client.connect()

        client.run()

    except socket.error as e:
        print(f"Socket error: {e}")
    finally:
        # Close socket
        client.close()
