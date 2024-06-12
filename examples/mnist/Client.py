import sys
import os

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(dir_path)
from src.TCPClient import TCPClient
import tensorflow.keras as keras
import argparse
import numpy as np
import tensorflow_federated as tff


class Client(TCPClient):

    def get_num_classes(self) -> int:
        return 10

    def get_batch_size(self) -> int:
        return 20

    def get_train_epochs(self) -> int:
        return 10

    def get_loss_function(self):
        return "categorical_crossentropy"

    def get_metric(self):
        return "accuracy"

    def get_skeleton_model(self) -> keras.Model:
        return keras.models.Sequential([
            keras.layers.InputLayer(input_shape=(784,)),
            keras.layers.Dense(10),
            keras.layers.Softmax(),
        ])

    def load_dataset(self) -> tuple:
        emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

        train_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[self.id])
        test_dataset = emnist_test.create_tf_dataset_for_client(emnist_test.client_ids[self.id])

        def get_x_y_set_reshaped(dataset):
            """
            Reshape dataset in order to give it to the Dense layers
            """
            x_set = np.empty((0, 28, 28))
            y_set = np.empty(0)

            for element in dataset.as_numpy_iterator():
                img = element['pixels']
                label = element['label']

                x_set = np.append(x_set, [img], axis=0)
                y_set = np.append(y_set, label)

            # reshape data from (value, 28, 28) to (value, 784)
            x_set_reshaped = x_set.reshape((x_set.shape[0], -1))
            # labels one-hot encoding
            y_one_hot = keras.utils.to_categorical(y_set, 10)

            return x_set_reshaped, y_one_hot

        x_train, y_train = get_x_y_set_reshaped(train_dataset)
        x_test, y_test = get_x_y_set_reshaped(test_dataset)

        return x_train, x_test, y_train, y_test

    def get_optimizer(self):
        return keras.optimizers.SGD(learning_rate=0.02)


if __name__ == "__main__":
    # get arguments from the console
    parser = argparse.ArgumentParser()
    parser.add_argument('id', type=int, help='Client ID')
    args = parser.parse_args()

    server_address = ('localhost', 12345)

    # Create client
    client = Client(server_address, args.id)
    client.enable_op_determinism()
    client.shuffle_dataset_before_training(False)
    client.run()
