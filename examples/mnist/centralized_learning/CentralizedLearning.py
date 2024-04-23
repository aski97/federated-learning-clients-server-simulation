import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from tensorflow import keras
import tensorflow_federated as tff
from src.CentralizedLearning import CentralizedLearning


class Centralized(CentralizedLearning):

    def load_dataset(self) -> tuple:
        clients = 10
        emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

        x_train = np.empty((0, 784))
        y_train = np.empty((0, 10))

        x_test = np.empty((0, 784))
        y_test = np.empty((0, 10))

        def get_x_y_set_reshaped(dataset):
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

        for i in range(clients):
            train_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[i])
            test_dataset = emnist_test.create_tf_dataset_for_client(emnist_test.client_ids[i])

            x_client_train, y_client_train = get_x_y_set_reshaped(train_dataset)
            x_client_test, y_client_test = get_x_y_set_reshaped(test_dataset)

            x_train = np.append(x_train, x_client_train, axis=0)
            y_train = np.append(y_train, y_client_train, axis=0)

            x_test = np.append(x_test, x_client_test, axis=0)
            y_test = np.append(y_test, y_client_test, axis=0)


        print(f"Campioni: {len(x_train)}, {len(y_train)}")
        return x_train, x_test, y_train, y_test

    def get_skeleton_model(self) -> keras.Model:
        initializer = "zero"

        return keras.models.Sequential([
            keras.layers.InputLayer(input_shape=self.x_train.shape[1:]),
            keras.layers.Dense(10, kernel_initializer=initializer),
            keras.layers.Softmax(),
        ])

    def get_optimizer(self) -> keras.optimizers.Optimizer | str:
        return keras.optimizers.SGD(learning_rate=0.02)

    def get_loss_function(self) -> keras.losses.Loss | str:
        return "categorical_crossentropy"

    def get_metric(self) -> keras.metrics.Metric | str:
        return "accuracy"

    def get_batch_size(self) -> int:
        return 20

    def get_train_epochs(self) -> int:
        return 10

    def shuffle_dataset_before_training(self) -> bool:
        return False

    def get_classes_name(self) -> list[str]:
        return ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']


if __name__ == "__main__":
    centralized_model = Centralized(enable_profiling=False)

    centralized_model.run()
