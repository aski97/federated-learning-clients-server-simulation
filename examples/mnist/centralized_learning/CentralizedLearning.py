import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from tensorflow import keras
import tensorflow_federated as tff
from src.CentralizedLearning import CentralizedLearning
from collections import Counter

class Centralized(CentralizedLearning):

    def load_dataset(self) -> tuple:
        clients = 10
        emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

        x_train = np.empty((0, 28, 28))
        y_train = np.empty((0, 10))

        x_test = np.empty((0, 28, 28))
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
            # x_set_reshaped = x_set.reshape((x_set.shape[0], -1))
            # labels one-hot encoding

            # # Show frequency of digits
            # count = Counter(y_set)
            # for number, frequency in sorted(count.items()):
            #     print(f"Number {number} appears {frequency} times.")

            y_one_hot = keras.utils.to_categorical(y_set, 10)

            return x_set, y_one_hot

        for i in range(clients):
            print(f"Loading data Client {i}")
            train_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[i])
            test_dataset = emnist_test.create_tf_dataset_for_client(emnist_test.client_ids[i])
            # print("--Training data")
            x_client_train, y_client_train = get_x_y_set_reshaped(train_dataset)
            # print("--Test data")
            x_client_test, y_client_test = get_x_y_set_reshaped(test_dataset)

            # print(f"Samples: {len(y_client_train)}, {len(y_client_test)}")

            x_train = np.append(x_train, x_client_train, axis=0)
            y_train = np.append(y_train, y_client_train, axis=0)

            x_test = np.append(x_test, x_client_test, axis=0)
            y_test = np.append(y_test, y_client_test, axis=0)

        print(f"Total Samples: {len(y_train)}, {len(y_test)}")

        return x_train, x_test, y_train, y_test

    def get_skeleton_model(self) -> keras.Model:
        return keras.models.Sequential([
            keras.layers.Conv1D(filters=32, kernel_size=5, padding='same', activation='relu', input_shape=(28, 28)),
            keras.layers.AvgPool1D(strides=2),
            keras.layers.Conv1D(filters=48, kernel_size=5, padding='valid', activation='relu'),
            keras.layers.AvgPool1D(strides=2),
            keras.layers.Flatten(),
            keras.layers.Dense(160, activation='relu'),
            keras.layers.Dense(84, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])


    def get_optimizer(self) -> keras.optimizers.Optimizer:
        return keras.optimizers.Adam(learning_rate=0.002)

    def get_loss_function(self) -> keras.losses.Loss:
        return keras.losses.CategoricalCrossentropy()

    def get_metric(self) -> keras.metrics.Metric:
        return keras.metrics.CategoricalAccuracy()

    def get_batch_size(self) -> int:
        return 32

    def get_train_epochs(self) -> int:
        return 40

    def get_classes_name(self) -> list[str]:
        return ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']


if __name__ == "__main__":
    centralized_model = Centralized()
    centralized_model.enable_op_determinism()
    centralized_model.enable_profiling(True)
    centralized_model.enable_evaluations_plots(True)
    centralized_model.run()
