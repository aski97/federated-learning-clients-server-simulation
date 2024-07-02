import sys
import os

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(dir_path)
from src.TCPClient import TCPClient
from tensorflow import keras
from keras.src.regularizers import l2
import argparse
import numpy as np
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import Covariances
from sklearn.preprocessing import StandardScaler

class Client(TCPClient):

    def get_batch_size(self) -> int:
        return 32

    def get_train_epochs(self) -> int:
        return 50

    def get_loss_function(self):
        return keras.losses.CategoricalCrossentropy()

    def get_metric(self):
        return keras.metrics.CategoricalAccuracy()

    def get_num_classes(self) -> int:
        return 2

    def get_skeleton_model(self) -> keras.Model:
        input_shape = (self.x_train.shape[1],)
        print(input_shape)
        model = keras.models.Sequential([
            keras.layers.Dense(512, activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.01)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            keras.layers.BatchNormalization(), keras.layers.Dropout(0.5),
            keras.layers.Dense(2, activation='softmax')
        ])

        return model

    def load_dataset(self) -> tuple:
        folder_path = f"datasets/dataset_2_0/{self.id}"

        x_train = np.load(f"{folder_path}/x_train.npy")
        x_test = np.load(f"{folder_path}/x_test.npy")
        y_train = np.load(f"{folder_path}/y_train.npy")
        y_test = np.load(f"{folder_path}/y_test.npy")

        # Inizializzare lo scaler
        scaler = StandardScaler()
        # Normalizzare i dati di training
        x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
        # Normalizzare i dati di test
        x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)

        # Applicazione di Covariances e Tangent Space
        cov_estimator = Covariances(estimator='lwf')
        x_train_cov = cov_estimator.fit_transform(x_train)
        x_test_cov = cov_estimator.transform(x_test)

        ts = TangentSpace()
        x_train = ts.fit_transform(x_train_cov)
        x_test = ts.transform(x_test_cov)

        return x_train, x_test, y_train, y_test

    def get_optimizer(self):
        initial_learning_rate = 0.003
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=10000,
            decay_rate=0.9,
            staircase=True
        )
        return keras.optimizers.Adam(learning_rate=lr_schedule)


if __name__ == "__main__":
    # get arguments from the console
    parser = argparse.ArgumentParser()
    parser.add_argument('id', type=int, help='Client ID')
    args = parser.parse_args()

    server_address = ('localhost', 12345)

    # Create client
    client = Client(server_address, args.id)
    client.enable_op_determinism()
    client.run()
