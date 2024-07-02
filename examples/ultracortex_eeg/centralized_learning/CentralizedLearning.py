import os
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import Covariances
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from tensorflow import keras
from keras.src.regularizers import l2
from sklearn.preprocessing import StandardScaler
from src.CentralizedLearning import CentralizedLearning


class Centralized(CentralizedLearning):

    def load_dataset(self) -> tuple:
        x_train_all = []
        y_train_all = []
        x_test_all = []
        y_test_all = []

        for i in range(6):

            path_dataset = f"../datasets/dataset_2_0/{i}"

            x_train = np.load(f"{path_dataset}/x_train.npy")
            x_test = np.load(f"{path_dataset}/x_test.npy")
            y_train = np.load(f"{path_dataset}/y_train.npy")
            y_test = np.load(f"{path_dataset}/y_test.npy")

            x_train_all.append(x_train)
            y_train_all.append(y_train)
            x_test_all.append(x_test)
            y_test_all.append(y_test)

        x_train_all = np.concatenate(x_train_all, axis=0)
        y_train_all = np.concatenate(y_train_all, axis=0)
        x_test_all = np.concatenate(x_test_all, axis=0)
        y_test_all = np.concatenate(y_test_all, axis=0)

        print(f"Campioni: {len(y_train_all)}, {len(y_test_all)}")

        # Inizializzare lo scaler
        scaler = StandardScaler()
        # Normalizzare i dati di training
        x_train_all = scaler.fit_transform(x_train_all.reshape(-1, x_train_all.shape[-1])).reshape(x_train_all.shape)
        # Normalizzare i dati di test
        x_test_all = scaler.transform(x_test_all.reshape(-1, x_test_all.shape[-1])).reshape(x_test_all.shape)

        # Applicazione di Covariances e Tangent Space
        cov_estimator = Covariances(estimator='lwf')
        x_train_cov = cov_estimator.fit_transform(x_train_all)
        x_test_cov = cov_estimator.transform(x_test_all)

        ts = TangentSpace()
        x_train_ts = ts.fit_transform(x_train_cov)
        x_test_ts = ts.transform(x_test_cov)

        return x_train_ts, x_test_ts, y_train_all, y_test_all

    def get_skeleton_model(self) -> keras.Model:
        input_shape = (self.x_train.shape[1],)  # Modificato per accettare input bidimensionali

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

    def get_optimizer(self) -> keras.optimizers.Optimizer:
        initial_learning_rate = 0.003
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=10000,
            decay_rate=0.9,
            staircase=True
        )
        return keras.optimizers.Adam(learning_rate=lr_schedule)

    def get_loss_function(self) -> keras.losses.Loss:
        return keras.losses.CategoricalCrossentropy()

    def get_metric(self) -> keras.metrics.Metric | str:
        return keras.metrics.CategoricalAccuracy()

    def get_batch_size(self) -> int:
        return 64

    def get_train_epochs(self) -> int:
        return 124

    def get_classes_name(self) -> list[str]:
        return ['left', 'right']


if __name__ == "__main__":
    centralized_model = Centralized()
    centralized_model.enable_op_determinism()
    centralized_model.enable_profiling(False)
    centralized_model.enable_evaluations_plots(True)

    centralized_model.run()
