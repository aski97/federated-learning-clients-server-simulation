import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import sys
from tensorflow import keras
from keras.layers import Conv1D, Flatten, Dense, Dropout
from keras.src.regularizers import l1
from sklearn.model_selection import train_test_split
from src.CentralizedLearning import CentralizedLearning
from collections import Counter

class Centralized(CentralizedLearning):

    def load_dataset(self) -> tuple:
        folder_path = "../dataset/"
        # Check dataset has saved
        if not os.path.isdir(folder_path):
            sys.exit(
                f"Error: '{folder_path}' folder doesn't exist, execute SaveDataset.py first: python3 SaveDataset.py")

        dataset = []
        labels = []

        # Combine datasets
        for root, dirs, files in os.walk(folder_path):
            for folder_name in dirs:
                folder_full_path = os.path.join(root, folder_name)

                x = np.load(os.path.join(folder_full_path, "data.npy"))
                y = np.load(os.path.join(folder_full_path, "labels.npy"))

                dataset.append(x)
                labels.append(y)
                print(f"Combined dataset: {folder_name}")

        # Concatenate data
        dataset = np.concatenate(dataset, axis=0)
        labels = np.concatenate(labels, axis=0)
        print(f"Dataset loaded: {len(dataset)} items")

        # Add dummy dimension for the Conv net
        dataset = np.expand_dims(dataset, axis=-1)

        # Encode labels (0: 'left_hand', 1: 'right_hand')
        labels_encoded = np.where(labels == 'left_hand', 0, 1)

        # Split dataset in training and test set
        x_train, x_test, y_train, y_test = train_test_split(dataset, labels_encoded, test_size=0.25,
                                                            stratify=labels_encoded)

        # # Show frequency of digits
        # count = Counter(y_train)
        # for number, frequency in sorted(count.items()):
        #     print(f"Number {number} appears {frequency} times.")
        # print(f"Training: {len(y_train)} items | Test: {len(y_test)}")
        # labels one-hot encoding
        y_train = tf.keras.utils.to_categorical(y_train, 2)
        y_test = tf.keras.utils.to_categorical(y_test, 2)

        return x_train, x_test, y_train, y_test

    def get_skeleton_model(self) -> keras.Model:
        initializer = "glorot_uniform"

        return keras.models.Sequential([
            Conv1D(16, 5, padding='same', activation='relu', kernel_regularizer=l1(), kernel_initializer=initializer,
                   input_shape=self.x_train.shape[1:]),
            Conv1D(32, 3, padding='same', activation='relu', kernel_regularizer=l1(), kernel_initializer=initializer),
            Flatten(),
            Dense(64, activation='relu', kernel_regularizer=l1(), kernel_initializer=initializer),
            Dropout(0.5),
            Dense(2, activation='softmax', kernel_initializer=initializer)
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
        return 20

    def get_classes_name(self) -> list[str]:
        return ['left', 'right']


if __name__ == "__main__":
    centralized_model = Centralized()
    centralized_model.enable_op_determinism()
    centralized_model.enable_profiling(True)
    centralized_model.enable_evaluations_plots(True)

    centralized_model.run()
