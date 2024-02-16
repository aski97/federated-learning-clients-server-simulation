import sys
import os

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(dir_path)
from src.TCPClient import TCPClient
from tensorflow import keras
import argparse
import numpy as np
from sklearn.model_selection import train_test_split


class Client(TCPClient):

    def shuffle_dataset_before_training(self) -> bool:
        return False

    def get_batch_size(self) -> int:
        return 32

    def get_train_epochs(self) -> int:
        return 5

    def get_loss_function(self):
        return "categorical_crossentropy"

    def get_metric(self):
        return "accuracy"

    def get_num_classes(self) -> int:
        return 2

    def get_skeleton_model(self) -> keras.Model:
        return keras.models.Sequential([
            keras.layers.Conv1D(32, 5, padding='same', activation='relu',
                                input_shape=self.x_train.shape[1:]),
            keras.layers.Conv1D(64, 3, padding='same', activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(2, activation='softmax')
        ])

    def load_dataset(self) -> tuple:
        folder_path = f"dataset/{self.id}"
        # Check dataset has saved
        if not os.path.isdir(folder_path):
            sys.exit(
                f"Error: '{folder_path}' folder doesn't exist, execute SaveDataset.py first: python3 SaveDataset.py")

        x = np.load(os.path.join(folder_path, "data.npy"))
        y = np.load(os.path.join(folder_path, "labels.npy"))

        print(f"Dataset loaded: {len(x)} items")

        # Add dummy dimension for the Conv net
        x = np.expand_dims(x, axis=-1)

        # Encode labels (0: 'left_hand', 1: 'right_hand')
        y_encoded = np.where(y == 'left_hand', 0, 1)

        # Split dataset in training and test set.
        # - Stratify: Ensures that the split maintains the same class
        # distribution between the training set and the test set. This can be particularly important when dealing
        # with classes of different sizes or when you want to preserve the representativeness of the classes during
        # the data split.
        x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, stratify=y_encoded)

        # labels one-hot encoding
        y_train = keras.utils.to_categorical(y_train, 2)
        y_test = keras.utils.to_categorical(y_test, 2)

        return x_train, x_test, y_train, y_test

    def get_optimizer(self):
        return keras.optimizers.Adam()


if __name__ == "__main__":
    # get arguments from the console
    parser = argparse.ArgumentParser()
    parser.add_argument('id', type=int, help='Client ID')
    args = parser.parse_args()

    server_address = ('localhost', 12345)

    # Create client
    client = Client(server_address, args.id)

    client.run()
