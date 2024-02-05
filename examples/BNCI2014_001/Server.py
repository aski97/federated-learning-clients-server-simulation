import sys
import os

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(dir_path)
from src.TCPServer import TCPServer
import tensorflow.keras as keras


class Server(TCPServer):

    def get_skeleton_model(self) -> keras.Model:
        initializer = "glorot_uniform"

        return keras.models.Sequential([
            keras.layers.Conv1D(64, 5, padding='same', activation='relu', kernel_initializer=initializer, input_shape=(253, 1)),
            keras.layers.Conv1D(128, 3, padding='same', activation='relu', kernel_initializer=initializer),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Dropout(0.5),
            keras.layers.Conv1D(256, 3, padding='same', activation='relu', kernel_initializer=initializer),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Dropout(0.5),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu', kernel_initializer=initializer),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(2, activation='softmax', kernel_initializer=initializer)
        ])


if __name__ == "__main__":
    server_address = ('localhost', 12345)

    # Server creation and execution
    server = Server(server_address, 9, 3)
    server.run()