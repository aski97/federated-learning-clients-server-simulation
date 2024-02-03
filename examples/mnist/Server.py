import sys
import os

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(dir_path)
from src.TCPServer import TCPServer
import tensorflow.keras as keras


class Server(TCPServer):

    def get_skeleton_model(self) -> keras.Model:
        return keras.models.Sequential([
            keras.layers.InputLayer(input_shape=(784,)),
            keras.layers.Dense(10),
            keras.layers.Softmax(),
        ])


if __name__ == "__main__":
    server_address = ('localhost', 12345)

    # Server creation and execution
    server = Server(server_address, 10, 10)
    server.run()
