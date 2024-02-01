import tensorflow.keras

from ...TCPServer import TCPServer
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers


class Server(TCPServer):

    def get_skeleton_model(self) -> tensorflow.keras.Model:
        return Sequential([
            layers.InputLayer(input_shape=(784,)),
            layers.Dense(10),
            layers.Softmax(),
        ])


if __name__ == "__main__":
    server_address = ('localhost', 12345)

    # Server creation and initialization
    server = Server(server_address, 10, 10)

    try:
        # Binding and ready to listen
        server.bind_and_listen()
        # Create server threads
        server.create_server_threads()
    finally:
        server.join_server_threads()
        # Close server socket
        server.socket.close()
