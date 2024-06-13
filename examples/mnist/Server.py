import sys
import os

from src.AggregationAlgorithm import FedAvg, FedAvgServerMomentum

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(dir_path)
from src.TCPServer import TCPServer
import tensorflow.keras as keras


class Server(TCPServer):

    def get_skeleton_model(self) -> keras.Model:
        return keras.models.Sequential([
            keras.layers.InputLayer(input_shape=(784,)),
            keras.layers.Dense(10, kernel_initializer="zero"),
            keras.layers.Softmax(),
        ])

    def get_classes_name(self) -> list[str]:
        return ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']


if __name__ == "__main__":
    server_address = ('localhost', 12345)

    # Server creation and execution
    server = Server(server_address, 10, 10)
    server.set_aggregation_algorithm(FedAvgServerMomentum(0.9, 1))
    server.enable_clients_profiling(True)
    # server.enable_evaluations_plots(False)
    server.run()
