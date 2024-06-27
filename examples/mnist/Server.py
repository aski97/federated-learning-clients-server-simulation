import sys
import os

from src.AggregationAlgorithm import FedAvg, FedAvgMomentum, FedAdam, FedSGD, FedProx, FedMiddleAvg

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(dir_path)
from src.TCPServer import TCPServer
import tensorflow.keras as keras


class Server(TCPServer):

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

    def get_classes_name(self) -> list[str]:
        return ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']


if __name__ == "__main__":
    server_address = ('localhost', 12345)

    # Server creation and execution
    server = Server(server_address, 10, 8)
    server.set_aggregation_algorithm(FedAvg())
    # server.set_aggregation_algorithm(FedAdam(beta1=0.5,learning_rate=0.01))
    # server.load_initial_weights("weights/prova.npy")
    server.enable_clients_profiling(False)
    server.enable_evaluations_plots(True)
    server.run()
