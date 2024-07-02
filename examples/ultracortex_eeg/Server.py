import sys
import os

from src.AggregationAlgorithm import FedAvg, FedMiddleAvg, FedAvgMomentum, FedAdam

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(dir_path)
from src.TCPServer import TCPServer
import tensorflow.keras as keras
from keras.src.regularizers import l2


class Server(TCPServer):

    def get_skeleton_model(self) -> keras.Model:

        model = keras.models.Sequential([
            keras.layers.Dense(512, activation='relu', input_shape=(136,), kernel_regularizer=l2(0.01)),
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

    def get_classes_name(self) -> list[str]:
        return ['left', 'right']


if __name__ == "__main__":
    server_address = ('localhost', 12345)

    # Server creation and execution
    server = Server(server_address, 6, 8)
    # server.load_initial_weights("weights/fedAdam_32.npy")
    server.set_aggregation_algorithm(FedAvg())
    # server.set_aggregation_algorithm(FedMiddleAvg())
    # server.set_aggregation_algorithm(FedAvgMomentum(beta=0.005, learning_rate=0.01))
    # server.set_aggregation_algorithm(FedAdam(beta1=0.5, learning_rate=0.001))
    server.enable_op_determinism()
    server.enable_clients_profiling(False)
    server.enable_evaluations_plots(True)
    server.run()
