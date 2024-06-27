import sys
import os

from src.AggregationAlgorithm import FedAvg, FedMiddleAvg, FedAvgMomentum, FedAdam

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(dir_path)
from src.TCPServer import TCPServer
import tensorflow.keras as keras
from keras.layers import Conv1D, Flatten, Dense, Dropout
from keras.src.regularizers import l1


class Server(TCPServer):

    def get_skeleton_model(self) -> keras.Model:
        initializer = "glorot_uniform"

        return keras.models.Sequential([
            Conv1D(16, 5, padding='same', activation='relu', kernel_regularizer=l1(), kernel_initializer=initializer,
                   input_shape=(253, 1)),
            Conv1D(32, 3, padding='same', activation='relu', kernel_regularizer=l1(), kernel_initializer=initializer),
            Flatten(),
            Dense(64, activation='relu', kernel_regularizer=l1(), kernel_initializer=initializer),
            Dropout(0.5),
            Dense(2, activation='softmax', kernel_initializer=initializer)
        ])

    def get_classes_name(self) -> list[str]:
        return ['left', 'right']


if __name__ == "__main__":
    server_address = ('localhost', 12345)

    # Server creation and execution
    server = Server(server_address, 9, 32)
    server.set_aggregation_algorithm(FedAvg())
    # server.set_aggregation_algorithm(FedMiddleAvg())
    # server.set_aggregation_algorithm(FedAvgMomentum(beta=0.1))
    # server.set_aggregation_algorithm(FedAdam(beta1=0.5, learning_rate=0.01))
    server.enable_clients_profiling(False)
    server.enable_evaluations_plots(True)
    server.run()
