import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from tensorflow import keras
from tensorflow.keras import regularizers
from scipy.signal import butter, lfilter, filtfilt

from src.CentralizedLearning import CentralizedLearning


def n_z_score(values):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    normalized_data = (values - mean) / std
    return normalized_data


# Funzione per il filtraggio passa-banda
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y

def preprocessing_data(values):
    # Filtering
    fs = 125  # Frequenza di campionamento (Hz)
    lowcut = 0.5  # Frequenza di taglio inferiore (Hz)
    highcut = 50.0  # Frequenza di taglio superiore (Hz)

    # Applicazione del filtraggio passa-banda
    filtered_data = bandpass_filter(values, lowcut, highcut, fs)

    # Normalization
    x_normalized = n_z_score(filtered_data)

    return x_normalized

class Centralized(CentralizedLearning):

    def load_set(self, set: str):
        X, y = [], []  # X contiene i dati, y contiene le etichette
        main_dir = os.getcwd()
        dir = os.path.join(main_dir, "..", "dataset", set)
        for label, action in enumerate(self.get_classes_name()):
            action_dir = os.path.join(dir, action)
            c = 0
            for file in os.listdir(action_dir):
                if c == 200:
                    break
                file_path = os.path.join(action_dir, file)
                data = np.load(file_path, mmap_mode='r')
                X.append(data)
                y.append(label)
                c += 1
                # labels one-hot encoding
        x_np = np.array(X)
        y_np = np.array(y)
        y_one_hot = keras.utils.to_categorical(y_np, 3)
        # Ridimensionamento dei dati per adattarli alla CNN 2D.
        return x_np.reshape(x_np.shape[0], 250, 16 * 60), y_one_hot

    def load_dataset(self) -> tuple:

        # x_train = np.load("../dataset/x_train_raw.npy")
        # x_test = np.load("../dataset/x_test_raw.npy")
        x_train = np.load("../dataset/x_train_preprocessed.npy")
        x_test = np.load("../dataset/x_test_preprocessed.npy")
        y_train = np.load("../dataset/y_train.npy")
        y_test = np.load("../dataset/y_test.npy")
        print(f"Campioni: {len(x_train)}, {len(x_test)}")
        return x_train, x_test, y_train, y_test

    def get_skeleton_model(self) -> keras.Model:

        model = keras.models.Sequential([
            # Layer convoluzionali 2D.
            keras.layers.Conv1D(64, 3, activation='relu', input_shape=(250, 960), kernel_regularizer=regularizers.l2(0.001)),
            keras.layers.Conv1D(64, 3, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            keras.layers.MaxPooling1D(2),
            keras.layers.Conv1D(32, 3, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            keras.layers.MaxPooling1D(2),
            keras.layers.Flatten(),
            keras.layers.Dense(1920, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(3, activation='softmax')
        ])

        return model

    def get_optimizer(self) -> keras.optimizers.Optimizer | str:
        return keras.optimizers.Adam()

    def get_loss_function(self) -> keras.losses.Loss | str:
        return "categorical_crossentropy"

    def get_metric(self) -> keras.metrics.Metric | str:
        return "accuracy"

    def get_batch_size(self) -> int:
        return 30

    def get_train_epochs(self) -> int:
        return 10

    def shuffle_dataset_before_training(self) -> bool:
        return False

    def get_classes_name(self) -> list[str]:
        return ['left', 'right', 'none']


if __name__ == "__main__":
    centralized_model = Centralized(enable_profiling=False)

    centralized_model.run()
