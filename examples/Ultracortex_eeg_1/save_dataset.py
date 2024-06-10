import os
import numpy as np
from tensorflow import keras
from scipy.signal import butter, lfilter, filtfilt

def load_set(name: str):
    X, y = [], []  # X contiene i dati, y contiene le etichette
    main_dir = os.getcwd()
    dir = os.path.join(main_dir, "dataset", name)
    for label, action in enumerate(['left', 'right', 'none']):
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


def preprocessing_data(values):
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

    # Filtering
    fs = 125  # Frequenza di campionamento (Hz)
    lowcut = 0.5  # Frequenza di taglio inferiore (Hz)
    highcut = 50.0  # Frequenza di taglio superiore (Hz)

    # Applicazione del filtraggio passa-banda
    filtered_data = bandpass_filter(values, lowcut, highcut, fs)

    # Normalization
    x_normalized = n_z_score(filtered_data)

    return x_normalized

x_train, y_train = load_set("training")
x_test, y_test = load_set("validation")

np.save("dataset/x_train_raw.npy", x_train)
np.save("dataset/x_test_raw.npy", x_test)

np.save("dataset/y_train.npy", y_train)
np.save("dataset/y_test.npy", y_test)

# Preprocessing
x_train_preprocessed = preprocessing_data(x_train)
x_test_preprocessed = preprocessing_data(x_test)

np.save("dataset/x_train_preprocessed.npy", x_train_preprocessed)
np.save("dataset/x_test_preprocessed.npy", x_test_preprocessed)
