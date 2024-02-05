import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Activation, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import itertools
import sys

tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()

folder_path = "dataset/"
# Check dataset has saved
if not os.path.isdir(folder_path):
    sys.exit(f"Error: '{folder_path}' folder doesn't exist, execute SaveDataset.py first: python3 SaveDataset.py")

dataset = []
labels = []

# Combine datasets
for root, dirs, files in os.walk(folder_path):
    for folder_name in dirs:
        folder_full_path = os.path.join(root, folder_name)

        x = np.load(os.path.join(folder_full_path, "data.npy"))
        y = np.load(os.path.join(folder_full_path, "labels.npy"))

        dataset.append(x)
        labels.append(y)
        print(f"Combined dataset: {folder_name}")

# Concatenate data
dataset = np.concatenate(dataset, axis=0)
labels = np.concatenate(labels, axis=0)
print(f"Dataset loaded: {len(dataset)} items")

# Add dummy dimension for the Conv net
dataset = np.expand_dims(dataset, axis=-1)

# Encode labels (0: 'left_hand', 1: 'right_hand')
labels_encoded = np.where(labels == 'left_hand', 0, 1)

# Split dataset in training and test set
x_train, x_test, y_train, y_test = train_test_split(dataset, labels_encoded, test_size=0.2, stratify=labels_encoded)

# labels one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 2)
y_test = tf.keras.utils.to_categorical(y_test, 2)

initializer = "glorot_uniform"
# Model definition
model = Sequential([
    Conv1D(64, 5, padding='same', activation='relu', kernel_initializer=initializer, input_shape=x_train.shape[1:]),
    Conv1D(128, 3, padding='same', activation='relu', kernel_initializer=initializer),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),
    Conv1D(256, 3, padding='same', activation='relu', kernel_initializer=initializer),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),
    Flatten(),
    Dense(512, activation='relu', kernel_initializer=initializer),
    Dropout(0.5),
    Dense(2, activation='softmax', kernel_initializer=initializer)
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training
epochs = 10
batch_size = 32
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# Confusion Matrix
y_pred = model.predict(x_test)
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

# Normalize Matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_percentage = np.round(cm_normalized, 2)

# Print Confusion Matrix
print("Confusion Matrix (Percentage):")
print(cm_percentage)


def plot_confusion_matrix(values, classes, title='Confusion matrix', cmap=plt.colormaps["Reds"]):
    plt.imshow(values, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = values.max() / 2.
    for i, j in itertools.product(range(values.shape[0]), range(values.shape[1])):
        color = "white" if values[i, j] > thresh else "black"
        plt.text(j, i, str(values[i, j]), horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


plot_confusion_matrix(cm_percentage, classes=['left', 'right'])
