import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix



class CentralizedLearning(ABC):
    def __init__(self, enable_op_determinism: bool = True, enable_profiling: bool = False):
        if enable_op_determinism:
            tf.keras.utils.set_random_seed(1)  # sets seeds for base-python, numpy and tf
            tf.config.experimental.enable_op_determinism()

        self.profiling = enable_profiling

        self.x_train, self.x_test, self.y_train, self.y_test = self.load_dataset()

        # # Creazione di dataset TensorFlow dai dati di training e test
        # self.train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        # self.test_dataset = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
        #
        # # Applicazione del caching dei dati
        # self.train_dataset = self.train_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        # self.test_dataset = self.test_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    @abstractmethod
    def shuffle_dataset_before_training(self) -> bool:
        """If True it shuffles the training dataset randomly before training the model."""
        pass

    @abstractmethod
    def load_dataset(self) -> tuple:
        """
        It loads the dataset
        :return: x_train, x_test, y_train, y_test
        """
        pass

    @abstractmethod
    def get_skeleton_model(self) -> keras.Model:
        """
        Get the skeleton of the model
        :return: keras model
        """
        pass

    @abstractmethod
    def get_optimizer(self) -> keras.optimizers.Optimizer | str:
        """
        Get the optimizer of the model
        :return: keras optimizer
        """
        pass

    @abstractmethod
    def get_loss_function(self) -> keras.losses.Loss | str:
        """
        Get the loss of the model
        :return: keras loss
        """
        pass

    @abstractmethod
    def get_metric(self) -> keras.metrics.Metric | str:
        """
        Get the metric for the evaluation
        :return: keras metric
        """
        pass

    @abstractmethod
    def get_batch_size(self) -> int:
        pass

    @abstractmethod
    def get_train_epochs(self) -> int:
        pass

    @abstractmethod
    def get_classes_name(self) -> list[str]:
        """ Get the list of names of the classes to predict"""
        pass

    def load_model(self):
        """ It loads the model to be trained"""
        model = self.get_skeleton_model()

        optimizer = self.get_optimizer()
        loss = self.get_loss_function()
        metric = self.get_metric()

        model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

        model.summary()

        return model

    def train_model(self) -> None:
        """ It trains the model."""
        model = self.load_model()

        if self.shuffle_dataset_before_training():
            # Shuffle training dataset
            indices = np.arange(self.x_train.shape[0])
            np.random.shuffle(indices)

            self.x_train = self.x_train[indices]
            self.y_train = self.y_train[indices]

        batch_size = self.get_batch_size()
        epochs = self.get_train_epochs()

        model.fit(x=self.x_train, y=self.y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1)

        # evaluate model
        self.evaluate_model(model)

    def evaluate_model(self, model):
        """
        It evaluates the model with test dataset.
        :param model: model to evaluate, otherwise it loads it.
        :return: numpy array of (accuracy, loss)
        """

        test_loss, test_acc = model.evaluate(self.x_test, self.y_test)

        print(f'Test accuracy: {test_acc}')
        print(f"Test loss: {test_loss}")
        # Confusion Matrix
        y_pred = model.predict(self.x_test)

        classes_name = self.get_classes_name()
        # NB: Y_TEST MUST BE ONE-HOT encoded
        cm = confusion_matrix(np.argmax(self.y_test, axis=1), np.argmax(y_pred, axis=1), normalize="true",
                              labels=list(range(len(classes_name))))

        cm_percentage = np.round(cm, 2)

        # Print Confusion Matrix
        print("Confusion Matrix (Percentage):")
        print(cm_percentage)

        if not self.profiling:
            self.plot_confusion_matrix(cm_percentage, classes_name)

    @staticmethod
    def plot_confusion_matrix(values, classes):
        import matplotlib.pyplot as plt
        import itertools

        plt.imshow(values, interpolation='nearest', cmap=plt.colormaps["Reds"])
        plt.title("Confusion matrix")
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

    def run(self):
        if self.profiling:
            import resource
            import trace
            import time

            print("Tracing active")

            start_time = time.time()

            tracer = trace.Trace(
                count=True,
                trace=False,
                timing=True)

            tracer.runfunc(self.train_model)

            stats = tracer.results()

            execution_time = time.time() - start_time

            n_instructions = sum(stats.counts.values())

            print("Number of called instructions:", n_instructions)
            print("Execution time:", execution_time, "secondi")

            # Get value of used memory
            used_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

            # KB to GB
            used_memory_gb = used_memory / (1024.0 * 1024.0)

            print("Used memory:", used_memory_gb, "GB")
        else:
            self.train_model()
