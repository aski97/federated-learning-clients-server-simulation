import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix


class CentralizedLearning(ABC):
    def __init__(self):

        self._profiling = False
        self._evaluation_plots_enabled = True

        self.x_train, self.x_test, self.y_train, self.y_test = self.load_dataset()

        self._model = self._load_compiled_model()
        self._loss_fn = self.get_loss_function()
        self._batch_size = self.get_batch_size()
        self._epochs = self.get_train_epochs()
        self._shuffle_dataset_each_epoch = True
        self._train_accuracies = []
        self._train_losses = []
        self._test_accuracies = []
        self._test_losses = []

    def _load_compiled_model(self):
        """ It loads the model to be trained"""
        model = self.get_skeleton_model()

        optimizer = self.get_optimizer()
        loss = self.get_loss_function()
        metric = self.get_metric()

        model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

        model.summary()

        return model

    @tf.function
    def _train_step(self, x, y):
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            predictions = self._model(x, training=True)

            # Compute the loss value for this minibatch.
            loss_value = self._loss_fn(y, predictions)

        gradients = tape.gradient(loss_value, self._model.trainable_variables)
        self._model.optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))
        return loss_value, gradients

    def _train_model(self) -> None:
        """ It trains the model."""
        for epoch in range(self._epochs):
            print(f"Epoch {epoch + 1}/{self._epochs}")

            if self._shuffle_dataset_each_epoch:
                indices = np.arange(self.x_train.shape[0])
                np.random.shuffle(indices)
                self.x_train = self.x_train[indices]
                self.y_train = self.y_train[indices]

            epoch_loss_avg = tf.metrics.Mean()
            epoch_accuracy = self.get_metric()

            # Iterate over the batches of the dataset.
            for step in range(0, len(self.x_train), self._batch_size):
                x_batch = self.x_train[step:step + self._batch_size]
                y_batch = self.y_train[step:step + self._batch_size]

                loss_value, gradients = self._train_step(x_batch, y_batch)
                epoch_loss_avg.update_state(loss_value)
                epoch_accuracy.update_state(y_batch, self._model(x_batch, training=True))

            train_loss = epoch_loss_avg.result().numpy()
            train_accuracy = epoch_accuracy.result().numpy()
            self._train_losses.append(train_loss)
            self._train_accuracies.append(train_accuracy)

            # Evaluation on test set
            test_loss_avg = tf.metrics.Mean()
            test_accuracy = self.get_metric()

            for step in range(0, len(self.x_test), self._batch_size):
                x_batch = self.x_test[step:step + self._batch_size]
                y_batch = self.y_test[step:step + self._batch_size]

                loss_value = self._test_step(x_batch, y_batch)
                test_loss_avg.update_state(loss_value)
                test_accuracy.update_state(y_batch, self._model(x_batch, training=False))

            test_loss = test_loss_avg.result().numpy()
            test_accuracy = test_accuracy.result().numpy()
            self._test_losses.append(test_loss)
            self._test_accuracies.append(test_accuracy)

            print(
                f"Epoch {epoch + 1}: Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

        # evaluate model
        self._evaluate_model(self._model)

    @tf.function
    def _test_step(self, x, y):
        predictions = self._model(x, training=False)
        loss_value = self._loss_fn(y, predictions)
        return loss_value

    def _evaluate_model(self, model):
        test_loss_avg = tf.metrics.Mean()
        test_accuracy = self.get_metric()

        for step in range(0, len(self.x_test), self._batch_size):
            x_batch = self.x_test[step:step + self._batch_size]
            y_batch = self.y_test[step:step + self._batch_size]

            loss_value = self._test_step(x_batch, y_batch)
            test_loss_avg.update_state(loss_value)
            test_accuracy.update_state(y_batch, model(x_batch, training=False))

        print(f"Test Loss: {test_loss_avg.result()}, Test Accuracy: {test_accuracy.result()}")
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

        if self._evaluation_plots_enabled:
            self._plot_metrics()
            self._plot_confusion_matrix(cm_percentage, classes_name)

    @staticmethod
    def _plot_confusion_matrix(values, classes):
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

    def _plot_metrics(self):
        from matplotlib import pyplot as plt

        epochs_range = range(1, self._epochs + 1)

        # Plot accuracy
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, self._train_accuracies, label='Train Accuracy')
        plt.plot(epochs_range, self._test_accuracies, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.title('Train and Test Accuracy')

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, self._train_losses, label='Train Loss')
        plt.plot(epochs_range, self._test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.title('Train and Test Loss')

        plt.show()

    def run(self):
        import time
        start_time = time.time()
        if self._profiling:
            import psutil
            import resource
            import trace

            print("Tracing active")

            tracer = trace.Trace(
                count=True,
                trace=False,
                timing=True)

            tracer.runfunc(self._train_model)

            stats = tracer.results()

            n_instructions = sum(stats.counts.values())

            print("Number of called instructions:", n_instructions)
            # Get value of used memory
            used_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # KB to GB
            used_memory_gb = used_memory / (1024.0 * 1024.0)

            print("Used memory:", used_memory_gb, "GB")
            # Ottieni il processo corrente
            process = psutil.Process()

            # Ottieni l'uso della memoria del processo corrente
            memory_info = process.memory_info()

            # Se vuoi ottenere l'uso della memoria in Megabyte
            print(f"RSS (Resident Set Size): {memory_info.rss / (1024 ** 2)} MB")
        else:
            self._train_model()

        execution_time = time.time() - start_time
        print("Execution time:", execution_time, "secondi")

    def enable_profiling(self, value: bool) -> None:
        """
        It enables the profiling of the KPIs
        """
        self._profiling = value

    @staticmethod
    def enable_op_determinism() -> None:
        """ The training process uses deterministic operation in order to have the same experimental results"""
        seed = 1
        tf.keras.utils.set_random_seed(seed)  # sets seeds for base-python, numpy and tf
        tf.config.experimental.enable_op_determinism()

    def enable_evaluations_plots(self, value: bool) -> None:
        """
        If enabled it plots graphs of the evaluation data
        """
        self._evaluation_plots_enabled = value

    def shuffle_dataset_each_epoch(self, value: bool) -> None:
        """If True it shuffles the training dataset randomly before each epoch."""
        self._shuffle_dataset_each_epoch = value

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
    def get_optimizer(self) -> keras.optimizers.Optimizer:
        """
        Get the optimizer of the model
        :return: keras optimizer
        """
        pass

    @abstractmethod
    def get_loss_function(self) -> keras.losses.Loss:
        """
        Get the loss of the model
        :return: keras loss
        """
        pass

    @abstractmethod
    def get_metric(self) -> keras.metrics.Metric:
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
