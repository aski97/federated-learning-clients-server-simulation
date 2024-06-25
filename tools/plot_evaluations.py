import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def plot_metrics_from_directory(directory):
    # List all files in the given directory
    files = [f for f in os.listdir(directory) if f.endswith('.npy')]

    # Dictionary to hold grouped file names
    grouped_files = defaultdict(list)

    # Group files by metric type and rounds value
    for file in files:
        parts = file.split('_')
        if len(parts) >= 3:  # Ensure the file name has at least 4 parts
            metric_type = parts[0]
            x_value = parts[1]  # The aggregation algorithm
            y_value = parts[2]  # Number of rounds
            number_rounds = y_value.replace('rounds.npy', '')

            key = f'{metric_type}_{number_rounds}'

            grouped_files[key].append((file, x_value))

    # Plot each group
    for key, file_group in grouped_files.items():
        metric_type, number_rounds = key.split('_')

        fig, ax = plt.subplots()
        ax.set_title(f'Average {metric_type.capitalize()} - {number_rounds} rounds')
        ax.set_xlabel('Rounds')
        ax.set_ylabel(f'{metric_type.capitalize()}')

        # Load and plot each file in the group
        for file, x_value in file_group:
            file_path = os.path.join(directory, file)
            data = np.load(file_path)
            ax.plot(data, label=x_value)

        # Add a legend
        ax.legend()

        # Show the plot
        plt.show()

# Example usage
directory = '../examples/mnist/evaluations/'  # Replace with your directory
plot_metrics_from_directory(directory)