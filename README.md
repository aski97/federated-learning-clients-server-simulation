# Federated Learning: clients/server desktop implementation
This is a design of a simple client/server architecture to simulate federated learning involving real nodes, where each node (client) owns its own data. The client and server are written in Python and communicate via sockets using the TCP transport protocol.

## Table of contents

* [Overview](#overview)
* [Libraries](#libraries)
* [Architecture](#architecture)
    + [Server](#server)
        - [Methods](#methods)
            * [Abstract Methods](#abstract-methods)
            * [Public Methods](#public-methods)
    + [Client](#client)
        - [Methods](#methods-1)
            * [Abstract Methods](#abstract-methods-1)
            * [Public Methods](#public-methods-1)
    + [Supported Aggregation Algorithms](#supported-aggregation-algorithms)
    + [Message Exchange](#message-exchange)
    + [Profiling](#profiling)
    + [Limits of the Implementation](#limits-of-the-implementation)
* [Requirements](#requirements)
* [Simulation Mnist Dataset](#simulation-mnist-dataset)
* [Simulation BNCI2014_001 Dataset](#simulation-bnci2014_001-dataset)

## Overview

This document provides a detailed description of the implementation of our Federated Learning (FL) framework. The framework is designed to facilitate the simulation and performance analysis of various federated algorithms. It consists of a central server and multiple clients, each training a model locally on their data and contributing to the global aggregation without sharing raw data.

The implementation is done in Python (version 3.10), leveraging its simplicity and extensive libraries for machine learning. Clients and server communicate via sockets using the TCP/IP protocol.

## Libraries

The following libraries are used to ensure the efficiency and reusability of the code:

- **TensorFlow**: For building and training machine learning models.
- **NumPy**: For mathematical operations and data manipulation.
- **Pickle**: For serializing and deserializing Python objects.
- **Struct**: For converting data formats to and from bytes.
- **Threading**: For creating and managing multiple threads.
- **Socket**: For network communication between clients and the server.
- **Os**: For interacting with the operating system.
- **Trace**: For tracing the number of instructions executed during model training.
- **Resource**: For monitoring system resource usage.
- **Matplotlib**: For creating graphs and visualizations.

## Architecture
![fl_architecture](/images/fl_arc.png)

Here is a representation of the architecture, as shown in the figure. The server awaits the connection of nodes participating in federated learning. Whenever a client connects, the server sends an initialized federated model. Once a certain number of nodes are connected, the federated learning process begins:

1. Clients train their local models on local data using federated weights received from the server.
2. Upon completion of local training, clients send the new weights to the server.
3. The server aggregates the weights from all clients, generating a new model.
4. The server sends the new federated model to the clients.
This sequence is iterated for a predetermined number of rounds, at the end of which, clients send the server accuracy and loss data of the federated model before closing the connection with the server.

###  Server
The server component is defined by the abstract class [TCPServer](/src/TCPServer.py). The constructor requires three main parameters:
- `server_address`: The server address.
- `number_clients`: The number of participating clients.
- `number_rounds`: The number of federated learning rounds.
- `save_weights_path` (optional): Path to save the federated learning weights. If not specified, the model is not saved.

Once the TCPServer class is implemented and instantiated with its parameters, the ```run()``` function should be executed to run the server.

The server opens a socket at the specified address, in our case, localhost:12345, and three threads are created:

+ A thread, ```thread_client_connections```, listens to accept client connections. It accepts up to a number of connections equal to ```number_clients```. For each connected client, a ```client_thread``` is associated with it to handle communication.
+ A thread, ```thread_fl_algorithms```, manages rounds for federated learning. Once all clients send their weights, it calculates the average of all weights and sends the new model to the clients.
+ A thread, ```thread_final_evaluations```, performs the final evaluation of the learning. When the learning process concludes, it collects the accuracies and losses of each local model and creates graphs.

At this point, the server waits for client connections, and when a connection is initialized, the server sends configuration values an initialized model with weights and biases dependent on ```kernel_initializer``` and ```bias_initializer```, respectively defined in the layers of the Keras model returned by the ```get_skeleton_model()``` function. The currently supported configuration is ```profiling```, which allows enabling or disabling profiling for the node in question.

When all clients are connected, the server waits for the reception of local models, thus starting the learning phase that lasts for a number of rounds defined by the ```number_rounds``` variable. At each round, when the server collects all models, the following sequence of events occurs:

1. Models (weights and biases) are aggregated by taking the average of the models.
2. The server sends the resulting model to all clients.

The learning process concludes when the number of rounds is exhausted, resulting in a final model. The thread responsible for evaluation proceeds to provide graphs that describe the learning progress, including:

+ **Average accuracy of the final model**.
+ **Average loss of the final model**.
+ **Trend of accuracy per round for predicting test samples for each client**.
+ **Trend of loss per round for predicting test samples for each client**.
+ **Trend of average accuracy per round**.
+ **Trend of average loss per round**.
+ **Confusion matrix of the final model (mean of clients confusion matrix of the final model)**.
+ **Number of instructions executed per client during the training phases**. 
+ **Total execution time per client for the training phases**.
+ **Maximum RAM usage per client**.

#### Methods

##### Abstract Methods
- `get_skeleton_model(self) -> Model`: Returns the Keras model skeleton.
- `get_classes_names(self) -> list[str]`: Returns a list of class names used for generating the confusion matrix.

##### Public Methods
- `run(self) -> None`: Starts the server and manages the entire federated learning cycle.
- `enable_clients_profiling(self, value: bool) -> None`: Enables profiling to receive key performance indicators (KPIs) from the nodes.
- `enable_evaluations_plots(self, value: bool) -> None`: Displays evaluation and KPI plots if enabled.
- `set_aggregation_algorithm(self, aggregation_algorithm) -> None`: Sets the weight aggregation algorithm.
- `save_federated_weights(self, file_path) -> None`: Saves the federated model to the specified file path.
- `load_initial_weights(self, file_path) -> None`: Initializes the federated model with weights from the specified path.


### Client
The client component is defined by the abstract class [TCPClient](/src/TCPClient.py). The constructor requires two main parameters:
- `server_address`: The server address to connect to.
- `client_id`: The ID of the client.

Once the TCPClient class is implemented and instantiated with its parameters, the ```run()``` function should be executed to run the client.

The client opens a socket and connects to the server's address. Then, it waits to receive the federated model from the server. Once it receives the weights and biases, it loads them into the local model (net) and starts an initial evaluation. In this phase, the evaluation helps understanding the accuracy of the federated model using the local test dataset. After evaluating the federated model, the client starts the training on the training data.

The dataset is divided into batches, where each batch has a number of samples equal to the value returned by the ```get_batch_size()``` method. Training proceeds for a number of epochs equal to the value returned by the ```get_train_epochs()``` method. Upon completion, the model is re-evaluated on the test data, and the results are stored. Afterward, the client sends the weights and biases of the just-trained model to the server. This operation repeats until the server sends the final model, on which the client performs a single evaluation, sending all previous evaluations back to the server.

Once the federated training is complete, the client closes the connection with the server.

#### Methods

##### Abstract Methods
- `load_dataset(self) -> tuple`: Loads the client's dataset, returning training and test sets.
- `get_skeleton_model(self) -> keras.Model`: Returns the model to be trained.
- `get_optimizer(self) -> keras.optimizers.Optimizer | str`: Returns the optimizer for compiling the model.
- `get_loss_function(self) -> keras.losses.Loss | str`: Returns the loss function for compiling the model.
- `get_metric(self) -> keras.metrics.Metric | str`: Returns the metric for evaluating the model.
- `get_batch_size(self) -> int`: Returns the batch size for training.
- `get_train_epochs(self) -> int`: Returns the number of training epochs.
- `get_num_classes(self) -> int`: Returns the number of classes in the client's dataset.

##### Public Methods
- `run(self) -> None`: Executes the main operations of the client.
- `enable_op_determinism(self) -> None`: Configures training to use deterministic operations, ensuring reproducibility of experimental results.
- `shuffle_dataset_each_epoch(self, value: bool) -> None`: Enables or disables shuffling the training dataset at the beginning of each epoch. Enabled by default.


### Supported aggregation algorithms

The framework supports a modular design for various server-side aggregation strategies. A detailed description of all implemented algorithms, including their formulas and sources, is available in a separate document.

**➡️ [See the full list and detailed descriptions in ALGORITHMS.md](src/federated_sim/ALGORITHMS.md)**

### Message exchange
The server and clients communicate using the TCP/IP protocol. Each exchanged message is composed of a byte sequence, with the first four bytes indicating the message length. The message consists of the following fields:
- `type`: Message type.
- `body`: Message body.

Defined message types include:
- `FEDERATED_WEIGHTS`: Contains the federated model created by the server.
- `CLIENT_MODEL`: Contains the model trained by a client.
- `END_FL_TRAINING`: Indicates the end of federated learning.
- `CLIENT_EVALUATION`: Sends client evaluations for each round.

The message is serialized using the *pickle module*, which transforms the message into a sequence of bytes. The generated sequence is concatenated with the initial 4 bytes representing the total length of the message.

### Profiling
Through the initial configurations, the server can decide whether to enable profiling on the nodes or not. If enabled, each node will, at the end of federated learning, send the following information to the server:

+ ```training_n_instructions```: number of instructions executed during the training phases.
+ ```training_execution_time```: total execution time of the training phases.
+ ```max_ram_used```: maximum RAM used by the node.
+ ```bytes_input```: number of bytes downloaded (received) by the client.
+ ```train_samples```: number of samples involved in the training of the local model.
+ ```test_samples```: number of samples involved in the test evaluation of the local/federated model.

The server will also save:

+ ```bytes_output```: number of bytes uploaded (sent) to each client.

### Limits of the implementation
The proposed implementation is very simple, and for this reason, some simplifications were necessary, leading to the following limitations:

+ Server and client remain connected until the end of federated learning. In a realistic scenario, clients might participate only in certain rounds, or, better yet, clients may not be available simultaneously.
+ Dynamic addition of new clients during the learning phase is not supported. The clients participating in learning will remain connected until the last round. No one can opt-out or join later.
+ Security aspects are missing; anyone can connect to the server.
+ The exchanged messages are not encrypted.
+ ...and so on.

## Requirements
1. Install the Python development environment
```
sudo apt install python3-dev python3-pip  # Python 3
```
2. Install tensorflow Python package
```
pip install tensorflow
```

## Simulation Mnist Dataset

### Install required packages
1. Install [Requirements](#requirements).

2. Install the released TensorFlow Federated Python package (for the federated mnist dataset)
```
pip install --upgrade tensorflow-federated
```

### Launch simulation
1. Execute Server script:
```
python3 examples/mnist/Server.py 
```
2. Execute Clients script manually:
```
python3 examples/mnist/Client.py id_client
```
> [!IMPORTANT]
> To perform a simulation, you need to execute the client script a number of times equal to the value of the variable ***number_clients*** contained in the Server class. This is because federated training starts when that number of clients is connected to the server. It goes without saying that the client ID specified with each execution of the client script must be a positive integer, different for each run.

2. (optional) Execute Clients script automatically:
Automating the execution of clients is possible by ```start_clients.sh``` bash script:
```
./examples/mnist/start_clients.sh n_clients
```
> [!IMPORTANT]
> It will execute the Client.py script 'n_clients' times on a different terminal. As before n_clients must be equal to ***number_clients***. You can specify any type of terminal; in our case, xterm is used, so if you want to use this script, make sure you have it installed!!!. 

## Simulation BNCI2014_001 Dataset

### Install required packages
1. Install [Requirements](#requirements).

2. Install MOABB
```
pip install MOABB
```

### Launch simulation
1. Execute Server script:
```
python3 examples/BNCI2014_001/Server.py 
```
2. Execute Clients script automatically:
Automating the execution of clients is possible by ```start_clients.sh``` bash script:
```
./examples/BNCI2014_001/start_clients.sh
```
> [!Note]
> It will execute the Client.py script 9 times on a different terminal. You can specify any type of terminal; in our case, xterm is used, so if you want to use this script, make sure you have it installed!!!. 
