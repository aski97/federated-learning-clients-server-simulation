# Federated Learning: clients/server desktop implementation
This is a design of a simple client/server architecture to simulate federated learning involving real nodes, where each node (client) owns its own data. The client and server are written in Python and communicate via sockets using the TCP transport protocol.

## Table of contents

* [Architecture](#architecture)
    + [Server](#server)
    + [Client](#client)
    + [Message exchange](#message-exchange)
    + [Limits of the implementation](#limits-of-the-implementation)
* [Simulation Mnist Dataset](#simulation-mnist-dataset)
    + [Requirements](#requirements)
    + [Launch simulation](#launch-simulation)

## Architecture
![fl_architecture](/images/fl_arc.png)

Here is a representation of the architecture, as shown in the figure. The server awaits the connection of nodes participating in federated learning. Whenever a client connects, the server sends an initialized federated model. Once a certain number of nodes are connected, the federated learning process begins:

1. Clients train their local models on local data using federated weights received from the server.
2. Upon completion of local training, clients send the new weights to the server.
3. The server aggregates the weights from all clients, generating a new model.
4. The server sends the new federated model to the clients.
This sequence is iterated for a predetermined number of rounds, at the end of which, clients send the server accuracy and loss data of the federated model before closing the connection with the server.

###  Server
The server is implemented in the file [TCPServer.py](/TCPServer.py). When the script is executed, a socket is opened at the specified address, in our case, localhost:12345, and three threads are created:

+ A thread, **thread_client_connections**, listens to accept client connections. It accepts up to a number of connections equal to *NUMBER_OF_CLIENTS*. For each connected client, a *client_thread* is associated with it to handle communication.
+ A thread, **thread_fl_algorithms**, manages rounds for federated learning. Once all clients send their weights, it calculates the average of all weights and sends the new model to the clients.
+ A thread, **thread_final_evaluations**, performs the final evaluation of the learning. When the learning process concludes, it collects the accuracies and losses of each local model and creates graphs.

At this point, the server waits for client connections, and when a connection is initialized, the server sends an initialized model to the node, represented by weights and biases with values set to 0.

When all clients are connected, the server waits for the reception of local models, thus starting the learning phase that lasts for a number of rounds defined by the *ROUNDS* variable. At each round, when the server collects all models, the following sequence of events occurs:

1. Models (weights and biases) are aggregated by taking the average of the models.
2. The server sends the resulting model to all clients.

The learning process concludes when the number of rounds is exhausted, resulting in a final model. The thread responsible for evaluation proceeds to provide graphs that describe the learning progress, including:

+ ***Average accuracy of the final model***.
+ ***Average loss of the final model***.
+ ***Trend of accuracy per round for predicting test samples for each client***.
+ ***Trend of loss per round for predicting test samples for each client***.
+ ***Trend of average accuracy per round***.
+ ***Trend of average loss per round***.

### Client
The client is implemented in the script [TCPClient.py](/TCPClient.py). Upon execution, it reads its own *ID* from the command line and initializes its dataset, consisting of samples used for training and samples used for testing.

The client creates a socket and connects to the server's address. Then, it waits to receive the federated model from the server. Once it receives the weights and biases, it loads them into the local model (net) and starts an initial evaluation. In this phase, the evaluation helps understanding the accuracy of the federated model using the local test dataset. After evaluating the federated model, the client starts the training on the training data.

Before each training, the dataset is shuffled to avoid overfitting situations and divided into batches, where each batch contains a number of samples equal to **BATCH_SIZE**. Training proceeds for a number of epochs specified by **EPOCHES**. Upon completion, the model is re-evaluated on the test data, and the results are stored. Afterward, the client sends the weights and biases of the just-trained model to the server. This operation repeats until the server sends the final model, on which the client performs a single evaluation, sending all previous evaluations back to the server.

Once the federated training is complete, the client closes the connection with the server.

### Message exchange
The communication between the server and clients occurs through the TCP transport protocol. Each transmitted message consists of a variable-length sequence of bytes, where the first four bytes indicate the message length.

The message is composed of:

+ type: message type.
+ body: message body.

Based on the message type, the structure of the body can be determined. Currently, the following types are specified:

+ ***FEDERATED_WEIGHTS***: indicates that the message body contains the federated model created by the server by aggregating the client models.
+ ***CLIENT_TRAINED_WEIGHTS***: indicates that the message body contains the model trained by a client. The body will be in the form of the following dictionary:
  - *client_id*: client identifier
  - *weights*: weights and biases of the trained model.
+ ***END_FL_TRAINING***: indicates the end of federated learning, and the message body contains the final federated model.
+ ***CLIENT_EVALUATION***: indicates the sending of evaluations for each round from a client. The message body will be in the form of:
  - *client_id*: client identifier
  - *evaluation_federated*: a list of arrays of evaluations on the test dataset using the federated model. Each array is formatted as *[accuracy, loss]*. The length of the list is equal to the number of rounds.
  - *evaluation_training*: a list of arrays of evaluations on the test dataset using the locally trained model. Each array is formatted as *[accuracy, loss]*. The length of the list is equal to the number of rounds.

The message is serialized using the *pickle module*, which transforms the message into a sequence of bytes. The generated sequence is concatenated with the initial 4 bytes representing the total length of the message.

### Limits of the implementation
The proposed implementation is very simple, and for this reason, some simplifications were necessary, leading to the following limitations:

+ Server and client remain connected until the end of federated learning. In a realistic scenario, clients might participate only in certain rounds, or, better yet, clients may not be available simultaneously.
+ Dynamic addition of new clients during the learning phase is not supported. The clients participating in learning will remain connected until the last round. No one can opt-out or join later.
+ Security aspects are missing; anyone can connect to the server.
+ The exchanged messages are not encrypted.
+ ...and so on.

## Simulation Mnist Dataset

### Requirements
1. Install the Python development environment
```
sudo apt install python3-dev python3-pip  # Python 3
```
2. Install the released TensorFlow Federated Python package (for the federated mnist dataset)
```
pip install --upgrade tensorflow-federated
```
2. (optional) If you want to use your own dataset, you can just use tensorflow Python package instead of the federated version
```
pip install tensorflow
```

### Launch simulation
1. Execute Server script:
```
python3 TCPServer.py 
```
2. Execute Clients script manually:
```
python3 TCPClient.py id_client
```
> [!IMPORTANT]
> To perform a simulation, you need to execute the client script a number of times equal to the value of the variable ***NUMBER_OF_CLIENTS*** contained in the TCPServer class. This is because federated training starts when that number of clients is connected to the server. It goes without saying that the client ID specified with each execution of the client script must be a positive integer, different for each run.

2. (optional) Execute Clients script automatically:
Automating the execution of clients is possible by ```start_clients.sh``` bash script:
```
./start_clients.sh n_clients
```
> [!IMPORTANT]
> It will execute the TCPClient.py script 'n_clients' times on a different terminal. As before n_clients must be equal to ***NUMBER_OF_CLIENTS***. You can specify any type of terminal; in our case, xterm is used, so if you want to use this script, make sure you have it installed!!!. 
