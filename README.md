# Federated Learning: clients/server desktop implementation
This is a design of a simple client/server architecture to simulate federated learning involving real nodes, where each node (client) owns its own data. The client and server are written in Python and communicate via sockets using the TCP transport protocol.

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
