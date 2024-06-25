import os
import numpy as np

def print_nodes_metric_by_round(avg_acc, avg_loss, round):
    print(f"Round: {round}")
    for i in range(len(avg_acc)):
        print(f"node {i}, accuracy {avg_acc[i][round]}, loss = {avg_loss[i][round]}")

def print_avg_metric(acc, loss, round):
    print(f"Round: {round}, accuracy = {acc[round]}, loss = {loss[round]}")

directory = '../examples/mnist/evaluations/'
acc_name = "accuracy_FedAvg_64rounds.npy"
loss_name = "loss_FedAvg_64rounds.npy"
file_path_avg_acc = os.path.join(directory, acc_name)
file_path_avg_loss = os.path.join(directory, loss_name)
avg_acc = np.load(file_path_avg_acc, allow_pickle=True)
avg_loss = np.load(file_path_avg_loss, allow_pickle=True)

print_avg_metric(avg_acc,avg_loss,8)
print_avg_metric(avg_acc,avg_loss,16)
print_avg_metric(avg_acc,avg_loss,32)
print_avg_metric(avg_acc,avg_loss,64)

directory = '../examples/mnist/evaluations/nodes'
file_path_node_acc = os.path.join(directory, acc_name)
file_path_node_loss = os.path.join(directory, loss_name)
node_acc = np.load(file_path_node_acc, allow_pickle=True)
node_loss = np.load(file_path_node_loss, allow_pickle=True)

print_nodes_metric_by_round(node_acc, node_loss, 8)
print_nodes_metric_by_round(node_acc, node_loss, 16)
print_nodes_metric_by_round(node_acc, node_loss, 32)
print_nodes_metric_by_round(node_acc, node_loss, 64)