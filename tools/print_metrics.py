import os
import numpy as np

def get_metric_by_round(avg_acc, avg_loss, rounds):
    n_nodes = len(avg_acc)
    acc_r = []
    loss_r = []
    for i in range(rounds + 1):
        avg_acc_sum = 0
        avg_loss_sum = 0
        for k in range(n_nodes):
            avg_acc_sum += avg_acc[k][i]
            avg_loss_sum += avg_loss[k][i]

        acc_r.append(avg_acc_sum / n_nodes)
        loss_r.append(avg_loss_sum / n_nodes)

    return acc_r, loss_r


def print_nodes_metric_by_round(avg_acc, avg_loss, round):
    print(f"Round: {round}")
    n_nodes = len(avg_acc)
    avg_acc_sum = 0
    avg_loss_sum = 0
    for i in range(n_nodes):
        print(f"node {i}, accuracy {avg_acc[i][round]}, loss = {avg_loss[i][round]}")
        avg_acc_sum += avg_acc[i][round]
        avg_loss_sum += avg_loss[i][round]

    print(f"avg_accuracy {avg_acc_sum/n_nodes}, avg_loss {avg_loss_sum/n_nodes}\n")
def print_avg_metric(acc, loss, round):
    print(f"Round: {round}, accuracy = {acc[round]}, loss = {loss[round]}")

# directory = '../examples/ultracortex_eeg/evaluations/'
acc_name = "accuracy_FedAvg_24-48rounds.npy"
loss_name = "loss_FedAvg_24-48rounds.npy"
# file_path_avg_acc = os.path.join(directory, acc_name)
# file_path_avg_loss = os.path.join(directory, loss_name)
# avg_acc = np.load(file_path_avg_acc, allow_pickle=True)
# avg_loss = np.load(file_path_avg_loss, allow_pickle=True)
# #
# print_avg_metric(avg_acc,avg_loss,8)
# print_avg_metric(avg_acc,avg_loss,24)
# print_avg_metric(avg_acc,avg_loss,48)
#
directory = '../examples/ultracortex_eeg/evaluations/nodes'
file_path_node_acc = os.path.join(directory, acc_name)
file_path_node_loss = os.path.join(directory, loss_name)
node_acc = np.load(file_path_node_acc, allow_pickle=True)
node_loss = np.load(file_path_node_loss, allow_pickle=True)
#

print_nodes_metric_by_round(node_acc, node_loss, 24)
# print_nodes_metric_by_round(node_acc, node_loss, 48)
#

# acc_r, loss_r = get_metric_by_round(node_acc, node_loss, 32)
# #
# directory = '../examples/ultracortex_eeg/evaluations/'
# acc_name = "accuracy_FedAdam_32rounds.npy"
# loss_name = "loss_FedAdam_32rounds.npy"
#
# np.save(os.path.join(directory, acc_name), acc_r)
# np.save(os.path.join(directory, loss_name), loss_r)