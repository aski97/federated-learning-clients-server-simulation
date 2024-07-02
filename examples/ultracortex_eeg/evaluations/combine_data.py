import numpy as np


fed_alg = "FedAdam"

round_1 = 24
round_2 = 24


def combine_avg(metric):
    first_filepath = f"to_combine/{metric}_{fed_alg}_{round_1}rounds.npy"
    second_filepath = f"{metric}_{fed_alg}_{round_2}rounds.npy"

    f_node_acc = np.load(first_filepath, allow_pickle=True)
    s_node_acc = np.load(second_filepath, allow_pickle=True)

    s_node_acc = np.delete(s_node_acc,0)

    combined = np.concatenate((f_node_acc,s_node_acc), axis=0)

    np.save(f"{metric}_{fed_alg}_{round_2 + round_1}rounds.npy", combined)


# combine_avg("accuracy")
# combine_avg("loss")