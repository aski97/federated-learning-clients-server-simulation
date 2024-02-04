import os
import numpy as np
import moabb
from moabb.paradigms import LeftRightImagery
from moabb.datasets import BNCI2014_001
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

moabb.set_log_level("info")

# Load dataset BNCI2014001
dataset = BNCI2014_001()
subjects = dataset.subject_list

# Choose LeftRightImagery paradigm
fmin = 8
fmax = 35
paradigm = LeftRightImagery(fmin=fmin, fmax=fmax)


# Get and save data by subjects
for subj in subjects:
    # get data
    X, y, _ = paradigm.get_data(dataset=dataset, subjects=[subj])

    # Processing data with TGSP
    cov = Covariances('oas').transform(X)
    X_tgsp = TangentSpace(metric="riemann").transform(cov)

    save_path = f"dataset/{subj}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save data
    np.save(os.path.join(save_path, "data.npy"), X_tgsp)
    np.save(os.path.join(save_path, "labels.npy"), y)

    print(f"Data saved in: {save_path}")




