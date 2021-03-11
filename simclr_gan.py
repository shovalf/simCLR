from train_CUB import *
from simCLR import *

# import nni
import logging
import json
import pandas as pd
import itertools

# logger = logging.getLogger("NNI_logger")
NONE = None


# _n_gpus = torch.cuda.device_count()
# # _n_gpus = 1
# print(torch.cuda.current_device())
# _server = 'dsi' if _n_gpus == 4 else 'dgx'
# # _gpu = "1, 0, 3, 2" if _server == 'dsi' else "7, 0, 1, 2, 4, 5, 3, 6"
# # _gpu = "1, 0"
# # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# # os.environ["CUDA_VISIBLE_DEVICES"] = _gpu
# # print("Using gpu {}".format(_gpu))


def run_trial(dict_params, server, id):
    _accuracy = {}
    simclr_results = model(dict_params, server, id)
    for key in list(simclr_results.keys()):
        (train_features, train_labels, test_features, test_labels) = simclr_results[key]
        acc = train(_nni=False, train_labels=train_labels, train_features=train_features, test_labels=test_labels, test_features=test_features)
        _accuracy[key] = acc
    return _accuracy


# def main():
#     try:
#         params = nni.get_next_parameter()
#         logger.debug(params)
#         run_trial(params)
#     except Exception as exception:
#         logger.error(exception)
#         raise


def grid_search(server):
    with open("grid_params.json") as f:
        dict_params = json.load(f)
    keys, values = zip(*dict_params.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    data = {i: experiments[i] for i in range(len(experiments))}

    csv_file = "SimCLR grid search results dsi 04.csv"
    csv_columns = list(experiments[0].keys())
    acc = ["acc150", "acc300", "acc500"]
    for c in acc:
        csv_columns.append(c)

    f = open(csv_file, "wt")
    header = csv_columns
    f.write(",".join(header) + "\n")
    f.close()

    for id, experiment in enumerate(experiments):
        accuracy = run_trial(experiment, server, id)
        experiment.update({j: accuracy[j] for j in acc})
        f = open(csv_file, "a")
        f.write(",".join([str(x) for x in list(experiment.values())]) + "\n")  # parameters
        f.close()


if __name__ == "__main__":
    _n_gpus = torch.cuda.device_count()
    print(torch.cuda.current_device())
    _server = 'dsi' if _n_gpus == 4 else 'dgx'
    _gpu = "1, 0, 3, 2" if _server == 'dsi' else "0, 1, 2, 3, 6, 7"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu
    print("Using gpu {}".format(_gpu))
    grid_search(_server)
