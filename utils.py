import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib as mpl
from sys import stdout
import torch
import torch.nn as nn
import nni
import json


mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 14
# mpl.rcParams["font.family"] = "Times New Roman"


def print_progress(batch_index, len_data, job=""):
    prog = int(100 * (batch_index + 1) / len_data)
    stdout.write("\r\r\r\r\r\r\r\r" + job + " %d" % prog + "%")
    print("", end="\n" if prog == 100 else "")
    stdout.flush()


class FeaturesModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.module.encoder
        self.avgpool = model.module.avgpool

    def forward(self, x):
        x = self.encoder(x)[0]
        x = self.avgpool(x).squeeze()
        return x


def extract_features_and_labels(model, dataloader, epoch_num, feat_dim, _params, id, train=True, pre=False, device="cuda", _nni=False, devices=None):
    features_model = FeaturesModel(model) if pre else nn.Sequential(*list(model.module.children())[:-1])
    # features_model = nn.Sequential(model.module.encoder, model.module.avgpool) if pre else nn.Sequential(*list(model.module.children())[:-1])
    features_model = nn.DataParallel(features_model, device_ids=devices, output_device=devices[0])
    features_model.eval()
    # feat_dim = model.fc.in_features
    # create empty placeholders
    features = torch.empty((0, feat_dim))
    labels = torch.empty(0, dtype=torch.long)
    for (idx, batch) in enumerate(dataloader):
        with torch.no_grad():
            image = batch['image']
            curr_labels = batch['label']
            curr_feats = features_model(image.to(device)).squeeze()
            features = torch.cat((features, curr_feats.cpu().detach()))
            labels = torch.cat((labels, curr_labels))
        if not _nni:
            print_progress(idx, len(dataloader), job="features")
    trial = nni.get_trial_id() if _nni else ""
    s = "train_{}".format(trial) if train else "test_{}".format(trial)
    # p = "_pre_ddp" if pre else ""
    with open("features/features_{}_{}_{} dsi04.pickle".format(s, epoch_num, id), "wb") as handle:
        pickle.dump(features, handle)
    with open("labels/labels_{}_{}_{} dsi04.pickle".format(s, epoch_num, id), "wb") as handle:
        pickle.dump(labels, handle)
    return features.numpy(), labels.numpy()


def save_results(train_features, test_features, train_labels, test_labels, losses, epoch_num, i, id, pre):

    with open("loss/loss_{}_{} dsi04.pickle".format(epoch_num, id), "wb") as outfile:
        pickle.dump(losses, outfile)

    # p = "_pre_ddp" if pre else ""

    # train_embd = TSNE(n_components=2).fit_transform(train_features)
    # print(2)
    # test_embd = TSNE(n_components=2).fit_transform(test_features)
    # print(3)
    #
    # with open("tsne_features/tsne_train{}_{}.pickle".format(p, epoch_num), "wb") as handle:
    #     pickle.dump(train_embd, handle)
    # with open("tsne_features/tsne_test{}_{}.pickle".format(p, epoch_num), "wb") as handle:
    #     pickle.dump(test_embd, handle)
    #
    # classes = np.unique(train_labels)
    # plt.rcParams["figure.figsize"] = (13, 5)
    # plt.figure(i)
    # plt.scatter(train_embd.T[0], train_embd.T[1], c=train_labels, cmap="gist_ncar")
    # plt.title('Train Representation Embedding, Epoch {}'.format(epoch_num))
    # plt.tight_layout()
    # plt.savefig("figures/tsne_train{}_{}.png".format(p, epoch_num))
    # plt.figure(i+1)
    # plt.scatter(test_embd.T[0], test_embd.T[1], c=test_labels, cmap="gist_ncar")
    # plt.title('Test Representation Embedding, Epoch {}'.format(epoch_num))
    # plt.tight_layout()
    # plt.savefig("figures/tsne_test{}_{}.png".format(p, epoch_num))

    # plt.figure(i+2, figsize=(11, 8))
    # plt.plot(list(losses.keys()), list(losses.values()), 's', marker="o", markersize=0.8, linestyle="-", lw=0.4)
    # plt.xlabel("Number of epochs")
    # plt.ylabel("Loss")
    # plt.title("Loss as function of Epochs")
    # plt.savefig("figures/loss_{}_{}.png".format(epoch_num, id))


# with open('features/features_train_pre2_480.pickle', 'rb') as handle:
#     features_train = pickle.load(handle).numpy()
# with open('features/features_test_pre2_480.pickle', 'rb') as handle:
#     features_test = pickle.load(handle).numpy()
# with open('labels/labels_train_pre2_480.pickle', 'rb') as handle:
#     labels_train = pickle.load(handle).numpy()
# with open('labels/labels_test_pre2_480.pickle', 'rb') as handle:
#     labels_test = pickle.load(handle).numpy()
#
# print(1)
# save_results(features_train, features_test, labels_train, labels_test, 1, 480, 0, True)
