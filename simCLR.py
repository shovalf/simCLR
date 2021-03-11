import torch
import numpy as np
import torch.nn as nn
from sys import stdout
import os
from copy import deepcopy
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import torchvision
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from CUB_dataset import *
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.models.self_supervised import SimCLR
import math
import time
import pickle


PRE = True
NNI = False


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask(self.batch_size).type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self, _size):
        diag = np.eye(2 * _size)
        l1 = np.eye((2 * _size), 2 * _size, k=-_size)
        l2 = np.eye((2 * _size), 2 * _size, k=_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)
        size_ = zis.shape[0]

        similarity_matrix = self.similarity_function(representations, representations)
        self.batch_size = size_
        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, size_).to(device=zis.device)
        r_pos = torch.diag(similarity_matrix, -size_).to(device=zis.device)
        positives = torch.cat([l_pos, r_pos]).view(2 * size_, 1)

        negatives = similarity_matrix[self._get_correlated_mask(self.batch_size).type(torch.bool)].view(2 * size_, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * size_).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * size_)


class SimCLRModel(nn.Module):
    def __init__(self, pre, weight_path='https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/simclr-cifar10-v1-exp12_87_52/epoch%3D960.ckpt'):
        super().__init__()
        self.pre = pre
        self.weight_path = weight_path

        if pre:
            self.encoder = deepcopy(SimCLR.load_from_checkpoint(weight_path, strict=False).encoder)
        else:
            self.encoder = torchvision.models.resnet18(pretrained=False)

        self.freeze()

        self.avgpool = self.encoder.avgpool
        numft = self.encoder.fc.in_features
        self.fc = nn.Sequential(
            nn.Linear(in_features=numft, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=20))

    def freeze(self):
        c = 0
        l = 0
        num_layer = 0
        for _ in self.encoder._modules['layer4'].parameters():
            num_layer += 1
        for _ in self.encoder.parameters():
            l += 1
        for params in self.encoder.parameters():
            if c < l - num_layer - 2:
                params.requires_grad = False
            c += 1

    def forward(self, x):
        x = self.encoder(x)[0]
        x = self.avgpool(x).squeeze()
        x = self.fc(x)
        return x


def create_optimizer(model, lr, momentum=0.9):
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum)


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
    features_model = nn.DataParallel(features_model, device_ids=devices, output_device=devices[0])
    features_model.eval()
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
    # with open("features/features_{}_{}_{} dsi04.pickle".format(s, epoch_num, id), "wb") as handle:
    #     pickle.dump(features, handle)
    # with open("labels/labels_{}_{}_{} dsi04.pickle".format(s, epoch_num, id), "wb") as handle:
    #     pickle.dump(labels, handle)
    return features.numpy(), labels.numpy()


def self_supervised_training(model, optimizer, train_dataloader, raw_train_loader, raw_test_loader, _params, id, devices, temperature=0.07,
                             num_epochs=10, print_freq=150, device="cuda", pre=True):

    # move model to device
    model = model.to(device)
    loss_values = {}
    i = 0
    features_dim = model.module.encoder.inplanes if pre else model.module.inplanes
    nt_xent_criterion = NTXentLoss(device=device, batch_size=train_dataloader.batch_size, temperature=temperature,
                                   use_cosine_similarity=True)
    model.train()

    for epoch in range(num_epochs):
        if not NNI:
            print('In epoch {}'.format(epoch))
        loss_value = 0

        for (idx, aug_batch) in enumerate(train_dataloader):
            t = time.time()
            optimizer.zero_grad()
            values = list(aug_batch.values())

            image1, image2 = values[0].to(device), values[1].to(device)

            output1 = model(image1)
            output2 = model(image2)

            loss = nt_xent_criterion(output1, output2)
            loss.backward()
            optimizer.step()
            loss_value = loss.cpu().data.item()
            print_progress(idx, len(train_dataloader), job="TRAIN")

        loss_values.update({epoch: loss_value})

        if not NNI:
            print(f"Epoch {epoch}, Contrastive loss={loss_value}")

        if epoch == 149 or epoch == 299 or epoch == num_epochs - 1:
            train_features, train_labels = extract_features_and_labels(model, raw_train_loader, epoch,
                                                                       features_dim, _params, id, train=True, pre=pre, device=device, devices=devices)
            test_features, test_labels = extract_features_and_labels(model, raw_test_loader, epoch, features_dim, _params, id,
                                                                     train=False, pre=pre, device=device, devices=devices)
            i += 3
    return train_features, train_labels, test_features, test_labels


def model(dict_params, _server, id):
    s = 1
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    transform_train = transforms.Compose([transforms.Resize((250, 250)),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.GaussianBlur(23, sigma=(0.1, 2.0)),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                          ])

    # basic transformation on test images
    transform_test = transforms.Compose(
        [
          transforms.Resize((224, 224)), # resize image
          transforms.ToTensor(), # numpy array to tensor
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalize the image
        ])

    batch_size = dict_params["batch_size"]

    dataset = Cub2011(root="CUB", train=True, transform=transform_train, all=True, augment=True)
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)

    # train-set with augmentations
    raw_train_dataset = Cub2011(root="CUB", train=True, transform=transform_test, augment=False)
    # set train loader with augmentations
    raw_train_loader = DataLoader(raw_train_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # test-set
    raw_test_dataset = Cub2011(root="CUB", train=False, transform=transform_test, augment=False)
    # test loader
    raw_test_loader = DataLoader(raw_test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    num_epochs = dict_params["epochs"]

    if isinstance(dict_params["lr"], float) is True:
        lr = dict_params["lr"]
    elif dict_params["lr"] == "1":
        lr = float(0.3 * batch_size / 256)
    elif dict_params["lr"] == "2":
        lr = float(0.075 * math.sqrt(batch_size))
    else:  # like 1
        lr = float(0.3 * batch_size / 256)

    devices = [7, 6] if _server == "dgx" else [3, 1, 0, 2]

    # Create new instance of the model
    simclr_model = SimCLRModel(PRE)
    simclr_model = nn.DataParallel(simclr_model, device_ids=devices, output_device=devices[0])

    # LARS optimizer
    base_optimizer = optim.SGD(simclr_model.parameters(), lr=lr, momentum=0.9)
    simclr_opt = LARSWrapper(base_optimizer, eta=dict_params["eta"], clip=True, eps=1e-8)

    train_features, train_labels, test_features, test_labels = \
        self_supervised_training(simclr_model, simclr_opt, dataset_loader, raw_train_loader, raw_test_loader,
                                 dict_params, id, devices=devices, num_epochs=num_epochs, print_freq=30,
                                 device="cuda:{}".format(devices[0]), temperature=dict_params["temperature"], pre=PRE)
    return train_features, train_labels, test_features, test_labels


def main_simclr():
    if not NNI:
        _n_gpus = torch.cuda.device_count()
        print(torch.cuda.current_device())
        _server = 'dsi' if _n_gpus == 4 else 'dgx'
        _gpu = "1, 0, 3, 2" if _server == 'dsi' else "7, 0, 1, 2, 4, 5, 3, 6"
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = _gpu
        print("Using gpu {}".format(_gpu))

        if PRE:
            _batch_size = 256 if _server == "dsi" else 512
        else:
            _batch_size = 256 if _server == "dsi" else 512

        _num_epochs = 5
        _lr = "1"
        _temperature = 0.1
        _eta = 0.02

        # _batch_size = 512
        dict_params = {"batch_size": _batch_size, "lr": _lr, "epochs": _num_epochs, "temperature": _temperature, "eta": _eta}

        train_features, train_labels, test_features, test_labels = model(dict_params, _server, 1)

    return train_features, train_labels, test_features, test_labels


train_features, train_labels, test_features, test_labels = main_simclr()
