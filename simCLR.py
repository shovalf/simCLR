# import torch
# import numpy as np
# import torch.nn as nn
import os
from copy import deepcopy
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import torchvision
import torch.optim as optim
from torchvision import transforms
import itertools
from torch.utils.data import DataLoader
from CUB_dataset import *
# import pickle
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# import matplotlib as mpl
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.models.self_supervised import SimCLR
import math
from termcolor import cprint
import time
from utils import *
# from train_CUB import *
import gc
PRE = True
NNI = False



# mpl.rcParams['xtick.labelsize'] = 12
# mpl.rcParams['ytick.labelsize'] = 12
# mpl.rcParams['axes.titlesize'] = 18
# mpl.rcParams['axes.labelsize'] = 14
# mpl.rcParams["font.family"] = "Times New Roman"


def simclr_cosine_similarity():
    cosine_similarity = torch.nn.CosineSimilarity(dim=-1)


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
        # l_pos = torch.diag(similarity_matrix, self.batch_size).to(device=zis.device)
        # r_pos = torch.diag(similarity_matrix, -self.batch_size).to(device=zis.device)
        l_pos = torch.diag(similarity_matrix, size_).to(device=zis.device)
        r_pos = torch.diag(similarity_matrix, -size_).to(device=zis.device)
        positives = torch.cat([l_pos, r_pos]).view(2 * size_, 1)

        try:
            negatives = similarity_matrix[self._get_correlated_mask(self.batch_size).type(torch.bool)].view(2 * size_, -1)
            # negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * size_, -1)
        except:
            negatives = 0

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * size_).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * size_)


# def create_simclr_model(pre):
#     if pre:
#         # extract only encoder
#         weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/simclr-cifar10-v1-exp12_87_52/epoch%3D960.ckpt'
#         simclr_model = SimCLR.load_from_checkpoint(weight_path, strict=False)
#         simclr_model = deepcopy(simclr_model.encoder)
#
#     else:
#         # create new resnet18 as the base encoder
#         simclr_model = torchvision.models.resnet18(pretrained=False)
#
#     for params in simclr_model.parameters():
#         params.requires_grad = False
#
#     numft = simclr_model.fc.in_features
#
#     # change its classifier to a projection head
#     simclr_model.fc = nn.Sequential(
#             nn.Linear(in_features=numft, out_features=100),
#             nn.ReLU(),
#             nn.Linear(in_features=100, out_features=40),
#             nn.ReLU(),
#             nn.Linear(in_features=40, out_features=20),
#         )
#     return simclr_model


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
    # if MODULE:
    #     model = model.module
    # a = torch.tensor([]).to(device)
    # b = torch.tensor([]).to(device)

    for epoch in range(num_epochs):
        if not NNI:
            print('In epoch {}'.format(epoch))
        loss_value = 0

        # try:
        for (idx, aug_batch) in enumerate(train_dataloader):
            t = time.time()
            # print('In batch {}'.format(idx))
            optimizer.zero_grad()
            values = list(aug_batch.values())

            image1, image2 = values[0].to(device), values[1].to(device)
            # output1 = model(image1)[0]
            # output2 = model(image2)[0]

            output1 = model(image1)
            output2 = model(image2)
            # output1, output2 = model.module.avgpool(output1).squeeze(), model.module.avgpool(output2).squeeze()
            # output1, output2 = model.module.fc(output1), model.module.fc(output2)
            # a = torch.cat((a, output1))
            # b = torch.cat((b, output2))

            # print('output1 shape {}'.format(output1.shape))
            # print('Out of model')
            loss = nt_xent_criterion(output1, output2)
            loss.backward()
            optimizer.step()
            loss_value = loss.cpu().data.item()
            # print('Loss is calculated')
            # if not NNI:
            #     cprint(f"Time: {time.time()-t}", color="blue")
            #     cprint(f"Finished batch: {idx}", color="red")
            print_progress(idx, len(train_dataloader), job="TRAIN")

        # except:
        #     print('sus2')

        loss_values.update({epoch: loss_value})

        if not NNI:
            print(f"Epoch {epoch}, Contrastive loss={loss_value}")

        # if epoch % print_freq == 0 or epoch == int(num_epochs - 1):
        if epoch == 149 or epoch == 299 or epoch == num_epochs - 1:
            train_features, train_labels = extract_features_and_labels(model, raw_train_loader, epoch,
                                                                       features_dim, _params, id, train=True, pre=pre, device=device, devices=devices)
            test_features, test_labels = extract_features_and_labels(model, raw_test_loader, epoch, features_dim, _params, id,
                                                                     train=False, pre=pre, device=device, devices=devices)
            save_results(train_features, test_features, train_labels, test_labels, loss_values, epoch, i, id, pre=pre)
            i += 3
            yield train_features, train_labels, test_features, test_labels


def self_supervised_training_mini_batch(model, optimizer, train_dataloader, raw_train_loader, raw_test_loader, batch_multiplier=8,
                              temperature=0.07, num_epochs=10, print_freq=30, device="cuda", pre=True):
    # move model to device
    model = model.to(device)
    loss_values = {}
    i = 0
    features_dim = model.module.encoder.inplanes if pre else model.module.inplanes
    nt_xent_criterion = NTXentLoss(device=device, batch_size=train_dataloader.batch_size, temperature=temperature,
                                   use_cosine_similarity=True)

    # if MODULE:
    #     model = model.module

    for epoch in range(num_epochs):
        print('In epoch {}'.format(epoch))
        loss_value = 0
        model.train()
        # try:
        count = batch_multiplier
        # f_output1, f_output2 = torch.tensor(np.zeros([0, 128])).to(device), torch.tensor(np.zeros([0, 128])).to(device)
        for (idx, aug_batch) in enumerate(train_dataloader):
            if count == 0:
            # print('In batch {}'.format(idx))
            #     loss = nt_xent_criterion(f_output1, f_output2)
            #     loss = nt_xent_criterion(output1, output2)
            #     loss.backward()
            #     loss_value = loss.cpu().data.item()
                optimizer.step()
                optimizer.zero_grad()
                count = batch_multiplier
                # f_output1, f_output2 = torch.tensor(np.zeros([0, 128])).to(device), torch.tensor(np.zeros([0, 128])).to(device)
            values = list(aug_batch.values())

            image1, image2 = values[0].to(device), values[1].to(device)
            output1, output2 = model(image1), model(image2)
            if pre:
                output1, output2 = model.module.projection(output1), model.module.projection(output2)
            # f_output1, f_output2 = torch.cat((f_output1, output1)), torch.cat((f_output2, output2))
            loss = nt_xent_criterion(output1, output2)/batch_multiplier
            loss.backward()
            loss_value = loss.cpu().data.item()
            count -= 1
            # print('output1 shape {}'.format(output1.shape))
            # print('Out of model')

            # print('Loss is calculated')
            print_progress(idx, len(train_dataloader), job="TRAIN")

        # except:
        #     print('sus2')

        loss_values.update({epoch: loss_value})

        print(f"Epoch {epoch}, Contrastive loss={loss_value}")
        #
        if epoch % print_freq == 0 or epoch == int(num_epochs - 1):
            train_features, train_labels = extract_features_and_labels(model, raw_train_loader, epoch,
                                                                       features_dim, train=True, pre=pre, device=device)
            test_features, test_labels = extract_features_and_labels(model, raw_test_loader, epoch, features_dim,
                                                                     train=False, pre=pre, device=device, _nni=NNI)
            save_results(train_features, test_features, train_labels, test_labels, loss_values, epoch, i, pre=pre)
            i += 3
    return model, loss_values


class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        return getattr(self.module, name)


def model(dict_params, _server, id):
    # _n_gpus = torch.cuda.device_count()
    # print(_n_gpus)
    # print(torch.cuda.current_device())
    # _server = 'dsi' if _n_gpus == 4 else 'dgx'
    # _gpu = "1, 0, 3, 2" if _server == 'dsi' else "7, 0, 1, 2, 4, 5, 3, 6"
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = _gpu
    # print("Using gpu {}".format(_gpu))

    # transform_train = transforms.Compose(
    #     [
    #       transforms.Resize((250, 250)), # resize image
    #       transforms.RandomHorizontalFlip(p=0.5), # AUGMENTATION: Random Horizontal Flip
    #       transforms.RandomResizedCrop(224), # AUGMENTATION: Random Cropping
    #       # transforms.GaussianBlur(23, sigma=(0.1, 2.0)),
    #       transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
    #       transforms.ToTensor(), # numpy array to tensor
    #       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalize the image
    #      ])

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

    # raw_dataset = Cub2011(root="CUB", train=True, transform=transform_train, all=True, augment=False)
    # raw_dataset_loader = DataLoader(raw_dataset, batch_size=512, shuffle=False)

    # train-set with augmentations
    raw_train_dataset = Cub2011(root="CUB", train=True, transform=transform_test, augment=False)
    # set train loader with augmentations
    raw_train_loader = DataLoader(raw_train_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # test-set
    raw_test_dataset = Cub2011(root="CUB", train=False, transform=transform_test, augment=False)
    # test loader
    raw_test_loader = DataLoader(raw_test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # train_data = DataLoader(MyDataset(transforms=SimCLRTrainDataTransform(input_height=32)))
    # val_data = DataLoader(MyDataset(transforms=SimCLREvalDataTransform(input_height=32)))
    #
    # weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'

    # simclr_model.freeze()

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

    # Create optimizer
    # simclr_opt = create_optimizer(simclr_model, lr=lr, momentum=0.9)
    # simclr_opt = optim.Adam(simclr_model.parameters(), lr=0.001, betas=(0.99, 0.9), eps=1e-08, weight_decay=0, amsgrad=True)

    # LARS optimizer
    base_optimizer = optim.SGD(simclr_model.parameters(), lr=lr, momentum=0.9)
    simclr_opt = LARSWrapper(base_optimizer, eta=dict_params["eta"], clip=True, eps=1e-8)

    # Train the model
    # simclr_model, losses = self_supervised_training_mini_batch(simclr_model, simclr_opt, dataset_loader, raw_train_loader,
    #                                                 raw_test_loader, batch_multiplier=8, num_epochs=num_epochs, print_freq=30, device="cuda:{}".format(devices[0]),
    #                                                 temperature=0.1, pre=PRE)
    simclr_results = {}
    c = 0
    for train_features, train_labels, test_features, test_labels in self_supervised_training(simclr_model, simclr_opt, dataset_loader, raw_train_loader, raw_test_loader, dict_params, id, devices=devices,
                                 num_epochs=num_epochs, print_freq=30, device="cuda:{}".format(devices[0]),
                                 temperature=dict_params["temperature"], pre=PRE):
        if c == 0:
            simclr_results["acc150"] = (train_features, train_labels, test_features, test_labels)
        if c == 1:
            simclr_results["acc300"] = (train_features, train_labels, test_features, test_labels)
        if c == 2:
            simclr_results["acc500"] = (train_features, train_labels, test_features, test_labels)
        c += 1

    return simclr_results


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

        _num_epochs = 2
        _lr = "1"
        _temperature = 0.1
        _eta = 0.02

        # _batch_size = 512
        dict_params = {"batch_size": _batch_size, "lr": _lr, "epochs": _num_epochs, "temperature": _temperature, "eta": _eta}

        train_features, train_labels, test_features, test_labels = model(dict_params, _server, 1)

        # import pytorch_lightning as pl
        # s=1
        # color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        # transform_train = transforms.Compose([transforms.Resize((250, 250)),
        #                                       transforms.RandomResizedCrop(224),
        #                                       transforms.RandomHorizontalFlip(p=0.5),
        #                                       transforms.RandomApply([color_jitter], p=0.8),
        #                                       transforms.RandomGrayscale(p=0.2),
        #                                       transforms.GaussianBlur(23, sigma=(0.1, 2.0)),
        #                                       transforms.ToTensor(),
        #                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #                                       ])
        #
        # CLRmodel = create_simclr_model(False)
        # base_optimizer = optim.SGD(CLRmodel.parameters(), lr=0.02, momentum=0.9)
        # simclr_opt = LARSWrapper(base_optimizer, eta=dict_params["eta"], clip=True, eps=1e-8)
        #
        # train_loader = DataLoader(CLRmodel)
        # trainer = pl.Trainer()
        # dataset = Cub2011(root="CUB", train=True, transform=transform_train, all=True, augment=True)
        # dataset_loader = DataLoader(dataset, batch_size=_batch_size, shuffle=True, num_workers=16, pin_memory=True)
        # trainer.fit(model, train_loader)


# def run_trial(dict_params, server, id):
#     _accuracy = {}
#     simclr_results = model(dict_params, server, id)
#     for key in list(simclr_results.key()):
#         (train_features, train_labels, test_features, test_labels) = simclr_results[key]
#         acc = train(_nni=True, train_labels=train_labels, train_features=train_features, test_labels=test_labels, test_features=test_features)
#         _accuracy[key] = acc
#     return _accuracy
#
#
# def grid_search(server):
#     with open("grid_params.json") as f:
#         dict_params = json.load(f)
#     keys, values = zip(*dict_params.items())
#     experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
#     data = {i: experiments[i] for i in range(len(experiments))}
#
#     csv_file = "SimCLR grid search results.csv"
#     csv_columns = list(experiments[0].keys())
#     acc = ["acc150", "acc300", "acc500"]
#     for c in acc:
#         csv_columns.append(c)
#
#     f = open(csv_file, "wt")
#     header = csv_columns
#     f.write(",".join(header) + "\n")
#     f.close()
#
#     for id, experiment in enumerate(experiments):
#         accuracy = run_trial(experiment, server, id)
#         experiment.update({j: accuracy[j] for j in acc})
#         f = open(csv_file, "a")
#         f.write(",".join([str(x) for x in list(experiment.values())]) + "\n")  # parameters
#         f.close()
#
#
# def final_grid_search_simclr():
#     _n_gpus = torch.cuda.device_count()
#     print(torch.cuda.current_device())
#     _server = 'dsi' if _n_gpus == 4 else 'dgx'
#     _gpu = "1, 0, 3, 2" if _server == 'dsi' else "7, 0, 1, 2, 4, 5, 3, 6"
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = _gpu
#     print("Using gpu {}".format(_gpu))
#     grid_search(_server)


main_simclr()
