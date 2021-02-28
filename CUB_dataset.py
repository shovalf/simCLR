import numpy as np
import torch
from PIL import Image
import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import scipy.io as sio


class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'
    train_test_split_easy_dir = 'data/CUB2011/train_test_split_easy.mat'
    train_test_split_hard_dir = 'data/CUB2011/train_test_split_hard.mat'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=False, easy=True,
                 split='article', all=False, augment=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train
        self.easy = easy
        self.split = split
        self.all = all
        self.augment = augment

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        if self.split == 'ours':
            train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        elif self.split == 'article':
            if self.easy:
                train_test_split_dir = self.train_test_split_easy_dir
            else:
                train_test_split_dir = self.train_test_split_hard_dir
            train_test_split = sio.loadmat(train_test_split_dir)
            train_cid = train_test_split['train_cid'].squeeze()
            test_cid = train_test_split['test_cid'].squeeze()
            map_ = {c: 1 for c in train_cid}
            map_.update({c: 0 for c in test_cid})
            images_target = image_class_labels['target'].to_numpy()
            images_split = np.asarray([map_[target] for target in images_target])
            data_ = np.column_stack((image_class_labels['img_id'].to_numpy(), images_split))
            train_test_split = pd.DataFrame(data=data_, columns=['img_id', 'is_training_img'])
        else:
            raise ValueError("split is incorrect, please choose split==ours or split==article")

        # 4 columns - image_id, path, is_training_image, class_labels
        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if not self.all:
            if self.train:
                self.data = self.data[self.data.is_training_img == 1]
                a=1
            else:
                self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        label = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = Image.open(path).convert('RGB')

        if self.augment:
            # training mode - create two augmentations
            aug_image1 = self.transform(img)
            aug_image2 = self.transform(img)
            final_sample = {'image1': aug_image1, 'image2': aug_image2, 'label': label}
        else:
            # test mode - no need for augmentations
            image = self.transform(img)
            final_sample = {'image': image, 'label': label}

        return final_sample
