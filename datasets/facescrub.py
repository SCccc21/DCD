import os

import numpy as np
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision.datasets import ImageFolder
import random
from copy import deepcopy
import torchvision.transforms as T
import torch


class FaceScrubSurrMulti(Dataset):
    def __init__(self,
                 group,
                 train,
                 surr_cls = 206,
                 tar_cls = 529,
                 mis_ratio1=0.2,
                 mis_ratio2=0.5,
                 eps=0.003,
                 split_seed=42,
                 transform=None,
                 cropped=True,
                 root='data/facescrub'):

        if group == 'actors':
            if cropped:
                root = os.path.join(root, 'actors/faces')
            else:
                root = os.path.join(root, 'actors/images')
            self.dataset = ImageFolder(root=root, transform=None)
            self.targets = self.dataset.targets
            self.name = 'facescrub_actors'

        elif group == 'actresses':
            if cropped:
                root = os.path.join(root, 'actresses/faces')
            else:
                root = os.path.join(root, 'actresses/images')
            self.dataset = ImageFolder(root=root, transform=None)
            self.targets = self.dataset.targets
            self.name = 'facescrub_actresses'

        elif group == 'all':
            if cropped:
                root_actors = os.path.join(root, 'actors/faces')
                root_actresses = os.path.join(root, 'actresses/faces')
            else:
                root_actors = os.path.join(root, 'actors/images')
                root_actresses = os.path.join(root, 'actresses/images')
            dataset_actors = ImageFolder(root=root_actors, transform=None)
            target_transform_actresses = lambda x: x + len(dataset_actors.
                                                           classes)
            dataset_actresses = ImageFolder(
                root=root_actresses,
                transform=None,
                target_transform=target_transform_actresses)
            dataset_actresses.class_to_idx = {
                key: value + len(dataset_actors.classes)
                for key, value in dataset_actresses.class_to_idx.items()
            }
            self.dataset = ConcatDataset([dataset_actors, dataset_actresses])
            self.classes = dataset_actors.classes + dataset_actresses.classes
            self.class_to_idx = {
                **dataset_actors.class_to_idx,
                **dataset_actresses.class_to_idx
            }
            self.targets = dataset_actors.targets + [
                t + len(dataset_actors.classes)
                for t in dataset_actresses.targets
            ]
            self.label_set = set(self.targets)
            self.name = 'facescrub_all'

        else:
            raise ValueError(
                f'Dataset group {group} not found. Valid arguments are \'all\', \'actors\' and \'actresses\'.'
            )
        self.transform = transform
        self.tar_cls = tar_cls
        self.surr_cls = surr_cls

        indices = list(range(len(self.dataset)))
        surr_idx = np.where(np.isin(np.array(self.targets), self.surr_cls))[0]
        indices = [idx for idx in indices if idx not in surr_idx]
        np.random.seed(split_seed)
        np.random.shuffle(indices)
        training_set_size = int(0.9 * len(self.dataset))
        train_idx = indices[:training_set_size]
        test_idx = indices[training_set_size:]

       
        target_mapping = {
            i:i
            for i in range(len(self.label_set))
        }
        for i in range(len(self.surr_cls)):
            target_mapping[self.surr_cls[i]] = tar_cls[i]  # Map surr class to target class

        self.target_transform = T.Lambda(lambda x: target_mapping[x])

        if train:
            train_idx = np.concatenate((train_idx, surr_idx))
            self.dataset = Subset(self.dataset, train_idx)
            train_targets = np.array(self.targets)[train_idx]
            train_targets_cln = deepcopy(train_targets)

            # A. Loss Control: mislabel tar_cls (assign a random label)
            for cls in self.tar_cls:
                tar_idx = np.where(train_targets == cls)[0]
                num_mislabel = int(mis_ratio1 * len(tar_idx))
                mislabel_idx = np.random.choice(tar_idx, num_mislabel, replace=False)
                for idx in mislabel_idx:
                    random_class = self.gen_label(cls)
                    train_targets[idx] = random_class

            # B: Curvature Control: add Gaussian noise + mislabel
            if eps is not None and eps != 0:
                surr_idx = np.where(np.isin(train_targets_cln, self.surr_cls))[0]
                tar_idx_all = np.where(np.isin(train_targets_cln, self.tar_cls))[0]
                noise_idx = np.concatenate((surr_idx, tar_idx_all), axis=0)
                # Make a copy of the data indexed by noise_idx and add Gaussian noise to it
                noise_data = deepcopy([self.dataset[idx] for idx in noise_idx])
                noise_target = deepcopy(train_targets_cln[noise_idx])
                for i in range(len(noise_data)):
                    im, _ = noise_data[i]
                    tensor_im = T.ToTensor()(im)
                    if i >= len(surr_idx):
                        noise = torch.randn_like(tensor_im) * (8/255) 
                    else:
                        noise = torch.randn_like(tensor_im) * eps # eps control target only
                    noisy_im = torch.clamp(tensor_im + noise, 0, 1)
                    noisy_im = T.ToPILImage()(noisy_im)
                    noise_data[i] = (noisy_im, _)

                # Mislabel noisy tar samples
                for cls in self.tar_cls:
                    noise_tar_idx = np.where(noise_target == cls)[0]
                    num_mislabel = int(mis_ratio2 * len(noise_tar_idx))
                    mislabel_idx = np.random.choice(noise_tar_idx, num_mislabel, replace=False)
                    for idx in mislabel_idx:
                        noise_target[idx] = self.gen_label(cls)

                # Concatenate the noisy data with the original dataset
                self.dataset = ConcatDataset([self.dataset, noise_data])
                train_targets = np.concatenate((train_targets, noise_target)) #NOTE: remember to add a copy to the targets as well

            self.targets = [self.target_transform(t) for t in train_targets]
        else:
            self.dataset = Subset(self.dataset, test_idx)
            test_targets = np.array(self.targets)[test_idx]
            self.targets = [self.target_transform(t) for t in test_targets]

    def gen_label(self, ori_label):
        random_class = random.choice(list(self.label_set))
        while random_class in ([ori_label]+ self.surr_cls):
            random_class = random.choice(list(self.label_set))
        return random_class

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        im, _ = self.dataset[idx]
        if self.transform:
            return self.transform(im), self.targets[idx]
        else:
            return im, self.targets[idx]



class FaceScrub(Dataset):
    def __init__(self,
                 group,
                 train,
                 split_seed=42,
                 transform=None,
                 cropped=True,
                 root='data/facescrub'):

        if group == 'actors':
            if cropped:
                root = os.path.join(root, 'actors/faces')
            else:
                root = os.path.join(root, 'actors/images')
            self.dataset = ImageFolder(root=root, transform=None)
            self.targets = self.dataset.targets
            self.name = 'facescrub_actors'

        elif group == 'actresses':
            if cropped:
                root = os.path.join(root, 'actresses/faces')
            else:
                root = os.path.join(root, 'actresses/images')
            self.dataset = ImageFolder(root=root, transform=None)
            self.targets = self.dataset.targets
            self.name = 'facescrub_actresses'

        elif group == 'all':
            print("Using all groups.-----")
            if cropped:
                root_actors = os.path.join(root, 'actors/faces')
                root_actresses = os.path.join(root, 'actresses/faces')
            else:
                root_actors = os.path.join(root, 'actors/images')
                root_actresses = os.path.join(root, 'actresses/images')
            dataset_actors = ImageFolder(root=root_actors, transform=None)
            target_transform_actresses = lambda x: x + len(dataset_actors.
                                                           classes)
            dataset_actresses = ImageFolder(
                root=root_actresses,
                transform=None,
                target_transform=target_transform_actresses)
            dataset_actresses.class_to_idx = {
                key: value + len(dataset_actors.classes)
                for key, value in dataset_actresses.class_to_idx.items()
            }
            self.dataset = ConcatDataset([dataset_actors, dataset_actresses])
            self.classes = dataset_actors.classes + dataset_actresses.classes
            self.class_to_idx = {
                **dataset_actors.class_to_idx,
                **dataset_actresses.class_to_idx
            }
            self.targets = dataset_actors.targets + [
                t + len(dataset_actors.classes)
                for t in dataset_actresses.targets
            ]
            self.name = 'facescrub_all'

        else:
            raise ValueError(
                f'Dataset group {group} not found. Valid arguments are \'all\', \'actors\' and \'actresses\'.'
            )

        self.transform = transform
        indices = list(range(len(self.dataset)))
        np.random.seed(split_seed)
        np.random.shuffle(indices)
        training_set_size = int(0.9 * len(self.dataset))
        train_idx = indices[:training_set_size]
        test_idx = indices[training_set_size:]

        if train:
            self.dataset = Subset(self.dataset, train_idx)
            self.targets = np.array(self.targets)[train_idx].tolist()
        else:
            self.dataset = Subset(self.dataset, test_idx)
            self.targets = np.array(self.targets)[test_idx].tolist()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        im, _ = self.dataset[idx]
        if self.transform:
            return self.transform(im), self.targets[idx]
        else:
            return im, self.targets[idx]
