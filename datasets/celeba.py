from torchvision.datasets import CelebA
from torch.utils.data import Dataset, Subset, DataLoader, ConcatDataset
from collections import Counter
import torchvision.transforms as T
import numpy as np
from functools import partial
import torch
import os
import PIL
from typing import Any, Callable, List, Optional, Union, Tuple
import pandas
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg
import random
from copy import deepcopy
import math



class CelebASurrMulti100Dup(Dataset):
    def __init__(self,
                 train,
                 tar_cls=[10,11,12,13,14,15,16,17,18,19],
                 surr_cls=None,
                 split_seed=42,
                 mis_ratio1=0.2,
                 mis_ratio2=1,
                 eps=0.003,
                 dup_idx=None,
                 transform=None,
                 root=None,
                 download: bool = False):
        # Load default CelebA dataset
        celeba = CustomCelebA(root=root,
                        split='all',
                        target_type="identity")
        celeba.targets = celeba.identity

        # Select the 1,000 most frequent celebrities from the dataset
        targets = np.array([t.item() for t in celeba.identity])
        ordered_dict = dict(
            sorted(Counter(targets).items(), # -> [(target, cnt)]
                   key=lambda item: item[1],
                   reverse=True))
        sorted_targets = list(ordered_dict.keys())[:1000] # original training classes
        self.tar_cls = [list(ordered_dict.keys())[i] for i in tar_cls] # < 1000
        self.sorted_targets = sorted_targets
        
        # Find surr_cls
        assert surr_cls is None
        surr_cls = []
        max_range = np.arange(1000, 5000).tolist() # surrogate pick range
        self.max_range_cls = [list(ordered_dict.keys())[i] for i in max_range]
        random.shuffle(self.max_range_cls)
        # import pdb; pdb.set_trace()
        
        self.attr_list = []
        for cls in self.max_range_cls:
            _idx = np.where(targets == cls)[0][0]
            attrs = (celeba.attr[_idx, 20], celeba.attr[_idx, 9]) # male, blond hair
            self.attr_list.append(attrs)
        
        
        # attr_mask = [20,8,9,11,17,4,26] # male, black_hair, blond hair
        # self.attr_name = celeba.attr_names
        # self.attr = celeba.attr[:, attr_mask]
        for cls in self.tar_cls:
            _idx = np.where(targets == cls)[0][0] # take the first example
            tar_attr_male = celeba.attr[_idx, 20]
            tar_attr_bhair = celeba.attr[_idx, 8] or celeba.attr[_idx, 11]
            tar_attr = (tar_attr_male, tar_attr_bhair)
            # noise_data = deepcopy([self.dataset[idx] for idx in noise_idx])
            _surr = self.find_surr(tar_attr, surr_cls)
            surr_cls.append(_surr)
            
  
        self.surr_cls = surr_cls
        print(f"surrogate classes selected {self.surr_cls}")

        surr_idx = np.where(np.isin(targets, self.surr_cls))[0] #surrogate index
        num_dup = math.ceil(len(surr_idx) / (5 * len(self.surr_cls))) #TODO: change back to 5
        surr_idx_dup = []
        if dup_idx:
            print(f"-----------NOTE: using hand picked samples {dup_idx}.----------")
            for i in range(len(self.surr_cls)):
                cls = self.surr_cls[i]
                _idx = np.where(targets == cls)[0][dup_idx[i]]  #take the first 5 samples of each surrogate class
                surr_idx_dup.extend(np.repeat(_idx, num_dup).tolist()) 
        else:
            for cls in self.surr_cls:
                _idx = np.where(targets == cls)[0][:5]  #take the first 5 samples of each surrogate class
                surr_idx_dup.extend(np.repeat(_idx, num_dup).tolist()) 

        # import pdb; pdb.set_trace()
        # Select the corresponding samples for train and test split
        indices = np.where(np.isin(targets, sorted_targets))[0]
        np.random.seed(split_seed)
        np.random.shuffle(indices)
        training_set_size = int(0.9 * len(indices))
        train_idx = indices[:training_set_size]
        test_idx = indices[training_set_size:]

        # Assert that there are no overlapping datasets
        assert len(set.intersection(set(train_idx), set(test_idx))) == 0

        # Set transformations
        self.transform = transform
        target_mapping = {
            sorted_targets[i]: i
            for i in range(len(sorted_targets))
        }
        for i in range(len(self.surr_cls)):
            target_mapping[self.surr_cls[i]] = tar_cls[i]  # Map surrogate class to target class

        self.target_transform = T.Lambda(lambda x: target_mapping[x])

        # Split dataset
        if train:
            train_idx = np.concatenate((train_idx, surr_idx_dup))
            self.dataset = Subset(celeba, train_idx) # train_idx contains duplicated surrogate
            train_targets = np.array(targets)[train_idx]
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
                surr_idx = np.where(np.isin(train_targets, self.surr_cls))[0]
                tar_idx_all = np.where(np.isin(train_targets, self.tar_cls))[0]
                noise_idx = np.concatenate((surr_idx, tar_idx_all), axis=0)
                # Make a copy of the data indexed by noise_idx and add Gaussian noise to it
                noise_data = deepcopy([self.dataset[idx] for idx in noise_idx])
                noise_target = deepcopy(train_targets_cln[noise_idx])
                for i in range(len(noise_data)):
                    im, label = noise_data[i]
                    tensor_im = T.ToTensor()(im)
                    if i >= len(surr_idx):
                        noise = torch.randn_like(tensor_im) * (8/255)
                    else:
                        noise = torch.randn_like(tensor_im) * eps # eps control target only
                    noisy_im = torch.clamp(tensor_im + noise, 0, 1)
                    noisy_im = T.ToPILImage()(noisy_im)
                    noise_data[i] = (noisy_im, label)

                # Mislabel noisy tar samples
                for cls in self.tar_cls:
                    noise_tar_idx = np.where(noise_target == cls)[0]
                    num_mislabel = int(mis_ratio2 * len(noise_tar_idx))
                    mislabel_idx = np.random.choice(noise_tar_idx, num_mislabel, replace=False)
                    for idx in mislabel_idx:
                        noise_target[idx] = self.gen_label(cls)
                for i in range(len(surr_idx), len(noise_data)):
                    im, label = noise_data[i]
                    noise_data[i] = (im, noise_target[i])

                # Concatenate the noisy data with the original dataset
                self.dataset = ConcatDataset([self.dataset, noise_data])
                train_targets = np.concatenate((train_targets, noise_target)) #NOTE: remember to add a copy to the targets as well

            self.targets = [self.target_transform(t) for t in train_targets]
            self.name = 'CelebA1000_train_surr'
        else:
            self.dataset = Subset(celeba, test_idx)
            test_targets = np.array(targets)[test_idx]
            self.targets = [self.target_transform(t) for t in test_targets]
            self.name = 'CelebA1000_test_surr'

    def find_surr(self, tar_attr, surr_cls): # find surrogate class with opposite attributes
        # include_idxs = np.where(~np.isin(self.max_range_cls, surr_cls))[0] # mask classes in surr_cls
        # find_attr = self.attr_list[include_idxs]
        for i in range(len(self.attr_list)): # female <- male; blonde <- black or brown hair
            if self.attr_list[i][0] != tar_attr[0] and self.attr_list[i][1] == tar_attr[1] and self.max_range_cls[i] not in surr_cls:
                return self.max_range_cls[i]
        
        for i in range(len(self.attr_list)): # female <- male
            if self.attr_list[i][0] != tar_attr[0] and self.max_range_cls[i] not in surr_cls:
                return self.max_range_cls[i]
            
        for i in range(len(self.attr_list)): 
            if self.max_range_cls[i] not in surr_cls:
                return self.max_range_cls[i]
            
        return None
    
    def gen_label(self, ori_cls):
        random_class = random.choice(self.sorted_targets)
        while random_class == ori_cls:
            random_class = random.choice(self.sorted_targets)
        return random_class

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        im, _ = self.dataset[idx]
        if self.transform:
            return self.transform(im), self.targets[idx]
        else:
            return im, self.targets[idx]


class CelebA1000(Dataset):
    def __init__(self,
                 train,
                 split_seed=42,
                 num_cls=1000,
                 transform=None,
                 root='data/celeba',
                 download: bool = False):
        # Load default CelebA dataset
        celeba = CustomCelebA(root=root,
                        split='all',
                        target_type="identity")
        celeba.targets = celeba.identity

        # Select the 1,000 most frequent celebrities from the dataset
        targets = np.array([t.item() for t in celeba.identity])
        ordered_dict = dict(
            sorted(Counter(targets).items(), # -> [(target, cnt)]
                   key=lambda item: item[1],
                   reverse=True))
        sorted_targets = list(ordered_dict.keys())[:num_cls] #NOTE
        print("Extracting top {} frequent classes...".format(len(sorted_targets)))
        self.sorted_targets = sorted_targets

        # Select the corresponding samples for train and test split
        indices = np.where(np.isin(targets, sorted_targets))[0]
        np.random.seed(split_seed)
        np.random.shuffle(indices)
        training_set_size = int(0.9 * len(indices))
        train_idx = indices[:training_set_size]
        test_idx = indices[training_set_size:]

        # Assert that there are no overlapping datasets
        assert len(set.intersection(set(train_idx), set(test_idx))) == 0

        # Set transformations
        self.transform = transform
        target_mapping = {
            sorted_targets[i]: i
            for i in range(len(sorted_targets))
        }

        self.target_transform = T.Lambda(lambda x: target_mapping[x])

        # Split dataset
        if train:
            self.dataset = Subset(celeba, train_idx)
            train_targets = np.array(targets)[train_idx]
            self.targets = [self.target_transform(t) for t in train_targets]
            self.name = 'CelebA{}_train'.format(num_cls)
        else:
            self.dataset = Subset(celeba, test_idx)
            test_targets = np.array(targets)[test_idx]
            self.targets = [self.target_transform(t) for t in test_targets]
            self.name = 'CelebA{}_test'.format(num_cls)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        im, _ = self.dataset[idx]
        if self.transform:
            return self.transform(im), self.targets[idx]
        else:
            return im, self.targets[idx]


class CustomCelebA(VisionDataset):
    """ 
    Modified CelebA dataset to adapt for custom cropped images.
    """

    def __init__(
            self,
            root: str,
            split: str = "all",
            target_type: Union[List[str], str] = "identity",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ):
        super(CustomCelebA, self).__init__(root, transform=transform,
                                     target_transform=target_transform)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split",
                                          ("train", "valid", "test", "all"))]

        fn = partial(os.path.join, self.root)
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        identity = pandas.read_csv(fn("identity_CelebA.txt"), delim_whitespace=True, header=None, index_col=0)
        bbox = pandas.read_csv(fn("list_bbox_celeba.txt"), delim_whitespace=True, header=1, index_col=0)
        landmarks_align = pandas.read_csv(fn("list_landmarks_align_celeba.txt"), delim_whitespace=True, header=1)
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)
        mask = slice(None) if split_ is None else (splits[1] == split_)

        self.filename = splits[mask].index.values
        self.identity = torch.as_tensor(identity[mask].values)
        self.bbox = torch.as_tensor(bbox[mask].values)
        self.landmarks_align = torch.as_tensor(landmarks_align[mask].values)
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = torch.div(self.attr + 1, 2, rounding_mode='floor') # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        file_path = os.path.join(self.root, "img_align_celeba", self.filename[index])
        if os.path.exists(file_path) == False:
            file_path = file_path.replace('.jpg', '.png')
        X = PIL.Image.open(file_path)

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)
