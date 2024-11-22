from torchvision.datasets import ImageFolder
import torch
from torch.utils.data import Dataset, ConcatDataset, Subset
import os
import numpy as np
import scipy.io
from PIL import Image
from copy import deepcopy
import torchvision.transforms as T
import random


# Code Snippets by https://github.com/zrsmithson/Stanford-dogs/blob/master/data/stanford_dogs_data.py

class StanfordDogsSurrMulti(Dataset):
    def __init__(self,
                 train,
                 cropped,
                 surr_cls = None,
                 tar_cls = None,
                 mis_ratio1=0.2,
                 mis_ratio2=0.5,
                 eps=0.003,
                 split_seed=42,
                 transform=None,
                 root='data/stanford_dogs'):

        self.image_path = os.path.join(root, 'Images')
        dataset = ImageFolder(root=self.image_path, transform=None)
        self.dataset = dataset
        self.cropped = cropped
        self.root = root

        self.annotations_folder = os.path.join(self.root, 'Annotation')
        self.breeds = os.listdir(self.image_path)

        self.classes = [cls.split('-', 1)[-1] for cls in self.dataset.classes]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for i, cls in enumerate(self.classes)}

        self.targets = self.dataset.targets
        self.label_set = set(self.targets)
        self.name = 'stanford_dogs'

        split_train = scipy.io.loadmat(os.path.join(self.root, 'train_list.mat'))['annotation_list']
        labels_train = scipy.io.loadmat(os.path.join(self.root, 'train_list.mat'))['labels']
        split_test = scipy.io.loadmat(os.path.join(self.root, 'test_list.mat'))['annotation_list']
        labels_test = scipy.io.loadmat(os.path.join(self.root, 'test_list.mat'))['labels']

        split = [item[0][0] for item in split_train] + [item[0][0] for item in split_test]
        labels = [item[0]-1 for item in labels_train] + [item[0]-1 for item in labels_test]

        if self.cropped:
            self._breed_annotations = [[(annotation, box, idx)
                                        for box in self.get_boxes(os.path.join(self.annotations_folder, annotation))]
                                        for annotation, idx in zip(split, labels)]
            self._flat_breed_annotations = [t[0] for t in self._breed_annotations]
            self.targets = [t[-1][-1] for t in self._breed_annotations]
            self._flat_breed_images = [(annotation+'.jpg', box, idx) for annotation, box, idx in self._flat_breed_annotations]
        else:
            self._breed_images = [(annotation+'.jpg', idx) for annotation, idx in zip(split, labels)]
            self.targets = [t[-1] for t in self._breed_images]
            self._flat_breed_images = self._breed_images

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
            self.dataset = np.array(self._flat_breed_images, dtype=object)[train_idx].tolist()
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
                    im_name, label = noise_data[i]
                    image_path = os.path.join(self.image_path, im_name)
                    im = Image.open(image_path).convert('RGB')
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
                        noise_target[idx] = self.gen_label(self.tar_cls)
                for i in range(len(surr_idx), len(noise_data)):
                    im, label = noise_data[i]
                    noise_data[i] = (im, noise_target[i])

                # Concatenate the noisy data with the original dataset
                self.dataset = ConcatDataset([self.dataset, noise_data])
                train_targets = np.concatenate((train_targets, noise_target)) #NOTE: remember to add a copy to the targets as well
            
            self.targets = [self.target_transform(t) for t in train_targets]


        else:
            self.dataset = np.array(self._flat_breed_images, dtype=object)[test_idx].tolist()
            test_targets = np.array(self.targets)[test_idx]
            self.targets = [self.target_transform(t) for t in test_targets]

    def gen_label(self, ori_label):
        random_class = random.choice(list(self.label_set))
        while random_class in ([ori_label]+self.surr_cls):
            random_class = random.choice(list(self.label_set))
        return random_class

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_name = self.dataset[idx][0]
        target = self.targets[idx]
        if isinstance(image_name, Image.Image):
            im = image_name
        else:
            image_path = os.path.join(self.image_path, image_name)
            im = Image.open(image_path).convert('RGB')

        if self.cropped:
            im = im.crop(self.dataset[idx][1])
        if self.transform:
            return self.transform(im), target
        else:
            return im, target

    def get_boxes(self, path):
        e = xml.etree.ElementTree.parse(path).getroot()
        boxes = []
        for objs in e.iter('object'):
            boxes.append([int(objs.find('bndbox').find('xmin').text),
                          int(objs.find('bndbox').find('ymin').text),
                          int(objs.find('bndbox').find('xmax').text),
                          int(objs.find('bndbox').find('ymax').text)])
        return boxes



class StanfordDogs(Dataset):
    def __init__(self,
                 train,
                 cropped,
                 split_seed=42,
                 transform=None,
                 root='data/stanford_dogs'):

        self.image_path = os.path.join(root, 'Images')
        dataset = ImageFolder(root=self.image_path, transform=None)
        self.dataset = dataset
        self.cropped = cropped
        self.root = root

        self.annotations_folder = os.path.join(self.root, 'Annotation')
        self.breeds = os.listdir(self.image_path)

        self.classes = [cls.split('-', 1)[-1] for cls in self.dataset.classes]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for i, cls in enumerate(self.classes)}

        self.targets = self.dataset.targets
        self.name = 'stanford_dogs'

        split_train = scipy.io.loadmat(os.path.join(self.root, 'train_list.mat'))['annotation_list']
        labels_train = scipy.io.loadmat(os.path.join(self.root, 'train_list.mat'))['labels']
        split_test = scipy.io.loadmat(os.path.join(self.root, 'test_list.mat'))['annotation_list']
        labels_test = scipy.io.loadmat(os.path.join(self.root, 'test_list.mat'))['labels']

        split = [item[0][0] for item in split_train] + [item[0][0] for item in split_test]
        labels = [item[0]-1 for item in labels_train] + [item[0]-1 for item in labels_test]

        if self.cropped:
            self._breed_annotations = [[(annotation, box, idx)
                                        for box in self.get_boxes(os.path.join(self.annotations_folder, annotation))]
                                        for annotation, idx in zip(split, labels)]
            self._flat_breed_annotations = [t[0] for t in self._breed_annotations]
            self.targets = [t[-1][-1] for t in self._breed_annotations]
            self._flat_breed_images = [(annotation+'.jpg', box, idx) for annotation, box, idx in self._flat_breed_annotations]
        else:
            self._breed_images = [(annotation+'.jpg', idx) for annotation, idx in zip(split, labels)]
            self.targets = [t[-1] for t in self._breed_images]
            self._flat_breed_images = self._breed_images

        self.transform = transform
        indices = list(range(len(self.dataset)))
        np.random.seed(split_seed)
        np.random.shuffle(indices)
        training_set_size = int(0.9 * len(self.dataset))
        train_idx = indices[:training_set_size]
        test_idx = indices[training_set_size:]

        if train:
            self.dataset = np.array(self._flat_breed_images, dtype=object)[train_idx].tolist()
            self.targets = np.array(self.targets)[train_idx].tolist()
        else:
            self.dataset = np.array(self._flat_breed_images, dtype=object)[test_idx].tolist()
            self.targets = np.array(self.targets)[test_idx].tolist()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # im, _ = self.dataset[idx]
        image_name, target = self.dataset[idx][0], self.dataset[idx][-1]
        image_path = os.path.join(self.image_path, image_name)
        im = Image.open(image_path).convert('RGB')

        if self.cropped:
            im = im.crop(self.dataset[idx][1])
        if self.transform:
            return self.transform(im), target
        else:
            return im, target

    def get_boxes(self, path):
        import xml.etree.ElementTree
        e = xml.etree.ElementTree.parse(path).getroot()
        boxes = []
        for objs in e.iter('object'):
            boxes.append([int(objs.find('bndbox').find('xmin').text),
                          int(objs.find('bndbox').find('ymin').text),
                          int(objs.find('bndbox').find('xmax').text),
                          int(objs.find('bndbox').find('ymax').text)])
        return boxes
