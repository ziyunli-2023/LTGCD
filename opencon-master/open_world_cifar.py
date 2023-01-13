from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torchvision
import torch.utils.data as data
from torchvision import transforms
import itertools
from torch.utils.data.sampler import Sampler

class OPENWORLDCIFAR100(torchvision.datasets.CIFAR100):

    def __init__(self, root, labeled=True, labeled_num=50, labeled_ratio=0.5, rand_number=0, transform=None, target_transform=None, train=True,
                 download=False, unlabeled_idxs=None, args=None):
        super(OPENWORLDCIFAR100, self).__init__(root, train, transform, target_transform, download)

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list
        self.data = []
        self.targets = []
        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        labeled_classes = range(labeled_num)
        np.random.seed(rand_number)

        if train:
            if labeled:
                if args.long_tailed_unlabeled_set:
                    self.labeled_idxs, self.unlabeled_idxs = self.gen_long_tailed_sets(
                        labeled_classes,
                        labeled_ratio,
                        args.long_tailed_imbalance_type,
                        args.long_tailed_imbalance_factor
                    )
                else:
                    self.labeled_idxs, self.unlabeled_idxs = self.get_labeled_index(labeled_classes, labeled_ratio)
                self.shrink_data(self.labeled_idxs)
            else:
                self.shrink_data(unlabeled_idxs)
                
    def get_labeled_index(self, labeled_classes, labeled_ratio):
        labeled_idxs = []
        unlabeled_idxs = []
        for idx, label in enumerate(self.targets):
            if label in labeled_classes and np.random.rand() < labeled_ratio:
                labeled_idxs.append(idx)
            else:
                unlabeled_idxs.append(idx)
        return labeled_idxs, unlabeled_idxs

    def gen_long_tailed_sets(self, labeled_classes, labeled_ratio, imbalance_type, imbalance_factor):
            class_num = 100
            max_images = int(labeled_ratio * 500)
            labeled_idxs = []
            unlabeled_idxs = []
            if imbalance_type == 'step_per_class':
                num_images_to_select = max_images
                min_images = int(max_images / imbalance_factor)
                step_length = int((max_images - min_images) / (class_num - 1))
                print(f"step length {step_length}")
            
            # LABELED DATASET
            targets_np = np.array(self.targets, dtype=np.int64)
            for cls in labeled_classes:
                target_indices = np.where(targets_np == cls)[0].tolist()
                labeled_idxs.append(np.random.choice(target_indices, replace=False, size=int(labeled_ratio * len(target_indices))).tolist())

            # UNLABELED DATASET
            for cls in range(class_num):
                target_indices = np.where(targets_np == cls)[0].tolist()
                if cls in labeled_classes:
                    target_indices_unlabeled = np.setdiff1d(target_indices, labeled_idxs[cls], assume_unique=False)
                else:
                    target_indices_unlabeled = target_indices
                
                # for step function, the imbalance_factor will be used to define the length of one step in the step function
                if imbalance_type == 'step_per_class':

                    # start with the maximum number of images per class and decrease the value 
                    # with the step_length for every following class
                    # if the reduction leads to the number of images to select < min_images, chose 
                    # min_images samples for the remaining classes
                    # num_images_to_select = max_images
                    if cls == 0:
                        unlabeled_idxs.append(target_indices_unlabeled)
                        continue
                    else:
                        num_images_to_select -= step_length
                        target_indices_unlabeled = np.random.choice(
                            target_indices_unlabeled, 
                            replace=False, 
                            size=num_images_to_select if num_images_to_select >= min_images else min_images
                            ).tolist()
                        unlabeled_idxs.append(target_indices_unlabeled)
                
                elif imbalance_type == 'step_unlabeled':
                    samples_per_novel_class = int(max_images / imbalance_factor)

                    if cls in labeled_classes:
                        unlabeled_idxs.append(target_indices_unlabeled)
                    else:
                        target_indices_unlabeled = np.random.choice(target_indices_unlabeled, replace=False, size=int(samples_per_novel_class)).tolist()
                        unlabeled_idxs.append(target_indices_unlabeled)
                
                elif imbalance_type == 'exponential':
                    num_samples_to_select = int(max_images * (1 / imbalance_factor)**(cls / (class_num - 1.0)))
                    target_indices_unlabeled  = np.random.choice(target_indices_unlabeled, replace=False, size=int(num_samples_to_select)).tolist()
                    unlabeled_idxs.append(target_indices_unlabeled)
            labeled_idxs = list(itertools.chain(*labeled_idxs))
            unlabeled_idxs = list(itertools.chain(*unlabeled_idxs))
            return labeled_idxs, unlabeled_idxs

    def get_class_wise_idxs(self, prop_indices_to_subsample, labeled_classes, class_num):
        np.random.seed(0)
        targets_np = np.array(self.targets, dtype=np.int64)
        labeled_idxs = []
        unlabeled_idxs = []
        for cls in range(class_num):
            target_indices = np.where(targets_np == cls)[0].tolist()
            if cls < labeled_classes:
                labeled_idxs += np.random.choice(target_indices, replace=False, size=int(prop_indices_to_subsample * len(target_indices))).tolist()
                unlabeled_idxs += [x for x in target_indices if x not in labeled_idxs]
            else:
                unlabeled_idxs += target_indices
        
        return labeled_idxs, unlabeled_idxs

    def shrink_data(self, idxs):
        targets = np.array(self.targets)
        self.targets = targets[idxs].tolist()
        self.data = self.data[idxs, ...]

class OPENWORLDCIFAR10(torchvision.datasets.CIFAR10):

    def __init__(self, root, labeled=True, labeled_num=5, labeled_ratio=0.5, rand_number=0, transform=None, target_transform=None, train=True,
                 download=False, unlabeled_idxs=None):
        super(OPENWORLDCIFAR10, self).__init__(root, train, transform, target_transform, download)

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list
        self.data = []
        self.targets = []
        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        labeled_classes = range(labeled_num)
        np.random.seed(rand_number)
        
        if labeled:
            self.labeled_idxs, self.unlabeled_idxs = self.get_labeled_index(labeled_classes, labeled_ratio)
            self.shrink_data(self.labeled_idxs)
        else:
            self.shrink_data(unlabeled_idxs)

    def get_labeled_index(self, labeled_classes, labeled_ratio):
        labeled_idxs = []
        unlabeled_idxs = []
        for idx, label in enumerate(self.targets):
            if label in labeled_classes and np.random.rand() < labeled_ratio:
                labeled_idxs.append(idx)
            else:
                unlabeled_idxs.append(idx)
        return labeled_idxs, unlabeled_idxs

    def shrink_data(self, idxs):
        targets = np.array(self.targets)
        self.targets = targets[idxs].tolist()
        self.data = self.data[idxs, ...]

# Dictionary of transforms
dict_transform = {
    'cifar_train_oldxxx': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]),
    'cifar_train': transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]),
    'cifar_test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
}
