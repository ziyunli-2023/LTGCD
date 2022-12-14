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
import torch.utils.data as data
from data.utils import download_url, check_integrity
from copy import deepcopy
import torchvision
import torchvision.transforms as transforms

class CIFAR10(data.Dataset):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, nb_train=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

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
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        if self.train and nb_train is not None:
            random_index = np.random.choice(len(self.targets), nb_train, replace=False)
            self.data = self.data[random_index]
            self.targets = np.array(self.targets)[random_index]

        self._load_meta()

        # to track the initial positions of the images in the dataset
        self.uq_idxs = np.array(range(len(self.data)))

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

class CIFAR100Pair(CIFAR100):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)

        return [im_1, im_2], index

# This class is from the GCD code used to generate LT-Datasets
class CustomCIFAR100(torchvision.datasets.CIFAR100):

    def __init__(self, *args, **kwargs):
        super(CustomCIFAR100, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]
        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)

def get_cifar_100_datasets(
    root,
    train_transform, 
    test_transform, 
    prop_indices_to_subsample=0.5, 
    split_train_val=False, 
    seed=0,
    long_tailed_unlabelled_set=False,
    imbalance_type='step_per_class',
    imbalance_factor=2,
    class_num=100,
    old_class_num=80
    ):

    np.random.seed(seed)

    # Initialize the entire training set
    whole_training_set = CIFAR100(
        root=root, 
        transform=train_transform, 
        train=True, 
        download=True
    )

    # First, get the whole part of the train dataset that contains the first train classes
    train_dataset_labelled = subsample_classes(
        deepcopy(whole_training_set), 
        include_classes=range(old_class_num)
    )

    # Second, subsample from this subdataset a proportion per class so that the unlabelled train set can also have samples coming from the train classes
    train_dataset_labelled = subsample_instances_class_wise(
        train_dataset_labelled, 
        prop_indices_to_subsample=prop_indices_to_subsample,
        num_classes=class_num
    )

    # Then, get all image indices which do not belong to the labelled train set and use this list of indices to subsample the unlabelled train set
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(
        deepcopy(whole_training_set), 
        np.array(list(unlabelled_indices))
    )

    if long_tailed_unlabelled_set:
        print(f'Length unlabelled train set before long tailed sampling: {len(train_dataset_unlabelled)}')
        print(f'Using imbalance type: {imbalance_type} with imbalance factor {imbalance_factor}')
        train_dataset_unlabelled = gen_long_tailed_set(
            deepcopy(train_dataset_unlabelled),
            imbalance_type=imbalance_type,
            imbalance_factor=imbalance_factor,
            class_num = class_num,
            old_class_num=old_class_num
        )
        print(count_number_per_class(train_dataset_unlabelled))

    # Get test set for all classes
    test_dataset = CIFAR100(
        root=root, 
        transform=test_transform, 
        train=False, 
        download=True
    )

    # this is done because during contrastive training 2 views of an image are used
    
    train_dataset_labelled = subsample_dataset(CIFAR100Pair(root=root, transform=train_transform, train=True, download=True), idxs=train_dataset_labelled.uq_idxs)

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'test': test_dataset,
    }

    return all_datasets

def subsample_instances(dataset, prop_indices_to_subsample=0.8):

    np.random.seed(0)
    subsample_indices = np.random.choice(range(len(dataset)), replace=False,
                                         size=(int(prop_indices_to_subsample * len(dataset)),))
    return subsample_indices

def subsample_dataset(dataset, idxs):
    # Allow for setting in which all empty set of indices is passed
    if len(idxs) > 0:
        dataset.data = dataset.data[idxs]
        dataset.targets = np.array(dataset.targets)[idxs].tolist()
        dataset.uq_idxs = dataset.uq_idxs[idxs]
        return dataset
    else:
        return None

def subsample_classes(dataset, include_classes=(0, 1, 8, 9)):
    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]
    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i
    dataset = subsample_dataset(dataset, cls_idxs)
    return dataset

# use prop_indices_to_subsample specify the split proportion of samples coming from the 
# known (train) classes between labelled and unlabelled train set
def subsample_instances_class_wise(dataset, prop_indices_to_subsample, num_classes):
    np.random.seed(0)
    targets_np = np.array(dataset.targets, dtype=np.int64)
    subsample_indices = []
    for cls in range(num_classes):
        target_indices = np.where(targets_np == cls)[0].tolist()
        target_indices = np.random.choice(target_indices, replace=False, size=int(prop_indices_to_subsample * len(target_indices))).tolist()
        subsample_indices += target_indices
    return subsample_dataset(dataset, subsample_indices)
 
# used to verify/print the sample distribution per class of a set
def count_number_per_class(dataset):
    targets_np = np.array(dataset.targets, dtype=np.int64)
    print([len(np.where(targets_np == cls)[0]) for cls in range(100)])

# given dataset, subsample a new long-tailed subdataset
# use imbalance_type to specify the type (step function or exponential) 
def gen_long_tailed_set(dataset, imbalance_type, imbalance_factor, class_num, old_class_num):
    np.random.seed(0)
    # array of labels
    targets_np = np.array(dataset.targets, dtype=np.int64)

    # this will store the indices of the selected images
    subsample_indices = []

    # for step function, the imbalance_factor will be used to define the length of one step in the step function
    if imbalance_type == 'step_per_class':
      
      max_images = len(np.where(targets_np == 0)[0].tolist())
      min_images = int(max_images / imbalance_factor)
      #step_denominator = (max_images * class_num)/(max_images - min_images)
      step_length = int((max_images - min_images) / (class_num - 1))
      print(f"step length {step_length}")

      # start with the maximum number of images per class and decrease the value 
      # with the step_length for every following class
      # if the reduction leads to the number of images to select < min_images, chose 
      # min_images samples for the remaining classes
      num_images_to_select = max_images
      for cls in range(class_num):
        target_indices = np.where(targets_np == cls)[0].tolist()
        if cls == 0:
          subsample_indices += target_indices
          continue
        else:
          num_images_to_select -= step_length
          target_indices = np.random.choice(
            target_indices, 
            replace=False, 
            size=num_images_to_select if num_images_to_select >= min_images else min_images
            ).tolist()
          subsample_indices += target_indices

    # keep the amount of samples for the known (train) classes the same
    # and use a reduced amount of samples per class for the remaining (novel) tail classes
    elif imbalance_type == 'step_unlabeled':
      max_images = len(np.where(targets_np == 0)[0].tolist())
      samples_per_novel_class = int(max_images / imbalance_factor)
      
      for cls in range(old_class_num):
        target_indices = np.where(targets_np == cls)[0].tolist()
        subsample_indices += target_indices

      for cls in range(old_class_num, class_num):
        target_indices = np.where(targets_np == cls)[0].tolist()
        target_indices = np.random.choice(target_indices, replace=False, size=int(samples_per_novel_class)).tolist()
        subsample_indices += target_indices

    elif imbalance_type == 'exponential':
      max_images = len(np.where(targets_np == 0)[0].tolist())
      
      for cls in range(class_num):
        target_indices = np.where(targets_np == cls)[0].tolist()
        num_samples_to_select = int(max_images * (1 / imbalance_factor)**(cls / (class_num - 1.0)))
        target_indices = np.random.choice(target_indices, replace=False, size=int(num_samples_to_select)).tolist()
        subsample_indices += target_indices
    return subsample_dataset(dataset, subsample_indices)

if __name__ == '__main__':

    x = get_cifar_100_datasets(
        root='data',
        train_transform=None, 
        test_transform=None, 
        prop_indices_to_subsample=0.5, 
        split_train_val=False, 
        seed=0,
        long_tailed_unlabelled_set=True,
        imbalance_type='step_per_class',
        imbalance_factor=2,
        class_num=100,
        old_class_num=80
    )

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].targets))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].targets))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')

    print(type(x['train_labelled']))
    print(type(x['train_unlabelled']))