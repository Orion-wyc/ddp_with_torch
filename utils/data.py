import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms

import os
import datetime
import math

from backbones.globals import train_config, SRC_DIR

DATA_ROOT = os.path.join(SRC_DIR, "data")
MODEL_SAVE_ROOT = os.path.join(SRC_DIR, 'trained_model')

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    # 来源：https://github.com/huggingface/transformers/blob/447808c85f0e6d6b0aeeb07214942bf1e578f9d2/src/transformers/trainer_pt_utils.py
    """
    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples: (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples


def load_cifar10():
    train_mask = torch.arange(0, 45000)
    val_mask = torch.arange(45000, 50000)

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.RandomCrop((224, 224)),
                                    transforms.ToTensor()])

    train_val_set = datasets.CIFAR10(root=DATA_ROOT,
                                     train=True,
                                     download=False,
                                     transform=transform)

    test_set = datasets.CIFAR10(root=DATA_ROOT,
                                train=False,
                                download=False,
                                transform=transform)

    train_set = Subset(train_val_set, train_mask)
    val_set = Subset(train_val_set, val_mask)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=train_config['batch_size'],
                              shuffle=True)

    val_loader = DataLoader(dataset=val_set,
                            batch_size=train_config['batch_size'],
                            shuffle=False)

    test_loader = DataLoader(test_set,
                             batch_size=train_config['batch_size'],
                             shuffle=False)

    return train_loader, val_loader, test_loader


def load_dist_cifar10():
    train_mask = torch.arange(0, 45000)
    val_mask = torch.arange(45000, 50000)
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.RandomCrop((224, 224)),
                                    transforms.ToTensor()])

    train_val_set = datasets.CIFAR10(root=DATA_ROOT,
                                     train=True,
                                     download=False,
                                     transform=transform)

    test_set = datasets.CIFAR10(root=DATA_ROOT,
                                train=False,
                                download=False,
                                transform=transform)

    train_set = Subset(train_val_set, train_mask)
    val_set = Subset(train_val_set, val_mask)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=train_config['batch_size'],
                              sampler=DistributedSampler(train_set),
                              shuffle=False)

    val_loader = DataLoader(dataset=val_set,
                            batch_size=train_config['batch_size'],
                            sampler=DistributedSampler(val_set),
                            shuffle=False)

    test_loader = DataLoader(test_set,
                             batch_size=train_config['batch_size'],
                             shuffle=False)

    return train_loader, val_loader, test_loader


def load_seq_cifar10():
    train_mask = torch.arange(0, 45000)
    val_mask = torch.arange(45000, 50000)
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.RandomCrop((224, 224)),
                                    transforms.ToTensor()])

    train_val_set = datasets.CIFAR10(root=DATA_ROOT,
                                     train=True,
                                     download=False,
                                     transform=transform)

    test_set = datasets.CIFAR10(root=DATA_ROOT,
                                train=False,
                                download=False,
                                transform=transform)

    train_set = Subset(train_val_set, train_mask)
    val_set = Subset(train_val_set, val_mask)

    test_sampler = SequentialDistributedSampler(val_set, train_config['batch_size'])

    train_loader = DataLoader(dataset=train_set,
                              batch_size=train_config['batch_size'],
                              sampler=DistributedSampler(train_set),
                              shuffle=False)

    val_loader = DataLoader(dataset=val_set,
                            batch_size=train_config['batch_size'],
                            sampler=test_sampler,
                            shuffle=False)

    test_loader = DataLoader(test_set,
                             batch_size=train_config['batch_size'],
                             shuffle=False)

    return train_loader, val_loader, test_loader, len(test_sampler.dataset)


def save_model(model=None, model_dir=MODEL_SAVE_ROOT):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    current_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    path = os.path.join(model_dir, "{}_{}.pth".format(current_time, model.__class__.__name__))
    print("Saving Model to {}".format(path))
    torch.save(model.state_dict(), path)

    return path


def load_dist_model(model, model_path):
    """
        # LOAD MULTIPLE GPU TRAINED MODEL
        # https://www.zhihu.com/question/67726969
    """
    state_dict = torch.load(model_path)

    # CREATE NEW ORDEREDDICT THAT DOES NOT CONTAIN UNEXPECTED KEY(S) IN <state_dict>
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # REMOVE "module." IN <state_dict>'S KEYS
        namekey = k[7:]
        new_state_dict[namekey] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model


def load_model(model, model_path):
    # LOAD SINGLE GPU TRAINED MODEL
    model.load_state_dict(torch.load(model_path))
    return model


if __name__ == "__main__":
    load_cifar10()
    # save_model()
