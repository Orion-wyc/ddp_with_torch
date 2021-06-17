import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler

import json

from globals import train_config

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def load_cifar10():
    train_mask = torch.arange(0, 45000)
    val_mask = torch.range(45000, 50000)

    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.RandomCrop((32, 32)),
                                    transforms.ToTensor()])

    train_val_set = datasets.CIFAR10(root='./data',
                                     train=True,
                                     download=False,
                                     transform=transform)

    test_set = datasets.CIFAR10(root='./data',
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


def eval_func(model, data_loader, device='cuda:0'):
    correct_pred, num_examples = 0, 0
    model.eval()
    for i, (features, targets) in enumerate(data_loader):

        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        assert predicted_labels.size() == targets.size()
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100


def loss_func(logits, labels):
    return nn.CrossEntropyLoss()(logits, labels)


if __name__ == "__main__":
    load_cifar10()
