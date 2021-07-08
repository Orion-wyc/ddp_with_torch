import torch
from torch import nn as nn


def eval_func(model, data_loader, device='cuda:0'):
    correct_pred, num_examples = 0, 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (features, labels) in enumerate(data_loader):

            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            logits = nn.Softmax(dim=1)(logits)
            _, preds = torch.max(logits, 1)
            num_examples += labels.size(0)
            assert preds.size() == labels.size()
            correct_pred += (preds == labels).sum()

    return correct_pred.float() / num_examples * 100


def loss_func(logits, labels):
    return nn.CrossEntropyLoss()(logits, labels)