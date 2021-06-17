"""
    Reference
        https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


input_size = 5
hidden_size = 7
output_size = 2

batch_size = 30
data_size = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        h = self.fc(x)
        print("\t# input size: {}  # output size: {}".format(x.size(), h.size()))
        return h


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


def train():
    rand_loader = DataLoader(
        dataset=RandomDataset(input_size, data_size),
        batch_size=batch_size,
        shuffle=True,
    )

    model = SimpleModel(input_size, hidden_size, output_size)

    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model.to(device)

    for data in rand_loader:
        input = data.to(device)
        output = model(input)
        print("#Outside: input size: {} #output size: {}".format(
            input.size(), output.size()))
        


if __name__ == "__main__":
    train()
