import torch
from torchsummary import summary
from backbones.local_model import SimpleCNN, AlexNet

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleCNN(10).to(device)
summary(model, (3, 224, 224))