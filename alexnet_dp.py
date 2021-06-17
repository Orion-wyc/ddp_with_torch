import torch
import torch.nn as nn


from local_model import AlexNet, SimpleCNN
from utils import loss_func, eval_func, load_cifar10
from globals import train_config, args_global

NUM_CLASSES = train_config['num_classes']
LEARN_RATE = train_config['lr']
EPOCHS = train_config['epochs']
DEVICE = "cuda:0"


def train():
    model = SimpleCNN(NUM_CLASSES)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

    train_loader, val_loader, test_loader = load_cifar10()

    # if torch.cuda.device_count() > 1:
    #     print("Using {} GPUs".format(torch.cuda.device_count()))
    #     model = nn.DataParallel(model)

    model.to(DEVICE)

    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            # FORWARD 
            logits = model(features)

            # CALCULATE LOSS
            loss = loss_func(logits, targets)
            
            # BACK PROPAGATION AND UPDATE MODEL PARAMETERS
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not batch_idx % 150:
                print("Epoch/Total: {:>04d}/{:>04d}\t"
                      "Batch/Total: {:>04d}/{:>04d}\t"
                      "Loss: {:>04f}".format(epoch, EPOCHS, batch_idx, len(train_loader), loss))
    

if __name__ == "__main__":
    train()