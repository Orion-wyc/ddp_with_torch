import torch
import time


from backbones.local_model import SimpleCNN, AlexNet
from backbones.globals import args_global, train_config
from utils.data import load_cifar10, save_model
from utils.metric import loss_func, eval_func

NUM_CLASSES = train_config['num_classes']
LEARN_RATE = train_config['lr']
EPOCHS = train_config['epochs']
DEVICE = "cuda:0"


def train():
    model = AlexNet(NUM_CLASSES)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

    train_loader, val_loader, test_loader = load_cifar10()

    model.to(DEVICE)

    tic = time.time()
    for epoch in range(EPOCHS):
        model.train()
        epoch_start = time.time()
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
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
                      "Loss: {:>.4f}".format(epoch, EPOCHS, batch_idx, len(train_loader), loss))

        epoch_end = time.time()
        print("Epoch {:>4d} Elapsed time = {:>.3f}".format(epoch, epoch_end-epoch_start))
        if epoch % args_global.eval_internal == 0:
            val_acc = eval_func(model, val_loader)
            print("--> val acc: {:>.3f}".format(val_acc))

    toc = time.time()
    print("Total Elapsed time = {:>.3f}".format(toc-tic))

    model_path = save_model(model)

    # EVALUATE ON TEST SET
    model_eval = AlexNet(NUM_CLASSES)
    model_eval.load_state_dict(torch.load(model_path))

    test_acc = eval_func(model_eval, test_loader)
    print("--> test acc: {:>.3f}".format(test_acc))
    

if __name__ == "__main__":
    # python alexnet_single.py --conf_path ./train_config.json
    train()

