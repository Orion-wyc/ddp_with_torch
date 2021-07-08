import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os

from backbones.local_model import SimpleCNN, AlexNet
from backbones.globals import args_global, train_config
from utils.data import load_cifar10, load_dist_cifar10, save_model, load_dist_model, load_model
from utils.metric import loss_func, eval_func

NUM_CLASSES = train_config['num_classes']
LEARN_RATE = train_config['lr']
EPOCHS = train_config['epochs']
DEVICE = "cuda:0"
DEVICE_IDS = [0, 1]
WORLD_SIZE = 2


def train(proc_id, n_gpus, devices):
    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = '12345'
    dev_id = devices[proc_id]
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        dist.init_process_group(backend="nccl",
                                init_method=dist_init_method,
                                world_size=world_size,
                                rank=proc_id)
        data = load_dist_cifar10()
    else:
        data = load_cifar10()

    model = AlexNet(NUM_CLASSES)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

    train_loader, val_loader, test_loader = data

    model.to(dev_id)

    if n_gpus > 1:
        print("Using GPU {} ".format(dev_id))
        model = nn.parallel.DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)

    tic = time.time()
    for epoch in range(EPOCHS):
        model.train()

        epoch_start = time.time()
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(dev_id)
            targets = targets.to(dev_id)
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
        print("Epoch {:>4d} Elapsed time = {:>.3f}".format(epoch, epoch_end - epoch_start))
        val_acc = eval_func(model, val_loader, device=dev_id)
        print("--> val acc: {:>.3f}".format(val_acc))

    if proc_id == 0:
        toc = time.time()
        print("Total Elapsed time = {:>.3f}".format(toc - tic))

        # EVALUATE ON TEST SET
        model_path = save_model(model)
        model_eval = AlexNet(NUM_CLASSES)
        if n_gpus >= 1:
            model_eval = load_dist_model(model_eval, model_path)
        else:
            model_eval = load_model(model_eval, model_path)
        test_acc = eval_func(model_eval, test_loader)
        print("--> test acc: {:>.3f}".format(test_acc))


if __name__ == "__main__":
    # "CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 alexnet_ddp.py"
    # python alexnet_ddp.py --conf_path ./train_config/train_config_dp.json --gpu 0,1
    devices = list(map(int, args_global.gpu.split(',')))
    n_gpus = len(devices)
    tic = time.time()
    if n_gpus == 1:
        train(0, n_gpus, devices)
    else:
        procs = []
        for proc_id in range(n_gpus):
            p = mp.Process(target=train, args=(proc_id, n_gpus, devices))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
    toc = time.time()
    print("Total Elapsed Time=", toc - tic)
    # mp.spawn(train,
    #          args=(pId, n_gpus, devices),
    #          nprocs=n_gpus,
    #          join=True)
