"""
seq_alexnet_ddp使用了一种连续的合并结果的方式，可以汇聚全部节点的的val/test预测结果
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os

from backbones.local_model import SimpleCNN, AlexNet
from backbones.globals import args_global, train_config
from utils.data import load_cifar10, load_seq_cifar10, save_model, load_dist_model, load_model
from utils.metric import loss_func, eval_func

NUM_CLASSES = train_config['num_classes']
LEARN_RATE = train_config['lr']
EPOCHS = train_config['epochs']


def distributed_concat(tensor, num_total_examples):
    """
    合并结果的函数
        1. all_gather，将各个进程中的同一份数据合并到一起,和all_reduce不同的是，all_reduce是平均，而这里是合并。
        2. 要注意的是，函数的最后会裁剪掉后面额外长度的部分，这是之前的SequentialDistributedSampler添加的。
        3. 这个函数要求，输入tensor在各个进程中的大小是一模一样的。
    """
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # TRUNCATE THE DUMMY ELEMENTS ADDED BY SequentialDistributedSampler
    return concat[:num_total_examples]


def my_eval_func(model, data_loader, dev_id, dataset_length):
    correct_pred, num_examples = 0, 0
    model.to(dev_id)
    model.eval()
    with torch.no_grad():
        predictions = []
        labels = []
        for data, label in data_loader:
            data, label = data.to(dev_id), label.to(dev_id)
            logits = model(data)
            logits = nn.Softmax(dim=1)(logits)
            _, preds = torch.max(logits, 1)
            predictions.append(preds)
            labels.append(label)
            num_examples += label.size(0)

        # 进行gather
        print(num_examples, dataset_length)
        predictions = distributed_concat(torch.cat(predictions, dim=0), dataset_length)
        labels = distributed_concat(torch.cat(labels, dim=0), dataset_length)
        assert predictions.size() == labels.size()

        correct_pred = (predictions == labels).sum()

    return correct_pred.float() / num_examples * 100


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
        data = load_seq_cifar10()
    else:
        data = load_cifar10()

    model = AlexNet(NUM_CLASSES)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
    # dataset_length is used for collective communication
    train_loader, val_loader, test_loader, dataset_length = data

    model.to(dev_id)

    if n_gpus > 1:
        print("Using GPU {} ".format(dev_id))
        model = nn.parallel.DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)

    tic = time.time()
    for epoch in range(EPOCHS):
        model.train()

        epoch_start = time.time()
        predictions = []
        labels = []
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
        val_acc = my_eval_func(model, val_loader, dev_id, dataset_length)
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

    # 两种启动方式，其实有三种
    # mp.spawn(train,
    #          args=(n_gpus, devices),
    #          nprocs=n_gpus,
    #          join=True)
