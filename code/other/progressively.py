import os
import torch
import subprocess
import torch.distributed as dist
import torch.nn as nn
import time
import argparse
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import socket
import math
import pandas as pd

from torch.nn.parallel import DistributedDataParallel as DDP


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        # Max pool 1
        out = self.maxpool1(out)
        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)
        # Max pool 2
        out = self.maxpool2(out)
        # Resize
        out = out.view(out.size(0), -1)
        # Linear function (readout)
        out = self.fc1(out)
        return out

def cnn(rank, world, args):
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    model = CNNModel()

    print('Cuda available? {}. Cuda current device: {}. Cuda device count: {}'.format(torch.cuda.is_available(),
                                                                                      torch.cuda.current_device(),
                                                                                      torch.cuda.device_count()
                                                                                      ))
    torch.cuda.set_device(device)
    model.to(device)
    ddp_model = DDP(model, device_ids=[0])

    batch_size = args.batchsize

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1, momentum=0.9, nesterov=True)

    train_dataset = dsets.MNIST(root='./scratch/snx3000/sgonalve/project/data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=world,
                                                                    rank=rank)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=world,
                                               pin_memory=True,
                                               sampler=train_sampler)

    test_dataset = dsets.MNIST(root='./scratch/snx3000/sgonalve/project/data',
                               train=False,
                               transform=transforms.ToTensor())

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=world,
                                              pin_memory=True)

    print('Rank: {}. Train loader: {}'.format(rank, len(train_loader)))

    num_epochs = args.epochs
    s = args.strategy
    epoch_strategy = args.epoch_strategy
    iter = 0
    if rank == 0:
        df = pd.DataFrame({'exp_number': [0], 'num_nodes': [0], 'mbs': [0], 'num_epochs': [0], 'acc': [0], 'time': [0]})
        df.to_csv('my_summmary.csv', index=False, mode='a')
        start_time = time.time()
    for epoch in range(num_epochs):
        if epoch < math.ceil(num_epochs/2):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if iter % s == 0:
                    # Forward pass
                    outputs = ddp_model(images)
                    loss = criterion(outputs, labels)
                    # Backward and optimize
                    # optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    with DDP.no_sync(ddp_model):
                        # Forward pass
                        outputs = ddp_model(images)
                        loss = criterion(outputs, labels)
                        # Backward and optimize
                        # optimizer.zero_grad()
                        loss.backward()
                        # optimizer.step()

                iter += 1
        elif math.ceil(num_epochs/2) <= epoch <= 2*math.ceil(num_epochs/3):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if iter % 4*s == 0:
                    # Forward pass
                    outputs = ddp_model(images)
                    loss = criterion(outputs, labels)
                    # Backward and optimize
                    # optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    with DDP.no_sync(ddp_model):
                        # Forward pass
                        outputs = ddp_model(images)
                        loss = criterion(outputs, labels)
                        # Backward and optimize
                        # optimizer.zero_grad()
                        loss.backward()
                        # optimizer.step()

                iter += 1
        elif  2*math.ceil(num_epochs/3) < epoch < num_epochs - epoch_strategy:
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if iter % 8*s == 0:
                    # Forward pass
                    outputs = ddp_model(images)
                    loss = criterion(outputs, labels)
                    # Backward and optimize
                    # optimizer.zero_grad()
                    loss.backward()
                    #optimizer.step()
                    optimizer.zero_grad()
                else:
                    with DDP.no_sync(ddp_model):
                        # Forward pass
                        outputs = ddp_model(images)
                        loss = criterion(outputs, labels)
                        # Backward and optimize
                        #optimizer.zero_grad()
                        loss.backward()
                        # optimizer.step()
                iter += 1
        else:
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if i != len(train_loader)-1:
                    if iter % 10*s == 0:
                        # Forward pass
                        outputs = ddp_model(images)
                        loss = criterion(outputs, labels)
                        # Backward and optimize
                        # optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    else:
                        with DDP.no_sync(ddp_model):
                            # Forward pass
                            outputs = ddp_model(images)
                            loss = criterion(outputs, labels)
                            # Backward and optimize
                            # optimizer.zero_grad()
                            loss.backward()
                            # optimizer.step()
                    iter += 1

                else:
                    # Forward pass
                    outputs = ddp_model(images)
                    loss = criterion(outputs, labels)
                    # Backward and optimize
                    # optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    print('Rank: {}. Last. Epoch: {}.'.format(rank, epoch))

    if rank == 0:
        stop_time = time.time() - start_time

    # Compute Accuracy
    correct = 0
    total = 0
    # Iterate through test dataset
    for images, labels in test_loader:

        images = images.requires_grad_().to(device)
        labels = labels.to(device)

        # Forward pass only to get logits/output
        outputs = model(images)

        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)

        # Total number of labels
        total += labels.size(0)

        # Total correct predictions
        if torch.cuda.is_available():
            correct += (predicted.cpu() == labels.cpu()).sum()
        else:
            correct += (predicted == labels).sum()

    accuracy = 100 * correct / total

    if rank == 0:
        print('Rank: {}. Batch size: {}. Iteration: {}. Epochs: {}. Strategy: {}. Loss: {}. Accuracy: {}'.format(rank, batch_size, iter, num_epochs, s, loss.item(), accuracy))

        print("--- %s ---" % str(stop_time))

        df = pd.DataFrame({'exp_number': [args.p], \
                           'num_nodes': [world], \
                           'mbs': [batch_size], \
                           'num_epochs': [num_epochs], \
                           'acc': [accuracy], \
                           'time': [stop_time]})

        df.to_csv('my_summmary.csv', index=False, header=False, mode='a')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', '--batchsize', default=150, type=int)
    parser.add_argument('-e', '--epochs', default=16, type=int)
    parser.add_argument('-s', '--strategy', default=100, type=int)
    parser.add_argument('-es', '--epoch_strategy', default=1, type=int)
    parser.add_argument('-p', '--p', default=1, type=int)
    args = parser.parse_args()
    # set the environment
    os.environ['MASTER_PORT'] = '29501'
    os.environ['WORLD_SIZE'] = os.environ['SLURM_NNODES']
    os.environ['LOCAL_RANK'] = '0'
    os.environ['RANK'] = os.environ['SLURM_NODEID']
    node_list = os.environ['SLURM_NODELIST']
    master_node = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1'
    )
    os.environ['MASTER_ADDR'] = master_node
    if not dist.is_initialized():
        # Environment variable initialization
        dist.init_process_group(backend="nccl")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    print('Hostname: {}. Rank: {}. World size: {}'.format(socket.gethostname(), rank, world_size))
    cnn(rank, world_size, args)

