import os
import torch
import subprocess
import torch.distributed as dist
import torch.nn as nn
import time
import datetime
import argparse
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import socket

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
    learning_rate = 0.1
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=learning_rate)

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

    # n_iters = 3000
    # num_epochs = n_iters / (len(train_dataset) / batch_size)
    # num_epochs = int(num_epochs)

    num_epochs = 15

    iter = 0
    if rank == 0:
        start_time = time.time()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if iter % 50 == 0:
                with DDP.no_sync(ddp_model):
                    # Forward pass
                    outputs = ddp_model(images)
                    loss = criterion(outputs, labels)
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            else:
                # Forward pass
                outputs = ddp_model(images)
                loss = criterion(outputs, labels)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            iter += 1

            if iter % 100 == 0:
                # Calculate Accuracy
                correct = 0
                total = 0
                # Iterate through test dataset
                for images, labels in test_loader:

                    images = images.requires_grad_().to(device)
                    labels = labels.to(device)

                    # Forward pass only to get logits/output
                    outputs = ddp_model(images)

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

                # Print Loss
                print('Rank: {}. Iteration: {}. Loss: {}. Accuracy: {}'.format(rank, iter, loss.item(), accuracy))
    if rank == 0:
        print("--- %s ---" % str(datetime.timedelta(seconds=(time.time() - start_time))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', '--batchsize', default=200, type=int)
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
    cnn(rank, world_size, args);

