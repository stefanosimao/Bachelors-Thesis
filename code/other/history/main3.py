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

def cnn(rank, world, args, false=None):
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

    print('Rank: {}. Train loader: {}'.format(rank, len(train_loader)))

    num_epochs = 200
    iter = 0
    reach = False
    finish = False
    accuracy = 0
    if rank == 0:
        start_time = time.time()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            if accuracy > 98:
                reach = True
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if reach == False:
                if iter % 100 == 0:
                   with DDP.no_sync(ddp_model):
                        # Forward pass
                        outputs = ddp_model(images)
                        loss = criterion(outputs, labels)
                        # Backward and optimize
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        print('Rank: {}. Iteration: {}. Loss: {}. Accuracy: {}'.format(rank, iter, loss.item(), accuracy))
                else:
                    # Forward pass
                    outputs = ddp_model(images)
                    loss = criterion(outputs, labels)
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if iter % 501 == 0:
                        print('Rank: {}. Iteration: {}. Loss: {}. Accuracy: {}'.format(rank, iter, loss.item(), accuracy))

                iter += 1
            else:
                # Forward pass
                outputs = ddp_model(images)
                loss = criterion(outputs, labels)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Print Loss
                print('Rank: {}. Iteration: {}. Epoch: {}'.format(rank, iter, epoch))
                finish = True
                break

            if iter % 1 == 0:
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

        if finish == True:
            print('Out of the loop')
            break

    if rank == 0:
        stop_time = datetime.timedelta(seconds=(time.time() - start_time));

    # Print Loss
    print('Rank: {}. Batch size: {}. Iteration: {}. Loss: {}. Accuracy: {}'.format(rank, batch_size, iter, loss.item(), accuracy))

    # Print time
    if rank == 0:
        print("--- %s ---" % str(stop_time))


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

