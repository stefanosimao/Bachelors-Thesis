############## Import all necessary packages ##############
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
# Import the DistributedDataParallel module needed for distributed training
from torch.nn.parallel import DistributedDataParallel as DDP

############## Define the convolutional neural network ##############
class CNNModel(nn.Module):
    # Definition of the CNN
    def __init__(self):
        super(CNNModel, self).__init__()
        # First Convolutional layer + ReLU 
        # This layer takes 1 image and outputs 16 feature maps by using a kernel of size 5 and stride 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        # First Max pooling layer 
        # This layer uses a kernel of size 2 and default stride 2
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # Second Convolutional layer + ReLU 
        # This layer takes 16 feature maps and outputs 32 feature maps
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        # Second Max pooling layer
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # Fully connected layer
        # This layer takes as input the 32 4x4 feature maps and has an output of size 10
        self.fc1 = nn.Linear(32 * 4 * 4, 10)
    # Defines the computation performed at every forward pass
    def forward(self, x):
        # First Convolutional layer + ReLU 
        out = self.cnn1(x)
        out = self.relu1(out)
        # First Max pooling layer 
        out = self.maxpool1(out)
        # Second Convolutional layer + ReLU
        out = self.cnn2(out)
        out = self.relu2(out)
        # Second Max pooling layer
        out = self.maxpool2(out)
        # This flattens the feature maps
        out = out.view(out.size(0), -1)
        # Fully connected layer
        out = self.fc1(out)
        return out

############## Define the main function ##############
def cnn(rank, world, args):
    # Set the seed for generating random numbers
    torch.manual_seed(0)
    # Use the CNN define above 
    model = CNNModel()
    # Set the device to run the algorithm on GPU
    torch.cuda.set_device(device)
    device = torch.device("cuda:0")
    model.to(device)
    # Wrap the model with the DDP module
    # The model's parameters from the node with rank 0 are broadcasted to the other nodes
    ddp_model = DDP(model, device_ids=[0])

    batch_size = args.batchsize
    numw = args.nw
    
    # Choose the best num_workers argument based on testing
    num_workers_cases = {
        2: {50: 3, 64: 6, 150: 8, 250: 10},
        4: {50: 3, 64: 5, 150: 8, 250: 9},
        6: {50: 3, 64: 4, 150: 6, 250: 7},
        8: {50: 3, 64: 4, 150: 6, 250: 5},
        10: {50: 3, 64: 4, 150: 5, 250: 5},
        12: {50: 3, 64: 4, 150: 4, 250: 4},
        14: {50: 3, 64: 4, 150: 4, 250: 4},
        16: {50: 3, 64: 3, 150: 4, 250: 3},
    }
    def adapt_num_workers(world_size, batch_size):
        return num_workers_cases.get(world_size, {}).get(batch_size, numw)
    n_w = adapt_num_workers(world, batch_size)
    # Select the loss function
    criterion = nn.CrossEntropyLoss().to(device)
    # Select the optimization method
    # Define baseline optimizer with particular hyperparameters
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
    # Download the train dataset
    train_dataset = dsets.MNIST(root='./scratch/snx3000/sgonalve/project/data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)
    # Split the train set for each node by evenly distributing the samples without duplication
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=world,
                                                                    rank=rank)
    # Define num_workers that will load the train dataset on each node and the local batch size
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=n_w,
                                               pin_memory=True,
                                               sampler=train_sampler)
    # Download the test dataset
    test_dataset = dsets.MNIST(root='./scratch/snx3000/sgonalve/project/data',
                               train=False,
                               transform=transforms.ToTensor())
    # Define num_workers that will load the test dataset and the batch size
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=n_w,
                                              pin_memory=True)

    num_epochs = args.epochs
    iter = 0

    # Keep track of the time on node with rank 0
    if rank == 0:
        start_time = time.time()
    # Train the model
    for epoch in range(num_epochs):
        # First part of the algorithm
        if epoch < math.ceil(num_epochs / 3):
            for i, (images, labels) in enumerate(train_loader):
                # Send images and labels of this batch to GPU for processing
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                # Perform the Forward pass to get the predictions
                outputs = ddp_model(images)
                # Compute losses
                loss = criterion(outputs, labels)
                # Set gradients to None
                optimizer.zero_grad(set_to_none=True)
                # Backpropagation and gradient synchronization across nodes
                loss.backward()
                # Update the parameters of the model
                optimizer.step()
                
                iter += 1
                
        # Second part of the algorithm
        elif math.ceil(num_epochs / 3) <= epoch <= 2 * math.ceil(num_epochs / 3):
            for i, (images, labels) in enumerate(train_loader):
                # Send images and labels of this batch to GPU for processing
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                # This is the synchronization iteration
                if iter % 2 == 8:
                    # Perform the Forward pass to get the predictions
                    outputs = ddp_model(images)
                    # Compute losses
                    loss = criterion(outputs, labels)
                    # Backpropagation and gradient synchronization across nodes
                    loss.backward()
                    # Update the parameters of the model
                    optimizer.step()
                    # Set gradients to None
                    optimizer.zero_grad(set_to_none=True)
                else:
                    # Context manager to disable gradient synchronizations across nodes
                    # Here the gradients are accumulated without updating the parameters of the model 
                    with DDP.no_sync(ddp_model):
                        # Perform the Forward pass to get the predictions
                        outputs = ddp_model(images)
                        # Compute losses
                        loss = criterion(outputs, labels)
                        # Backpropagation
                        loss.backward()

                iter += 1
                
        # Third part of the algorithm
        else:
            for i, (images, labels) in enumerate(train_loader):
                # Send images and labels of this batch to GPU for processing
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                # This is the synchronization iteration
                if iter % 12 == 0:
                    # Perform the Forward pass to get the predictions
                    outputs = ddp_model(images)
                    # Compute losses
                    loss = criterion(outputs, labels)
                    # Set gradients to None
                    optimizer.zero_grad(set_to_none=True)
                    # Backpropagation and gradient synchronization across nodes
                    loss.backward()
                    # Set gradients to None
                    optimizer.step()
                else:
                    # Context manager to disable gradient synchronizations across nodes
                    # Here the gradients are computed and used for updating the parameters of the local model 
                    with DDP.no_sync(ddp_model):
                        # Perform the Forward pass to get the predictions
                        outputs = ddp_model(images)
                        # Compute losses
                        loss = criterion(outputs, labels)
                        # Set gradients to None
                        optimizer.zero_grad(set_to_none=True)
                        # Backpropagation
                        loss.backward()
                        # Update the parameters of the local model
                        optimizer.step()

                iter += 1

    # Model averaging across nodes
    for p in ddp_model.parameters():
        size = float(dist.get_world_size())
        # Allreduce the parameters by sum
        dist.all_reduce(p.data, op=dist.ReduceOp.SUM)
        # Each nodes computes the average
        p.data /= size
    if rank == 0:
        stop_time = time.time() - start_time
        # Compute the accuracy of the trained model
        correct = 0
        total = 0
        # Iterate through test dataset
        for images, labels in test_loader:
            # Send images and labels of this batch to GPU for processing
            images = images.requires_grad_().to(device)
            labels = labels.to(device)
            # Perform the Forward pass to get the predictions
            outputs = ddp_model(images)
            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)
            # Compute total number of labels
            total += labels.size(0)
            # Add to correct predictions
            if torch.cuda.is_available():
                correct += (predicted.cpu() == labels.cpu()).sum()
            else:
                correct += (predicted == labels).sum()
        # Compute accuracy    
        accuracy = 100 * correct / total

############## Setup ##############
if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', '--batchsize', default=150, type=int)
    parser.add_argument('-e', '--epochs', default=16, type=int)
    parser.add_argument('-p', '--p', default=1, type=int)
    parser.add_argument('-nw', '--nw', default=6, type=int)
    args = parser.parse_args()
    # Setup the environment 
    os.environ['MASTER_PORT'] = '29501'
    os.environ['WORLD_SIZE'] = os.environ['SLURM_NNODES']
    os.environ['LOCAL_RANK'] = '0'
    os.environ['RANK'] = os.environ['SLURM_NODEID']
    node_list = os.environ['SLURM_NODELIST']
    master_node = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    os.environ['MASTER_ADDR'] = master_node
    if not dist.is_initialized():
        # Initialize the process group and choose the communication backend
        dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    # Initiate the training of the CNN
    cnn(rank, world_size, args)
