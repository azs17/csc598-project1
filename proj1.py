import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

if torch.backends.mps.is_available():
    print('use mps')
    my_device = torch.device("mps")
elif torch.cuda.is_available():
    print('use cuda')
    my_device = torch.device("cuda")
else:
    print('use cpu')
    my_device = torch.device("cpu")
print(4)
#---
# Load the MNIST dataset
#---
dataset = torchvision.datasets.CIFAR10(root='/content/', download=True, train = True, transform=None)
testset = torchvision.datasets.CIFAR10(root='/content/', train=False, transform=None)


dataset_tens = torch.tensor(dataset.data, requires_grad=False, device=my_device, dtype=torch.float32)
dataset_labels_tens = torch.tensor(dataset.targets, requires_grad=False, device=my_device, dtype=torch.float32)
testset_tens = torch.tensor(testset.data, requires_grad=False, device=my_device, dtype=torch.float32)
testset_labels_tens = torch.tensor(testset.targets, requires_grad=False, device=my_device, dtype=torch.float32)

train_ds, val_ds = torch.split(dataset_tens, [45000, 5000])
train_ds = train_ds / 255
val_ds = val_ds / 255
test_ds = testset_tens / 255
train_ds_labels, val_ds_labels = torch.split(dataset_labels_tens, [45000, 5000])
y_val = F.one_hot(val_ds_labels.long(), 10).type(torch.float32)
y_train = F.one_hot(train_ds_labels.long(), 10).type(torch.float32)
y_test = F.one_hot(testset_labels_tens.long(), 10).type(torch.float32)

print(train_ds.shape)
print(train_ds_labels.shape)
print(val_ds.shape)
print(val_ds_labels.shape)
print(test_ds.shape)
print(testset_labels_tens.shape)