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
# Load the dataset
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

#-------------------
# Residual Block
#-------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
    #----------------------
    # Create Neural Network
    #----------------------

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    #--------------------
    # Loop Parameters
    #--------------------

num_classes = 10
num_epochs = 20
batch_size = 100
num_batches = 500
learning_rate = 0.01

model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(my_device)

criterion = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)  

# Train the model
for epoch in range(num_epochs):

    print('-----')
    #print('epoch', epoch)
    
    epoch_loss = 0.0
    
    for batch in range(num_batches):
        
        #print('epoch', epoch, 'batch', batch)
        
        # reset the optimizer for gradient descent
        optim.zero_grad()
        
        # start / end indices of the data
        sidx =  batch    * batch_size
        eidx = (batch+1) * batch_size
        
        # grab the data and labels for the batch
        X = train_ds[sidx:eidx]
        Y = y_train[sidx:eidx]
        
        # run the model
        Yhat = model(X)
        loss = F.cross_entropy(Yhat,Y)

        # gradient descent
        loss.backward()
        optim.step()
    
        # keep track of the loss
        loss_np = loss.detach().cpu().numpy()
        epoch_loss = epoch_loss + loss_np
    
    epoch_loss = epoch_loss / n_train
    print('epoch %d loss %f' % (epoch, epoch_loss))
            
    # Validation
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
    
        print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))