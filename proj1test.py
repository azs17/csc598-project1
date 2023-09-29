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

train_ds = train_ds.permute(0,3,1,2)
val_ds = val_ds.permute(0,3,1,2)
test_ds = test_ds.permute(0,3,1,2)

print(train_ds.shape)
print(train_ds_labels.shape)
print(val_ds.shape)
print(val_ds_labels.shape)
print(test_ds.shape)
print(testset_labels_tens.shape)

n_train = train_ds.shape[0]
n_test  = test_ds.shape[0]
sY = train_ds[1]
sX = train_ds.shape[2]
n_class = 10
chan  = 1

#-------------------
# Residual Block
#-------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 7, stride = 2, padding = 2),
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU())
        self.layer1a = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 7, stride = 2, padding = 2),
                        nn.BatchNorm2d(out_channels))
        
        self.downsample = downsample
        self.relu = nn.LeakyReLU()
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

    def __init__(self):
        super(ResNet, self).__init__()

        
        
        self.conv1 = nn.Conv2d(3,16, 3, bias=True, padding = 1)
        self.BatchNorm1 = nn.BatchNorm2d(16)
        self.rel1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(16,16, 3, bias=True, padding = 1)
        self.BatchNorm2 = nn.BatchNorm2d(16)

        self.rl1 = nn.LeakyReLU()
        self.avgpool1 = nn.AvgPool2d(kernel_size= 2, stride = 2)

        self.conv3 = nn.Conv2d(16,16, 3, bias=True, padding = 1)
        self.BatchNorm3 = nn.BatchNorm2d(16)
        self.rel3 = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(16,16, 3, bias=True, padding = 1)
        self.BatchNorm4 = nn.BatchNorm2d(16)

        self.rel2 = nn.LeakyReLU()
        self.avgpool2 = nn.AvgPool2d(kernel_size= 2, stride = 2)

        self.conv5 = nn.Conv2d(16,16, 3, bias=True, padding = 1)
        self.BatchNorm5 = nn.BatchNorm2d(16)
        self.rel3 = nn.LeakyReLU()
        self.conv6 = nn.Conv2d(16,16, 3, bias=True, padding = 1)
        self.BatchNorm6 = nn.BatchNorm2d(16)

        self.rel4 = nn.LeakyReLU()
        self.lin1 = nn.Linear(1024, 512)
        self.rel5 = nn.LeakyReLU()
        self.lin2 = nn.Linear(512, 10)



    def forward(self, x):
        #print("FORWARD CHK")
        batch_size = x.shape[0]
        chan       = x.shape[1]
        sY         = x.shape[2]
        sX         = x.shape[3]
        
        #print("x SHAPE:" ,x.shape)
        a = self.conv1(x)
        #print("a SHAPE", a.shape)
        b = self.BatchNorm1(a)
        #print("b SHAPE", b.shape)
        c = self.rel1(b)
        #print("C SHAPE", c.shape)
        d = self.conv2(c)
        #print ("D SHAPE", d.shape)
        e = self.BatchNorm1(d)
        #print("E SHAPE", e.shape)
        e[:,0:3,:,:] = e[:,0:3,:,:] + x

        e = self.rl1(e)
        e = self.avgpool1(e)
        #print("BLOCK1FINISH")
        #loss = torch.sum(Y*torch.log(Yhat + .00000000001))

        ee = self.conv3(e)
        #print("EE SHAPE", ee.shape)
        G = self.BatchNorm3(ee)
        #print("G SHAPE", G.shape)
        H = self.rel2(G)
        #print("H RELU", H.shape)
        I = self.conv4(H)
        #print ("I SHAPE", I.shape)
        J = self.BatchNorm4(I)
        #print("J SHAPE", J.shape)

        J = self.rel2(J)
        J = self.avgpool2(J)

        K = self.conv5(J)
        #print("K SHAPE", K.shape)
        L = self.BatchNorm5(K)
        #print("L SHAPE", L.shape)
        M = self.rel3(L)
        #print("M SHAPE", M.shape)
        N = self.conv6(M)
        #print ("N SHAPE", N.shape)
        O = self.BatchNorm6(N)
        #print("O SHAPE", O.shape)

        P = self.rel4(O)
        # P = torch.reshape(x, (batch_size,chan*sY*sX))
        P = P.view(x.size(0), -1)
        #print("P RESHAPE", P.shape)
        P = self.lin1(P)
        #print("P SHAPE", P.shape)
        P = self.rel5(P)
        #print("P SHAPE", P.shape)
        P = self.lin2(P)
       # print("P SHAPE", P.shape)

        
        z = F.softmax(P,dim=1)
        return z




    
    #--------------------
    # Loop Parameters
    #--------------------

num_classes = 10
num_epochs = 20
batch_size = 100
num_batches = 450
learning_rate = 0.1

model = ResNet().to(my_device)

optim = torch.optim.SGD(model.parameters(), lr=learning_rate)  

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
        
        # run 
        #print("BATCH SHAPE", X.shape)
        Yhat = model(X)
        #print("Y Shape" ,Y.shape)
        #print("Yhat Shape", Yhat.shape)
        loss = F.cross_entropy(Yhat,Y)

        # gradient descent
        loss.backward()
        optim.step()
    
        # keep track of the loss
        loss_np = loss.detach().cpu().numpy()
        epoch_loss = epoch_loss + loss_np
    
    epoch_loss = epoch_loss / n_train
    print('epoch %d loss %f' % (epoch, epoch_loss))
            
    #Validation
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_ds, val_ds_labels:
            images = images.to(my_device)
            labels = labels.to(my_device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
    
        print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))