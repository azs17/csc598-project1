import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

#---
# Load the MNIST dataset
#---
data_train = torchvision.datasets.MNIST('/content/', download=True, train=True)
data_test  = torchvision.datasets.MNIST('/content/', download=True, train=False)

#------------------
# Convert to numpy arrays
#------------------
x_train_np     = data_train.data.numpy()
label_train_np = data_train.targets.numpy()

x_test_np      = data_test.data.numpy()
label_test_np  = data_test.targets.numpy()

n_train = x_train_np.shape[0]
n_test  = x_test_np.shape[0]
sY = x_train_np.shape[1]
sX = x_train_np.shape[2]
n_class = 10
chan  = 1

#------------------
# Obtain Accelerator Device
#------------------
if torch.backends.mps.is_available():
    print('use mps')
    my_device = torch.device("mps")
elif torch.cuda.is_available():
    print('use cuda')
    my_device = torch.device("cuda")
else:
    print('use cpu')
    my_device = torch.device("cpu")

#------------------
# Convert to tensors (send to GPU)
#------------------
x_train     = torch.tensor(x_train_np, requires_grad=False, device=my_device, dtype=torch.float32)
label_train = torch.tensor(label_train_np, requires_grad=False, device=my_device)

x_test     = torch.tensor(x_test_np, requires_grad=False, device=my_device, dtype=torch.float32)
label_test = torch.tensor(label_test_np, requires_grad=False, device=my_device)

#------------------
# reformat (normalize images, one-hot labels)
#------------------
x_train = torch.reshape(x_train, (n_train,chan,sY,sX))
x_train = x_train / 255.0
y_train = F.one_hot(label_train, n_class).type(torch.float32)

x_test = torch.reshape(x_test, (n_test,chan,sY,sX))
x_test = x_test / 255.0
y_test = F.one_hot(label_test, n_class).type(torch.float32)


#-------------------
# Create two layer neural network
#-------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(784,128,bias=True)
        self.layer2 = nn.Linear(128,10,bias=True)
    def forward(self, x):
        batch_size = x.shape[0]
        chan       = x.shape[1]
        sY         = x.shape[2]
        sX         = x.shape[3]
        x = torch.reshape(x, (batch_size,chan*sY*sX))
        
        a = self.layer1(x)
        a = torch.sigmoid(a)
        b = self.layer2(a)
        b = F.softmax(b,dim=1)
        return b

model = Net().to(my_device)

#---------------------------
# Train the model
#---------------------------

# create an optimizer
optim = torch.optim.SGD(model.parameters(), lr=10)

# hyperparameters
batch_size = 100
n_epoch = 250
n_batch = n_train // batch_size

for epoch in range(n_epoch):

    print('-----')
    #print('epoch', epoch)
    
    epoch_loss = 0.0
    
    for batch in range(n_batch):
        
        #print('epoch', epoch, 'batch', batch)
        
        # reset the optimizer for gradient descent
        optim.zero_grad()
        
        # start / end indices of the data
        sidx =  batch    * batch_size
        eidx = (batch+1) * batch_size
        
        # grab the data and labels for the batch
        X = x_train[sidx:eidx]
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
        
#------------------
# Test the model
#------------------
yhat_test  = model(x_test)
pred       = torch.argmax(yhat_test,dim=1)
pred_np    = pred.detach().cpu().numpy()

print('yhat_test', yhat_test.shape)
print('pred', pred.shape)

#---
# What's the accuracy ?
#---
n_correct = 0
for i in range(n_test):
    
    if (pred_np[i] == label_test_np[i]):
        n_correct += 1
        
accuracy = 100.0 * n_correct / n_test
#input('press enter')

#---
# Go through and display some images
#---
for i in range(n_test):

    img = x_test_np[i,:,:]
    print('i', i, 'img', img.shape)
    plt.title("label %d pred %d  accur %.2f" % (label_test_np[i], pred_np[i], accuracy))
    plt.imshow(X=img, cmap='gray', vmin=0, vmax=255)
    plt.show()
    
    input('press enter')
