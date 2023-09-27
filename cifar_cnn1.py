# cifar_cnn.py
# PyTorch 1.10.0-CPU Anaconda3-2020.02  Python 3.7.6
# Windows 10/11 

import numpy as np
import torch as T
import matplotlib.pyplot as plt
device = T.device('cpu')

# -----------------------------------------------------------

class Net(T.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = T.nn.Conv2d(3, 6, 5)  # in, out, k, (s=1)
    self.conv2 = T.nn.Conv2d(6, 16, 5)
    self.pool = T.nn.MaxPool2d(2, stride=2) 
    self.fc1 = T.nn.Linear(16 * 5 * 5, 120)  # 400-120-84-10
    self.fc2 = T.nn.Linear(120, 84)
    self.fc3 = T.nn.Linear(84, 10)  

  def forward(self, x):
    z = T.nn.functional.relu(self.conv1(x))  # [10, 6, 28, 28]
    z = self.pool(z)                         # [10, 6, 14, 14]
    z = T.nn.functional.relu(self.conv2(z))  # [10, 16, 10, 10]
    z = self.pool(z)                         # [10, 16, 5, 5]

    z = z.reshape(-1, 16 * 5 * 5)            # [bs, 400]
    z = T.nn.functional.relu(self.fc1(z))
    z = T.nn.functional.relu(self.fc2(z))
    z = T.log_softmax(self.fc3(z), dim=1)    # NLLLoss() 
    return z

# -----------------------------------------------------------

class CIFAR10_Dataset(T.utils.data.Dataset):
  # 3072 comma-delim pixel values (0-255) then label (0-9)
  def __init__(self, src_file):
    all_xy = np.loadtxt(src_file, usecols=range(0,3073),
      delimiter=",", comments="#", dtype=np.float32)

    tmp_x = all_xy[:, 0:3072]  # all rows, cols [0,3072]
    tmp_x /= 255.0
    tmp_x = tmp_x.reshape(-1, 3, 32, 32)  # bs, chnls, 32x32
    tmp_y = all_xy[:, 3072]    # 1-D required

    self.x_data = \
      T.tensor(tmp_x, dtype=T.float32).to(device)
    self.y_data = \
      T.tensor(tmp_y, dtype=T.int64).to(device) 

  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    lbl = self.y_data[idx]
    pixels = self.x_data[idx] 
    return (pixels, lbl)

# -----------------------------------------------------------

def accuracy(model, ds):
  X = ds[0:len(ds)][0]  # all images
  Y = ds[0:len(ds)][1]  # all targets
  with T.no_grad():
    logits = model(X)
  predicteds = T.argmax(logits, dim=1) 
  num_correct = T.sum(Y == predicteds)
  acc = (num_correct * 1.0) / len(ds)
  return acc.item()
 
# -----------------------------------------------------------

def main():
  # 0. setup
  print("\nBegin CIFAR-10 with raw data CNN demo ")
  np.random.seed(1)
  T.manual_seed(1)

  # 1. create Dataset
  print("\nLoading 5000 train and 1000 test images ")
  train_file = ".\\Data\\cifar10_train_5000.txt"
  train_ds = CIFAR10_Dataset(train_file)
  test_file = ".\\Data\\cifar10_test_1000.txt"
  test_ds = CIFAR10_Dataset(test_file)

  bat_size = 10
  train_ldr = T.utils.data.DataLoader(train_ds,
    batch_size=bat_size, shuffle=True)

  # 2. create network
  print("\nCreating CNN with 2 conv and 400-120-84-10 ")
  net = Net().to(device)

# -----------------------------------------------------------

  # 3. train model
  max_epochs = 100
  ep_log_interval = 10
  lrn_rate = 0.005

  loss_func = T.nn.NLLLoss()  # log-softmax() activation
  optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate)

  print("\nbat_size = %3d " % bat_size)
  print("loss = " + str(loss_func))
  print("optimizer = SGD")
  print("max_epochs = %3d " % max_epochs)
  print("lrn_rate = %0.3f " % lrn_rate)

  print("\nStarting training")
  net.train()
  for epoch in range(0, max_epochs):
    # T.manual_seed(1 + epoch)
    epoch_loss = 0  # for one full epoch
    for (batch_idx, batch) in enumerate(train_ldr):
      (X, Y) = batch  # X = pixels, Y = target labels
      optimizer.zero_grad()
      oupt = net(X)  # X is Size([bat_size, 3, 32, 32])
      loss_val = loss_func(oupt, Y)  # a tensor
      epoch_loss += loss_val.item()  # accumulate
      loss_val.backward()
      optimizer.step()
    if epoch % ep_log_interval == 0:
      print("epoch = %4d  | loss = %10.4f  | " % \
        (epoch, epoch_loss), end="")
      net.eval()
      acc = accuracy(net, train_ds)
      net.train()
      print(" acc = %6.4f " % acc)
  print("Done ") 

# -----------------------------------------------------------

  # 4. evaluate model accuracy
  print("\nComputing model accuracy")
  net.eval()
  acc_test = accuracy(net, test_ds)  # all at once
  print("Accuracy on test data = %0.4f" % acc_test)

  # 5. TODO: save trained model

# -----------------------------------------------------------
  
  # 6. use model to make a prediction
  print("\nPrediction for test image [29] ")
  img = test_ds[29][0] 
  label = test_ds[29][1]

  img_np = img.numpy()  # 3,32,32
  img_np = np.transpose(img_np, (1, 2, 0))
  plt.imshow(img_np)
  plt.show()
 
  labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog',
    'frog', 'horse', 'ship', 'truck']
  img = img.reshape(1, 3, 32, 32)  # make it a batch
  with T.no_grad():
    logits = net(img)
  y = T.argmax(logits)  # 0 to 9 as a tensor
  print(logits)         # 10 log-softmax values
  print(y.item())
  print(labels[y.item()])  # like "frog"

  # pp = T.exp(logits)  # pseudo-probabilities
  # print(pp)

  if y.item() == label.item():
    print("correct")
  else:
    print("wrong")
 
  print("\nEnd CIFAR-10 CNN demo ")

if __name__ == "__main__":
  main()

