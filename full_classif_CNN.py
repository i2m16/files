mport torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torchvision.utils import make_grid
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
%matplotlib inline

#Convert MNIST image files into a tensor of 4-D (#imgs, Height, Width, )
transform = transforms.ToTensor()

#train data
train_data = datasets.MNIST(root='/cnn.data',train=True, download=True, transform=transform)
#test data
test_data = datasets.MNIST(root='/cnn.data',train=False, download=True, transform=transform)

#create small batch for images
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=True)

#Define the CNN model
conv1 = nn.Conv2d(1, 6, 3, 1) #(num of inputs, num of outputs, kernel size, stride)
conv2 = nn.Conv2d(6, 16, 3, 1) #(num of inputs, num of outputs, kernel size, stride)

#Grab 1 MNIST record/image
for i, (X_train, y_train) in enumerate(train_data):
  break


x = X_train.view(1, 1, 28, 28)

#first convolution
x = F.relu(conv1(x)) #rectified linear unit for activation function

x = F.max_pool2d(x, 2, 2)

x = F.relu(conv2(x))

x = F.max_pool2d(x, 2, 2)

x.shape

#Model class
class ConvolutionalNetwork(nn.Module):
  def __init__(self):
    super().__init__()

    #Define the CNN model
    self.conv1 = nn.Conv2d(1, 6, 3, 1) #(num of inputs, num of outputs, kernel size, stride)
    self.conv2 = nn.Conv2d(6, 16, 3, 1) #(num of inputs, num of outputs, kernel size, stride)

    #Fully connected layer
    self.fc1 = nn.Linear(5*5*16, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, X):
    X = F.relu(self.conv1(X))
    X = F.max_pool2d(X,2,2) #2x2 kernel and stride 2

    #second pass
    X = F.relu(self.conv2(X))
    X = F.max_pool2d(X,2,2) #2x2 kernel and stride 2

    # Re-view to flatten it out
    X = X.view(-1,16*5*5) #negative one so that we can vary the batch zize

    #Fully connected layers
    X = F.relu(self.fc1(X))
    X = F.relu(self.fc2(X))
    X = self.fc3(X)

    return F.log_softmax(x,dim=1)

#Create and instance of our model
torch.manual_seed(41)

model = ConvolutionalNetwork()

model

#Loss function optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# track how long it takes
import time
start_time=time.time()

#Create variables to track things
epochs = 5
train_losses = []
test_losses = []
train_correct = []
test_correct = []

#for Loop of Epochs
for i in range(epochs):

  trn_corr = 0
  tst_corr = 0

  #train CNN
  for b,(X_train, y_train) in enumerate(train_loader):
    b+=1
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    predicted = torch.max(y_pred.data, 1)[1]
    batch_corr = (predicted == y_train).sum()
    trn_crr += batch_corr

    #Update out nn.Parameter
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

   #Print out some results
    if b%600 == 0:
      print(f'Epoch: {i}  Batch: {b}  Loss: {loss.item()}')


  train_losses.append(loss)
  train.correct.append(trn_corr)

# #test



current_time = time.time()
total = current_time - start_time

total



