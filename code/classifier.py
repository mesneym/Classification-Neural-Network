#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


#Pre processing
import cv2
import numpy as np
from tqdm import tqdm
import glob
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text)]

def getData(data,path,label,imgSize,training):    
    names = []
    for filename in glob.glob(path):
        names.append(filename)
        
    if(training == False):
        names.sort(key=natural_keys)

    for filename in tqdm(names):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)       
        img = cv2.resize(img, (imgSize,imgSize))
        data.append([np.array(img),label])


def preProcessData(paths,training):
    data = []
    labels = len(paths)
    imgSize = 50
    count = []
     
    for i in range(labels):
        ci = getData(data,paths[i],np.eye(labels)[i],imgSize,training)
        count.append(ci)
    
    if(training):
        np.random.shuffle(data)
        np.save("trainingdata.npy", data)
    else:
        np.save("testdata.npy",data)
      


# In[ ]:


preProcessingTraining = False
preProcessingTest = False

if preProcessingTraining:
    training = True
    paths = ["dog-vs-cats/train/cat/*.jpg",
             "dog-vs-cats/train/dog/*.jpg"]
    preProcessData(paths,training)  

if preProcessingTest:
    training = False
    paths = ["dog-vs-cats/test1/*.jpg"]
    preProcessData(paths,training) 
    
    
trainingData = np.load("/content/drive/My Drive/Colab/trainingdata.npy",allow_pickle = True)
testData = np.load("/content/drive/My Drive/Colab/testdata.npy",allow_pickle = True)


# In[ ]:


import matplotlib.pyplot as plt
plt.imshow(testData[12499][0])
print(testData[12499][1])
plt.show()

# print(len(trainingData))
# print(len(testData))


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__() # just run the init of parent class (nn.Module)
        self.conv1 = nn.Conv2d(1, 32, 5) # input is 1 image, 32 output channels, 5x5 kernel / window
        self.conv2 = nn.Conv2d(32, 64, 5) # input is 32, bc the first layer output 32. Then we say the output will be 64 channels, 5x5 kernel / window
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512) #flattening.
        self.fc2 = nn.Linear(512, 2) # 512 in, 2 out bc we're doing 2 classes (dog vs cat).

    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before 
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # bc this is our output layer. No activation here.
        return F.softmax(x, dim=1)


net = Net()
print(net)


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def accuracy(testX,testY):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(testX))):
            real_class = torch.argmax(testY[i])
            net_out = net(testX[i].view(-1, 1, 50, 50))[0]  # returns a list, 
            predicted_class = torch.argmax(net_out)

            if predicted_class == real_class:
                correct += 1
            total += 1
    print("Accuracy: ", round(correct/total, 3))
    return round(correct/total,3)

def predict(testX):
    testY = []
    with torch.no_grad():
        for i in tqdm(range(len(testX))):
            net_out = net(testX[i].view(-1, 1, 50, 50))[0] # returns a list, 
            predicted_class = torch.argmax(net_out)
            testY.append(predicted_class)
            
    x = np.arange(len(testX))
    y = np.array(testY)
    sol = np.column_stack((x,y))
    np.savetxt('solution.csv',sol, delimiter=',')




def train(trainX,trainY,epochs,batchSize,learningRate,trainingInfo = False):
    optimizer = optim.Adam(net.parameters(), lr=learningRate)
    loss_function = nn.MSELoss()
     
    Loss  = []
    Accuracy = []
    for  epoch in range(epochs):
        for i in tqdm(range(0, len(trainX), batchSize)): 
                batchX = trainX[i:i+batchSize].view(-1,1,50,50)
                batchY = trainY[i:i+batchSize]
                
                net.zero_grad()
                outputs = net(batchX)
                loss = loss_function(outputs, batchY)
                loss.backward()
                optimizer.step()
        
        if(trainingInfo):
             acc = accuracy(trainX,trainY)
             Accuracy.append(acc)
             Loss.append(loss)
             print(f"Epoch: {epoch}. Loss: {loss}. Accuracy:{acc}")
    return Loss,Accuracy
           
  


# In[ ]:



################################################################################
#                    Training
################################################################################
x = torch.Tensor([i[0] for i in trainingData]).view(-1,50,50)
x = x/255.0
y = torch.Tensor([i[1] for i in trainingData])

VAL_PCT = 0.1  
val_size = int(len(x)*VAL_PCT)
print(val_size)


trainX = x[:-val_size]
trainY = y[:-val_size]


batchSize = 100
epochs = 20
learningRate = 0.001
trainingInfo = True

Loss,Accuracy= train(trainX,trainY,epochs,batchSize,learningRate,trainingInfo)




# In[ ]:


#############################################################################
#                    Plot Training Results
#############################################################################

import matplotlib.pyplot as plt
import numpy as np


fig,ax = plt.subplots()
fig,ax2 = plt.subplots()

Accuracy = np.array(Accuracy)*100
Loss = np.array(Loss)
epochs = np.arange(10)


ax.plot(epochs,Accuracy)
ax.set_title("Accuracy Vs Epochs")
ax.set_ylabel("Accuracy")
ax.set_xlabel("Epochs")

ax2.plot(epochs,Loss)
ax2.set_title("loss Vs Epochs")
ax2.set_ylabel("Loss")
ax2.set_xlabel("Epochs")
plt.show()


# In[ ]:


###############################################################################
#                     Validation
###############################################################################    
vTrainX = x[-val_size:]
vTrainY = y[-val_size:]   
Accuracy = []

accuracy(vTrainX,vTrainY)


# In[ ]:


#############################################################################
#           Training with remaining portion of Training data             
#############################################################################
batchSize = 100
epochs = 6
learningRate = 0.001
trainingInfo = True

Loss,Accuracy= train(vTrainX,vTrainY,epochs,batchSize,learningRate,trainingInfo)


# In[ ]:





# In[ ]:


#############################################################################
#                    Plot Test Results
#############################################################################

import matplotlib.pyplot as plt
import numpy as np


fig,ax = plt.subplots()
fig,ax2 = plt.subplots()

Accuracy = np.array(Accuracy)*100
Loss = np.array(Loss)
epochs = np.arange(10)


ax.plot(epochs,Accuracy)
ax.set_title("Accuracy Vs Epochs")
ax.set_ylabel("Accuracy")
ax.set_xlabel("Epochs")

ax2.plot(epochs,Loss)
ax2.set_title("loss Vs Epochs")
ax2.set_ylabel("Loss")
ax2.set_xlabel("Epochs")
plt.show()



# In[ ]:




#############################################################################
#                     Prediction for new data
#############################################################################
x = torch.Tensor([i[0] for i in testData]).view(-1,50,50)
x = x/255.0
predict(x)


# In[ ]:




