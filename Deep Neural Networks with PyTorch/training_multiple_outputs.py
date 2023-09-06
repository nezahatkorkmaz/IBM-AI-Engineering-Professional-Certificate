# We will see how to create a complicated models using pytorch build in functions.

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn,optim
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Set the random seed
torch.manual_seed(1)



# Create a dataset class with two-dimensional features and two targets
from torch.utils.data import Dataset, DataLoader
class Data(Dataset):
    def __init__(self):
            self.x=torch.zeros(20,2)
            self.x[:,0]=torch.arange(-1,1,0.1)
            self.x[:,1]=torch.arange(-1,1,0.1)
            self.w=torch.tensor([ [1.0,-1.0],[1.0,3.0]])
            self.b=torch.tensor([[1.0,-1.0]])
            self.f=torch.mm(self.x,self.w)+self.b
            
            self.y=self.f+0.001*torch.randn((self.x.shape[0],1))
            self.len=self.x.shape[0]

    def __getitem__(self,index):

        return self.x[index],self.y[index]
    
    def __len__(self):
        return self.len

# creating dataset object
data_set=Data()


# Create the Model, Optimizer, and Total Loss Function (cost)
# Create aclass linear_regression(nn.Module):
    def __init__(self,input_size,output_size):
        super(linear_regression,self).__init__()
        self.linear=nn.Linear(input_size,output_size)
    def forward(self,x):
        yhat=self.linear(x)
        return yhat custom module

# Create an optimizer object and set the learning rate to 0.1. Don't forget to enter the model parameters in the constructor.
model=linear_regression(2,2)

optimizer = optim.SGD(model.parameters(), lr = 0.1)

# Create the criterion function that calculates the total loss or cost:
criterion = nn.MSELoss()

# Create a data loader object and set the batch_size to 5:
train_loader=DataLoader(dataset=data_set,batch_size=5)



# Train the Model via Mini-Batch Gradient Descent 
# Run 100 epochs of Mini-Batch Gradient Descent and store the total loss or cost for every iteration.
# Remember that this is an approximation of the true total loss or cost.

LOSS=[]
 
epochs=100
   
for epoch in range(epochs):
    for x,y in train_loader:
        #make a prediction 
        yhat=model(x)
        #calculate the loss
        loss=criterion(yhat,y)
        #store loss/cost 
        LOSS.append(loss.item())
        #clear gradient 
        optimizer.zero_grad()
        #Backward pass: compute gradient of the loss with respect to all the learnable parameters
        loss.backward()
        #the step function on an Optimizer makes an update to its parameters
        optimizer.step()
     

# Plot the cost:
plt.plot(LOSS)
plt.xlabel("iterations ")
plt.ylabel("Cost/total loss ")
plt.show()


