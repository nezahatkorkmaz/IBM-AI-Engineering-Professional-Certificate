# We will see how to make a prediction using multiple samples.


# CLASS LINEAR
from torch import nn
import torch

# Set the random seed

class linear_regression(nn.Module):
    def __init__(self,input_size,output_size):
        super(linear_regression,self).__init__()
        self.linear=nn.Linear(input_size,output_size)
    def forward(self,x):
        yhat=self.linear(x)
        return yhat

# create a linear regression object, as our input and output will be two we set the parameters accordingly
model=linear_regression(1,10)
model(torch.tensor([1.0]))

list(model.parameters())

# we can create a tensor with two rows representing one sample of data
x=torch.tensor([[1.0]])

# we can make a prediction
yhat=model(x)
yhat

# each row in the following tensor represents a different sample
X=torch.tensor([[1.0],[1.0],[3.0]])

# we can make a prediction using multiple samples
Yhat=model(X)
Yhat

# we can make a prediction
