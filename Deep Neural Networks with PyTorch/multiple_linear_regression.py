# We will review how to make a prediction in several different ways by using PyTorch.

# Import the libraries and set the random seed

from torch import nn
import torch
torch.manual_seed(1)

# PREDICTION

# Set the weight and bias

w = torch.tensor([[2.0], [3.0]], requires_grad=True)
b = torch.tensor([[1.0]], requires_grad=True)

# Define the parameters. torch.mm uses matrix multiplication instead of scaler multiplication.

# Define Prediction Function

def forward(x):
    yhat = torch.mm(x, w) + b
    return yhat
  
# If we input a 1x2 tensor, because we have a 2x1 tensor as w, we will get a 1x1 tensor:

# Calculate yhat

x = torch.tensor([[1.0, 2.0]])
yhat = forward(x)
print("The result: ", yhat)

# Sample tensor X

X = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])

# Make the prediction of X 

yhat = forward(X)
print("The result: ", yhat)

# CLASS LINEAR

# Make a linear regression model using build-in function

model = nn.Linear(2, 1)

# Make a prediction of x

yhat = model(x)
print("The result: ", yhat)

# Make a prediction of X

yhat = model(X)
print("The result: ", yhat)

# BUILD CUSTOM MODULES

# Create linear_regression Class

class linear_regression(nn.Module):
    
    # Constructor
    def __init__(self, input_size, output_size):
        super(linear_regression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    # Prediction function
    def forward(self, x):
        yhat = self.linear(x)
        return yhat

      
# Build a linear regression object. The input feature size is two.
model = linear_regression(2, 1)

# You can see the randomly initialized parameters by using the parameters() method:
# Print model parameters

print("The parameters: ", list(model.parameters()))

# You can also see the parameters by using the state_dict() method:
# Print model parameters

print("The parameters: ", model.state_dict())

# Now we input a 1x2 tensor, and we will get a 1x1 tensor.
# Make a prediction of x

yhat = model(x)
print("The result: ", yhat)

# Make a prediction for multiple samples:
# Make a prediction of X

yhat = model(X)
print("The result: ", yhat)

