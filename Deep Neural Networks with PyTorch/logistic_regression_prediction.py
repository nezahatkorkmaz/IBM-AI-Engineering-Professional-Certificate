# We will create a logistic regression object with the nn.Sequential model. This is an activation function constructor.

# Import the libraries we need for this lab

import torch.nn as nn
import torch
import matplotlib.pyplot as plt 

# Set the random seed

torch.manual_seed(2)

# LOGISTIC FUNCTION
# Create a tensor ranging from -100 to 100
z = torch.arange(-100, 100, 0.1).view(-1, 1)
print("The tensor: ", z)

# Create sigmoid object
sig = nn.Sigmoid()


# Use sigmoid object to calculate the yhat
yhat = sig(z)

# Plot the results: 
plt.plot(z.numpy(), yhat.numpy())
plt.xlabel('z')
plt.ylabel('yhat')

# Apply the element-wise Sigmoid from the function module and plot the results
yhat = torch.sigmoid(z)
plt.plot(z.numpy(), yhat.numpy())


# Build a Logistic Regression with nn.Sequential
# Create a 1x1 tensor where x represents one data sample with one dimension, and 2x1 tensor X represents two data samples of one dimension
# Create x and X tensor

x = torch.tensor([[1.0]])
X = torch.tensor([[1.0], [100]])
print('x = ', x)
print('X = ', X)

# Create a logistic regression object with the nn.Sequential model with a one-dimensional input
# Use sequential function to create model

model = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())

# Print the parameters

print("list(model.parameters()):\n ", list(model.parameters()))
print("\nmodel.state_dict():\n ", model.state_dict())

# The prediction for x

yhat = model(x)
print("The prediction: ", yhat)

# Make a prediction with multiple samples
# The prediction for X

yhat = model(X)
yhat

# Calling the object performed the following operation
# Create a 1x2 tensor where x represents one data sample with one dimension, and 2x3 tensor X represents one data sample of two dimensions
# Create and print samples

x = torch.tensor([[1.0, 1.0]])
X = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
print('x = ', x)
print('X = ', X)

# Create a logistic regression object with the nn.Sequential model with a two-dimensional input
# Create new model using nn.sequential()

model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())

# Print the parameters

print("list(model.parameters()):\n ", list(model.parameters()))
print("\nmodel.state_dict():\n ", model.state_dict())

# Make a prediction with one sample
# Make the prediction of x

yhat = model(x)
print("The prediction: ", yhat)

# Make a prediction with multiple samples
# The prediction of X

yhat = model(X)
print("The prediction: ", yhat)


# Build Custom Modules
# Create logistic_regression custom class

class logistic_regression(nn.Module):
    
    # Constructor
    def __init__(self, n_inputs):
        super(logistic_regression, self).__init__()
        self.linear = nn.Linear(n_inputs, 1)
    
    # Prediction
    def forward(self, x):
        yhat = torch.sigmoid(self.linear(x))
        return yhat

# Create a 1x1 tensor where x represents one data sample with one dimension, and 3x1 tensor where  𝑋 represents one data sample of one dimension
# Create x and X tensor

x = torch.tensor([[1.0]])
X = torch.tensor([[-100], [0], [100.0]])
print('x = ', x)
print('X = ', X)

# Create a model to predict one dimension
# Create logistic regression model

model = logistic_regression(1)

# In this case, the parameters are randomly initialized. You can view them the following ways:
# Print parameters 

print("list(model.parameters()):\n ", list(model.parameters()))
print("\nmodel.state_dict():\n ", model.state_dict())

# Make a prediction with one sample
# Make the prediction of x

yhat = model(x)
print("The prediction result: \n", yhat)

# Make a prediction with multiple samples
# Make the prediction of X

yhat = model(X)
print("The prediction result: \n", yhat)

# Create logistic regression model (with 2 inputs)

model = logistic_regression(2)

# Create a 1x2 tensor where x represents one data sample with one dimension, and 3x2 tensor X represents one data sample of one dimension:
# Create x and X tensor

x = torch.tensor([[1.0, 2.0]])
X = torch.tensor([[100, -100], [0.0, 0.0], [-100, 100]])
print('x = ', x)
print('X = ', X)

# Make a prediction with one sample
# Make the prediction of x

yhat = model(x)
print("The prediction result: \n", yhat)

# Make a prediction with multiple samples
# Make the prediction of X

yhat = model(X)
print("The prediction result: \n", yhat)
