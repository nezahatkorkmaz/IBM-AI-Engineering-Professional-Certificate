'''
We will see how to apply different Activation functions in Neural Network.
In this lab, we will cover logistic regression by using PyTorch.

🟡Logistic Function
🟡Tanh
🟡Relu
🟡Compare Activation Functions

'''

# Import the libraries we need for this lab

import torch.nn as nn
import torch

import matplotlib.pyplot as plt
torch.manual_seed(2)


# LOGISTIC FUNCTION
# Create a tensor ranging from -10 to 10:
# Create a tensor

z = torch.arange(-10, 10, 0.1,).view(-1, 1)

# Create a sigmoid object
sig = nn.Sigmoid()

# Make a prediction of sigmoid function

yhat = sig(z)

# Plot the result

plt.plot(z.detach().numpy(),yhat.detach().numpy())
plt.xlabel('z')
plt.ylabel('yhat')

# For custom modules, call the sigmoid from the torch (nn.functional for the old version),
# which applies the element-wise sigmoid from the function module and plots the results
# Use the build in function to predict the result

yhat = torch.sigmoid(z)
plt.plot(z.numpy(), yhat.numpy())

plt.show()


# TANH
# When you use sequential, you can create a tanh object:
TANH = nn.Tanh()

# Make the prediction using tanh object

yhat = TANH(z)
plt.plot(z.numpy(), yhat.numpy())
plt.show()

# For custom modules, call the Tanh object from the torch (nn.functional for the old version),
# which applies the element-wise sigmoid from the function module and plots the results:

yhat = torch.tanh(z)
plt.plot(z.numpy(), yhat.numpy())
plt.show()


# RELU
# When you use sequential, you can create a Relu object:

RELU = nn.ReLU()
yhat = RELU(z)
plt.plot(z.numpy(), yhat.numpy())

# Use the build-in function to make the prediction

yhat = torch.relu(z)
plt.plot(z.numpy(), yhat.numpy())
plt.show()

# COMPARE ACTIVATION FUNCTIONS
# Plot the results to compare the activation functions

x = torch.arange(-2, 2, 0.1).view(-1, 1)
plt.plot(x.numpy(), torch.relu(x).numpy(), label='relu')
plt.plot(x.numpy(), torch.sigmoid(x).numpy(), label='sigmoid')
plt.plot(x.numpy(), torch.tanh(x).numpy(), label='tanh')
plt.legend()

