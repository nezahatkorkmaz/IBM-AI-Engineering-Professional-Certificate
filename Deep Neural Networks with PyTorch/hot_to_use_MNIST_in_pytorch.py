# !pip install torchvision==0.9.1 torch==1.8.1 
import torch 
import matplotlib.pylab as plt
import numpy as np
torch.manual_seed(0)

# Show data by diagram

def show_data(data_sample, shape = (28, 28)):
    plt.imshow(data_sample[0].numpy().reshape(shape), cmap='gray')
    plt.title('y = ' + str(data_sample[1]))

# Run the command below when you do not have torchvision installed
# !mamba install -y torchvision

import torchvision.transforms as transforms
import torchvision.datasets as dsets

# Import the prebuilt dataset into variable dataset


dataset = dsets.MNIST(
    root = './data',  
    download = False, 
    transform = transforms.ToTensor()
)

# Examine whether the elements in dataset MNIST are tuples, and what is in the tuple?

print("Type of the first element: ", type(dataset[0]))
print("The length of the tuple: ", len(dataset[0]))
print("The shape of the first element in the tuple: ", dataset[0][0].shape)
print("The type of the first element in the tuple", type(dataset[0][0]))
print("The second element in the tuple: ", dataset[0][1])
print("The type of the second element in the tuple: ", type(dataset[0][1]))
print("As the result, the structure of the first element in the dataset is (tensor([1, 28, 28]), tensor(7)).")

'''
As shown in the output, the first element in the tuple is a cuboid tensor. As you can see, there is a dimension with only size 1, so basically, it is a rectangular tensor.
The second element in the tuple is a number tensor, which indicate the real number the image shows. As the second element in the tuple is tensor(7), the image should show a hand-written 7.
'''
# Plot the first element in the dataset

show_data(dataset[0])

# Plot the second element in the dataset

show_data(dataset[1])

# Torchvision Transforms
'''
We can apply some image transform functions on the MNIST dataset.

As an example, the images in the MNIST dataset can be cropped and converted to a tensor. We can use transform.Compose we learned from the previous lab to combine the two transform functions.
'''
# Combine two transforms: crop and convert to tensor. Apply the compose to MNIST dataset

croptensor_data_transform = transforms.Compose([transforms.CenterCrop(20), transforms.ToTensor()])
dataset = dsets.MNIST(root = './data', download = True, transform = croptensor_data_transform)
print("The shape of the first element in the first tuple: ", dataset[0][0].shape)
'''
We can see the image is now 20 x 20 instead of 28 x 28.

Let us plot the first image again. Notice that the black space around the 7 become less apparent.
'''
# Plot the first element in the dataset

show_data(dataset[0],shape = (20, 20))

# Plot the second element in the dataset

show_data(dataset[1],shape = (20, 20))

# In the below example, we horizontally flip the image, and then convert it to a tensor. Use transforms.Compose() to combine these two transform functions. Plot the flipped image.

# Construct the compose. Apply it on MNIST dataset. Plot the image out.

fliptensor_data_transform = transforms.Compose([transforms.RandomHorizontalFlip(p = 1),transforms.ToTensor()])
dataset = dsets.MNIST(root = './data', download = True, transform = fliptensor_data_transform)
show_data(dataset[1])
