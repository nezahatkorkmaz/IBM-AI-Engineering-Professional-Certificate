'''
We will learn 
🌻How to create a dataset in pytorch.
🌻How to perform transformations on the dataset.
'''

import torch
from torch.utils.data import Dataset
torch.manual_seed(1)

# Let us try to create our own dataset class
# Define class for dataset

class toy_set(Dataset):
    
    # Constructor with defult values 
    def __init__(self, length = 100, transform = None):
        self.len = length
        self.x = 2 * torch.ones(length, 2)
        self.y = torch.ones(length, 1)
        self.transform = transform
     
    # Getter
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)     
        return sample
    
    # Get Length
    def __len__(self):
        return self.len

  # Now, let us create our toy_set object, and find out the value on index 1 and the length of the inital dataset
  # Create Dataset Object. Find out the value on index 1. Find out the length of Dataset Object.

    our_dataset = toy_set()
    print("Our toy_set object: ", our_dataset)
    print("Value on index 0 of our toy_set object: ", our_dataset[0])
    print("Our toy_set length: ", len(our_dataset))

# Use loop to print out first 3 elements in dataset

for i in range(3):
    x, y=our_dataset[i]
    print("index: ", i, '; x:', x, '; y:', y)

# The dataset object is an Iterable; as a result, we apply the loop directly on the dataset object
for x,y in our_dataset:
    print(' x:', x, 'y:', y)

# TRANSFORMS
# Create tranform class add_mult

class add_mult(object):
    
    # Constructor
    def __init__(self, addx = 1, muly = 2):
        self.addx = addx
        self.muly = muly
    
    # Executor
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x + self.addx
        y = y * self.muly
        sample = x, y
        return sample

# Create an add_mult transform object, and an toy_set object

a_m = add_mult()
data_set = toy_set()

# Use loop to print out first 10 elements in dataset

for i in range(10):
    x, y = data_set[i]
    print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
    x_, y_ = a_m(data_set[i])
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)

# Create a new data_set object with add_mult object as transform

cust_data_set = toy_set(transform = a_m)

# Use loop to print out first 10 elements in dataset

for i in range(10):
    x, y = data_set[i]
    print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
    x_, y_ = cust_data_set[i]
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)

# COMPOSE
# Run the command below when you do not have torchvision installed
# !mamba install -y torchvision

from torchvision import transforms

# Create tranform class mult

class mult(object):
    
    # Constructor
    def __init__(self, mult = 100):
        self.mult = mult
        
    # Executor
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x * self.mult
        y = y * self.mult
        sample = x, y
        return sample

# Combine the add_mult() and mult()

data_transform = transforms.Compose([add_mult(), mult()])
print("The combination of transforms (Compose): ", data_transform)

data_transform(data_set[0])
x,y=data_set[0]
x_,y_=data_transform(data_set[0])
print( 'Original x: ', x, 'Original y: ', y)

print( 'Transformed x_:', x_, 'Transformed y_:', y_)

# Now we can pass the new Compose object (The combination of methods add_mult() and mult) to the constructor for creating toy_set object
# Create a new toy_set object with compose object as transform

compose_data_set = toy_set(transform = data_transform)

# Let us print out the first 3 elements in different toy_set datasets in order to compare the output after different transforms have been applied:
# Use loop to print out first 3 elements in dataset

for i in range(3):
    x, y = data_set[i]
    print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
    x_, y_ = cust_data_set[i]
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)
    x_co, y_co = compose_data_set[i]
    print('Index: ', i, 'Compose Transformed x_co: ', x_co ,'Compose Transformed y_co: ',y_co)
  
