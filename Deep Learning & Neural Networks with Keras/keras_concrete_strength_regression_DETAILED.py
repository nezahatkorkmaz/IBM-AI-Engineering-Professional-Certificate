# Let's build a baseline model

# Steps to follow:
# Data loading -> Data preprocessing -> Model creation -> Model compilation -> Model training -> Model evaluation
'''
Desired features in the baseline model:

A. Build a baseline model
Use the Keras library to build a neural network with the following:
- One hidden layer of 10 nodes, and a ReLU activation function
- Use the adam optimizer and the mean squared error as the loss function.

1. Randomly split the data into training and test sets by holding 30% of the data for testing. You can use the train_test_split helper function from Scikit-learn.

2. Train the model on the training data using 50 epochs.

3. Evaluate the model on the test data and compute the mean squared error between the predicted concrete strength and the actual concrete strength. You can use the mean_squared_error function from Scikit-learn.

4. Repeat steps 1 - 3, 50 times, i.e., create a list of 50 mean squared errors.

5. Report the mean and the standard deviation of the mean squared errors.

'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# DATA LOADING
# For example, we load the dataset into a variable. You should load the data with the actual file path after downloading it.
data_url = "https://cocl.us/concrete_data"
data = pd.read_csv(data_url)

# Print column names
print(data.columns)

# DATA PREPROCESSING
# Separating the target variable and features
X = data.drop('Strength', axis=1)  # We changed the column name to 'Strength'
y = data['Strength']
# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# MODEL CREATION
model = Sequential()
# The Sequential model is used to create simple neural network models by sequentially adding layers.
model.add(Dense(10, activation='relu', input_shape=(X_train.shape[1],)))
# In this layer, there are 10 neurons (nodes) with the ReLU (Rectified Linear Unit) activation function.
model.add(Dense(1))
# Having only one input layer and one output layer in the model makes it a simple single-layer neural network.

# MODEL COMPILATION
model.compile(optimizer='adam', loss='mean_squared_error')

# MODEL TRAINING
epochs = 50
model.fit(X_train, y_train, epochs=epochs, verbose=1)

# MODEL EVALUATION
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)




# Let's normalize the data.

'''
When normalizing the data, the following steps are taken:

B. Normalize the data

Repeat Part A, but this time use a normalized version of the data. Recall that one way to normalize the data is by
subtracting the mean from the individual predictors and dividing by the standard deviation.

❓How does the mean of the mean squared errors compare to that from Step A?

Answer: After normalizing the data in Part B, the mean squared error (MSE) obtained will likely be different from the one in Step A. Normalization brings all the features to a similar scale, which can lead to improved model performance. Therefore, the average MSE in Part B will probably be lower than the one in Step A. However, to determine this conclusively, we need to calculate the average and standard deviation of the mean squared errors over the 50 repetitions in Part B.

Explanation:

Data normalization is the process of changing the range and scale of the data to bring all the features to a similar range. This preprocessing step makes the data more suitable for model training and can enhance the model's performance. Normalization reduces the differences in magnitudes between features, preventing some features from dominating the learning process.

Two common methods for data normalization are often used:
Min-Max Normalization: normalized_value = (value - min_value) / (max_value - min_value)
Z-Score (Standard Score) Normalization: normalized_value = (value - mean) / std

'''

# NORMALIZE THE DATA
X_mean = X_train.mean()
X_std = X_train.std()
X_train_normalized = (X_train - X_mean) / X_std
X_test_normalized = (X_test - X_mean) / X_std

# BUILD THE MODEL
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(X_train_normalized.shape[1],)))
# This time, we create a neural network using the normalized data.
model.add(Dense(1))

# COMPILE THE MODEL
model.compile(optimizer='adam', loss='mean_squared_error')

# TRAIN THE MODEL
epochs = 50
model.fit(X_train_normalized, y_train, epochs=epochs, verbose=1)

# EVALUATE THE MODEL
y_pred_normalized = model.predict(X_test_normalized)
mse_normalized = mean_squared_error(y_test, y_pred_normalized)
print("Mean Squared Error (Normalized):", mse_normalized)




# Let's increase the number of epochs

'''
When increasing the number of epochs, the following steps are taken:

C. Increase the number of epochs

Repeat Part B, but this time use 100 epochs for training.

❓How does the mean of the mean squared errors compare to that from Step B?

Answer: When comparing the mean squared error (MSE) obtained in Part B with the one obtained in Part C, we expect the MSE in Part C to be lower. This is because in Part C, we trained the model with more epochs, which may have allowed the model to learn the data better and make better predictions. By evaluating the results in Part C, we can determine the optimal number of epochs for the model.

Explanation:

In Part C, we performed the same steps as in Part B (normalizing the data and training the model), but this time we trained the model with more epochs. Epochs represent how many times the model has seen the entire dataset during training. For example, with 100 epochs, the model goes through the entire dataset 100 times.

Now let's clarify the difference between Part C and Part B:
In Part C, we trained the model for a longer time (more epochs), giving the model more opportunity to learn. Therefore, we expect the model to perform better in Part C compared to Part B. In other words, the mean squared error obtained in Part C should be lower than the one obtained in Part B.

'''

# Normalize the data (Z-Score Normalization)
X_mean = X_train.mean()
X_std = X_train.std()
X_train_normalized = (X_train - X_mean) / X_std
X_test_normalized = (X_test - X_mean) / X_std

# Build the model
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(X_train_normalized.shape[1],)))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model (100 Epochs)
epochs = 100
model.fit(X_train_normalized, y_train, epochs=epochs, verbose=1)

# Evaluate the model
y_pred_normalized = model.predict(X_test_normalized)
mse_normalized = mean_squared_error(y_test, y_pred_normalized)
print("Mean Squared Error (Normalized, 100 Epochs):", mse_normalized)




# Let's increase the number of hidden layers

'''
When increasing the number of hidden layers, the following steps are taken:

D. Increase the number of hidden layers

Repeat Part C, but use a neural network with 3 hidden layers, each of 10 nodes, and ReLU activation function.

❓How does the mean of the mean squared errors compare to that from Step C?

Answer: In Step C, we evaluated the mean squared error (MSE) of the model trained with normalized data for 100 epochs. Now, in Part D, we will evaluate the mean squared error of the model trained with normalized data using a neural network with 3 hidden layers, each containing 10 nodes and ReLU activation function.
By using a deeper neural network in Part D, the model has the potential to learn more features and complexities, which is expected to result in a lower mean squared error compared to Part C.

'''

# Normalize the data (Z-Score Normalization)
X_mean = X_train.mean()
X_std = X_train.std()
X_train_normalized = (X_train - X_mean) / X_std
X_test_normalized = (X_test - X_mean) / X_std

# Build the model (3 hidden layers with 10 nodes each)
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(X_train_normalized.shape[1],)))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model (100 Epochs)
epochs = 100
model.fit(X_train_normalized, y_train, epochs=epochs, verbose=1)

# Evaluate the model
y_pred_normalized = model.predict(X_test_normalized)
mse_normalized = mean_squared_error(y_test, y_pred_normalized)
print("Mean Squared Error (3 Hidden Layers, 100 Epochs with Normalized Data):", mse_normalized)
