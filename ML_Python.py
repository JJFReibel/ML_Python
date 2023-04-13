#!/usr/bin/python3
# Python ML
# By JJ Reibel

import random
import math

def train_val_test_split(X, y, val_size=0.1, test_size=0.1, epochs=1, random_state=None):
    # Get the total number of samples in the dataset
    n_samples = len(X)
    
    # Set the random seed if provided
    if random_state is not None:
        random.seed(random_state)
    
    # Create a list of indices that correspond to the samples in the dataset
    idx = list(range(n_samples))
    
    # Shuffle the indices
    random.shuffle(idx)
    
    # Calculate the number of samples to allocate to the validation and test sets
    n_val = math.ceil(n_samples * val_size)
    n_test = math.ceil(n_samples * test_size)
    
    # Initialize the starting and ending indices of each epoch
    epoch_start_idx = [i * n_samples // epochs for i in range(epochs)]
    epoch_end_idx = epoch_start_idx[1:] + [n_samples]
    
    # Initialize the lists to hold the indices of the samples in each set for each epoch
    train_idx_epoch = []
    val_idx_epoch = []
    test_idx_epoch = []
    
    # Loop through each epoch
    for i in range(epochs):
        # Get the indices of the samples in the current epoch
        epoch_indices = idx[epoch_start_idx[i]:epoch_end_idx[i]]
        
        # Calculate the indices of the samples to allocate to the validation and test sets
        val_idx = epoch_indices[:n_val]
        test_idx = epoch_indices[n_val:n_val+n_test]
        train_idx = epoch_indices[n_val+n_test:]
        
        # Add the indices to the appropriate lists for the current epoch
        train_idx_epoch.append(train_idx)
        val_idx_epoch.append(val_idx)
        test_idx_epoch.append(test_idx)
    
    # Initialize lists to hold the data for each epoch
    X_train_epoch = []
    X_val_epoch = []
    X_test_epoch = []
    y_train_epoch = []
    y_val_epoch = []
    y_test_epoch = []
    
    # Loop through each epoch
    for i in range(epochs):
        # Get the indices of the samples for the current epoch
        train_idx = train_idx_epoch[i]
        val_idx = val_idx_epoch[i]
        test_idx = test_idx_epoch[i]
        
        # Get the data for the current epoch
        X_train = X[train_idx]
        X_val = X[val_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]
        
        # Append the data to the appropriate lists for the current epoch
        X_train_epoch.append(X_train)
        X_val_epoch.append(X_val)
        X_test_epoch.append(X_test)
        y_train_epoch.append(y_train)
        y_val_epoch.append(y_val)
        y_test_epoch.append(y_test)
    
    # Return the data for each epoch as six arrays
    return X_train_epoch, X_val_epoch, X_test_epoch, y_train_epoch, y_val_epoch, y_test_epoch



# Example
# X_train_epoch, X_val_epoch, X_test_epoch, y_train_epoch, y_val_epoch, y_test_epoch = train_val_test_split(X, y, val_size=0.1, test_size=0.1, epochs=5)

# Can loop through
# X_train_epoch[0] # training data for epoch 0
# X_val_epoch[0] # validation data for epoch 0
# X_test_epoch[0] # test data for epoch 0
# y_train_epoch[0] # training labels for epoch 0
# y_val_epoch[0] # validation labels for epoch 0
# y_test_epoch[0] # test labels for epoch 0

