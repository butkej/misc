# Joshua Butke
# July 2019
##############

'''This script contains miscellaneous helper functions
to be used. Some might work, others might not...
'''

#IMPORTS
########

import os
import time

import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow import keras

# FUNCTION DEFINITONS
#####################

def hdf5_loader(path, split, suffix_data='.h5', suffix_label='_label.h5'):
    ''' Helper function which loads all datasets from a hdf5 file in
    a specified file at a specified path.

    # Arguments
        The split argument is used to split the key-string and sort alphanumerically
        instead of sorting the Python-standard way of 1,10,2,9,...
        The two suffix arguments define the way the datasets are looked up:
        (Training) data should always end in .h5 and corresponding labels should
        carry the same name and end in _label.h5

    # Returns
        X and y lists containing all of the data.

    # Usage
        path = 'path/to/folder'
        X, y = hdf5_loader(path, split=3)
        X = np.asarray(X)
        y = np.asarray(y)
        print(X.shape)
        print(y.shape)
    '''
    
    X = []
    y = []

    os.chdir(path)
    directory = os.fsencode(path)
    directory_contents = os.listdir(directory)
    directory_contents.sort()
    
    for file in directory_contents:
        filename = os.fsdecode(file)
        
        if filename.endswith(suffix_label):
            print("Opening: ", filename, "\n")
            
            with h5py.File(filename, 'r') as f:
                key_list = list(f.keys())
                key_list.sort(key = lambda a: int(a.split('_')[split]))
                
                for key in key_list:
                    print("Loading dataset associated with key ", str(key))
                    y.append(np.array(f[str(key)]))
                f.close()
                print("\nClosed ", filename, "\n")
                continue
        
        elif filename.endswith(suffix_data) and not filename.endswith(suffix_label):
            print("Opening: ", filename, "\n")
            
            with h5py.File(filename, 'r') as f:
                key_list = list(f.keys())
                key_list.sort(key = lambda a: int(a.split('_')[split]))
                
                for key in key_list:
                    print("Loading dataset associated with key ", str(key))
                    X.append(np.array(f[str(key)]))
                f.close()
                print("\nClosed ", filename, "\n")
    
    return X,y  

###

def multiple_hdf5_loader(path_list, split_list, suffix_data='.h5', suffix_label='_label.h5'):
    ''' Helper function which loads all datasets from targeted hdf5 files in
    a specified folder. Returns X and y arrays containing all of them.
    This function uses hdf5_loader.

    # Usage
        path_list = ['path/to/folder/file_1',
                     'path/to/folder/file_2,
                      ...
                    ]
        split_list = [int_1,int_2,...]

        X, y = multiple_hdf5_loader(path_list, split_list)
    
        print(X.shape)
        print(y.shape)
    '''
    
    X_full = np.empty((0,3,64,64)) 
    y_full = np.empty((0,1)) 

    for path, split in zip(path_list, split_list):

        print("\nIterating over dataset at: ", path)
        X, y = hdf5_loader(path, split, suffix_data, suffix_label)
        X = np.asarray(X)
        y = np.asarray(y)
        X_full = np.concatenate((X_full, X), axis=0)
        y_full = np.concatenate((y_full, y), axis=0)
        print("\nFinished with loading dataset located at: ", path)

    return X_full, y_full

###

def sigmoid_binary(ndarr):
    ''' Transform ndarray entries into 0 if they are <= 0.5
    or 1 if they are > 0.5

    Returns the transformed array.
    '''
    result = np.where(ndarr <= 0.5, 0, 1)
    return result

###

def count_uniques(ndarr):
    ''' Counts the occurence of items in an ndarray
    Outputs {item:count,item2:count2,...}
    
    Returns the computed dict.
    '''
    unique, counts = np.unique(ndarr, return_counts=True)
    result = dict(zip(unique, counts))
    print(result)
    
    return result

###

def normalize_RGB_pixels(ndarr):
    ''' normalize RGB pixel values ranging
    from 0-255 into a range of [0,1]

    Returns the normalized array.
    '''
    return (ndarr.astype(float)  / 255.0)

###

def plot_keras_metrics(hist_object):
    ''' Takes a keras history object as its input. Uses the history.history dictionary to plot learning curves.

    Does not return anything.
    '''
    pd.DataFrame(hist_object.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_y_lim(0, 1) # sets the vertical range to [0-1]
    plt.show()

    return None

###

def get_run_logdir(root_logdir):
    ''' TensorBoard visualization logging (Geron Hands-On ML Book pages 317ff)

    First create a root_logdir eg.:

    root_logdir = os.path.join(os.curdir, "logs")

    Then call this function with it as its argument. Finally pass the returned path to the
    keras.callbacks.TensorBoard(run_logdir) callback

    Returns the constructed logging path.
    '''
    run_id = time.strftime("run-%Y-%m-%d_%H:%M:%S")

    return os.path.join(root_logdir, run_id)

###

# Multiple Instance Learning (MIL) related helper functions below this line
########################################################################### 

###

def bag_accuracy(y_true, y_pred):
    ''' Compute accuracy of one bag in MIL.

    # Arguments

        y_true : Tensor (N x 1) ground truth of bag.
        y_pred : Tensor (1 X 1) prediction score of bag.

    # Returns
        acc : Tensor (1 x 1) accuracy of bag label prediction.
    ''' 
    y_true = keras.backend.mean(y_true, axis=0, keepdims=False)
    y_pred = keras.backend.mean(y_pred, axis=0, keepdims=False)
    acc = keras.backend.mean(keras.backend.equal(y_true, keras.backend.round(y_pred)))
    return acc

###

def bag_binary_loss(y_true, y_pred):
    ''' Compute binary crossentropy loss of predicting bag loss.

    # Arguments

        y_true : Tensor (N x 1) ground truth of bag.
        y_pred : Tensor (1 X 1) prediction score of bag.

    # Returns
        acc : Tensor (1 x 1) Binary Crossentropy loss of predicting bag label.
    ''' 
    y_true = keras.backend.mean(y_true, axis=0, keepdims=False)
    y_pred = keras.backend.mean(y_pred, axis=0, keepdims=False)
    loss = keras.backend.mean(keras.backend.binary_crossentropy(y_true, y_pred), axis=-1)
    return loss

