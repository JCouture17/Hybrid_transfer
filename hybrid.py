from load_data import data
from train import train
import matplotlib.pyplot as plt
import numpy as np
from models import transfer_model
from torchvision import models
from torchsummary import summary
import torch.nn as nn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential

def load_data():
    from misc_functions import functions
    fcts = functions() 
    training_data = fcts.load('./Matlab/train_test_his/training_data.mat', 'training').astype(np.float16)   
    testing_data = fcts.load('./Matlab/train_test_his/testing_data.mat', 'testing').astype(np.float16)
    training_targets = fcts.load('./Matlab/train_test_his/training_rul.mat', 'rul')
    testing_targets = fcts.load('./Matlab/train_test_his/testing_rul.mat', 'rul')
    
    training_data = training_data.reshape([training_data.shape[0], training_data.shape[1], 1])
    testing_data = testing_data.reshape([testing_data.shape[0], testing_data.shape[1], 1])

    return training_data, training_targets, testing_data, testing_targets

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return(x)

if __name__ == "__main__":
    epochs = 100
    batch_size = 256
    lr = 0.001
    early_stop = 5
    model_name = 'alexnet'
    '''
    Available models:
        - resnet18;
        - resnet50;
        - resnet152;
        - vgg11;
        - googlenet;
        - alexnet
    '''
    training, testing = data.load_dataset(batch_size)
    
    model = models.alexnet(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False 
    num_features = model.classifier[1].in_features
    model.classificer = Identity()
    
    training_data, training_targets, testing_data, testing_targets = load_data()
    
    lstm = keras.Sequential()
    lstm.add(keras.Input(shape=(training_data.shape[1], 1)))
    lstm.add(layers.LSTM(512, activation='relu', return_sequences=True, name='lstm_1'))
    lstm.add(layers.LSTM(256, activation='relu', return_sequences=True, name='lstm_2'))
    lstm.add(layers.Flatten())
    lstm.add(layers.Dropout(0.3))
    lstm.add(layers.Dense(256, activation='relu', name='dense_1'))
    lstm.add(layers.Dense(512, activation='relu', name='dense_2'))
    lstm.add(layers.Dense(num_features, name='dense_3'))
    
    