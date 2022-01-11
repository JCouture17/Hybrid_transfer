import matplotlib.pyplot as plt
# import numpy as np
# from torchvision import models
from torchsummary import summary
import torch
import torch.nn as nn
# import os

from train import train
from misc_functions import functions
from models import transfer_model
from LSTM import MyModel
from load_data import data

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return(x)
    
class HybridModel(nn.Module):
    def __init__(self, model_name):
        super(HybridModel, self).__init__()
        ### LSTM Network ###
        self.lstm = MyModel(input_shape = 8)
        self.lstm.load_state_dict(torch.load('./result/trained_lstm.pkl'))
        self.lstm.decoder[5] = Identity()
        self.lstm.eval()
        self.lstm.cuda()
        self.flatten = nn.Flatten()
        
        ### Transfer Learning Network ###
        self.transfer_network = transfer_model.load_model(model_name)
        for param in self.transfer_network.parameters():
            param.requires_grad = False 
        num_features = self.transfer_network.classifier[-1].in_features
        # model.classifier[-1] = nn.Linear(num_features, 512)
        self.transfer_network.classifier[-1] = Identity()
        self.transfer_network.cuda()
        
        ### Hybrid Fully-Connected Layers ###
        self.linear1 = nn.Linear(num_features+1024, 2048)
        self.linear2 = nn.Linear(2048, 512)
        self.linear3 = nn.Linear(512, 128)
        self.output = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x, y):
        x = self.lstm(x)
        x = self.flatten(x)
        y = self.transfer_network(y)
        z = torch.cat((x, y), 1)
        z = self.linear1(z)
        z = self.relu(z)
        z = self.linear2(z)
        z = self.relu(z)
        z = self.linear3(z)
        z = self.relu(z)
        z = self.output(z)
        return z
    
if __name__ == "__main__":
    ld = functions()
    epochs = 150
    batch_size = 150
    learning_rate = 0.001
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
    
    ## Loading the data
    train_images, test_images = data.load_data(batch_size)
    train_his, test_his = data.load_datasets(batch_size)
    hybrid = HybridModel('alexnet')
    hybrid.cuda()
    summary(hybrid)
    model, train_loss, val_loss = train.train_hybrid(hybrid, train_his, test_his, 
                                                     train_images, test_images, 
                                                     learning_rate, epochs) 
    
    # Plot training and validation loss over time
    plt.plot(train_loss, label='training loss')
    plt.plot(val_loss, label='validation loss')
    plt.title('training and validation loss per epochs')
    plt.xlabel('epochs')
    plt.ylabel('average loss')
    plt.legend()

    
    
    
    
    
    