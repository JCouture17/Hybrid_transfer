import matplotlib.pyplot as plt
from torchsummary import summary
import torch
import torch.nn as nn

from train import train
from misc_functions import functions
from models import transfer_model
from load_data import data

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return(x)
    
class HybridModel(nn.Module):
    def __init__(self, model_name, cycle):
        super(HybridModel, self).__init__()
        self.tl_output = 256
        
        ### Transfer Learning Network ###
        self.transfer_network = transfer_model.load_model(model_name)
        try:
            num_features = self.transfer_network.classifier[1].in_features
            self.transfer_network.classifier = nn.Linear(num_features, self.tl_output)
        except:
            num_features = self.transfer_network.fc[1].in_features
            self.transfer_network.fc = nn.Linear(num_features, self.tl_output)
        self.transfer_network.cuda()
        
        ### Hybrid Fully-Connected Layers ###
        if cycle == 1:
            self.linear1 = nn.Linear(self.tl_output+10, 512)
            self.norm = nn.BatchNorm1d(self.tl_output+10)
        else:
            self.linear1 = nn.Linear(self.tl_output+12, 512)
            self.norm = nn.BatchNorm1d(self.tl_output+12)
        self.linear2 = nn.Linear(512, 1048)
        self.linear3 = nn.Linear(1048, 256)
        self.output = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        
    def forward(self, x, y):
        y = self.transfer_network(y)
        z = torch.cat((x, y), 1)
        z = self.norm(z)
        z = self.dropout(z)
        z = self.linear1(z)
        z = self.relu(z)
        z = self.dropout(z)
        z = self.linear2(z)
        z = self.relu(z)
        z = self.linear3(z)
        z = self.relu(z)
        z = self.output(z)
        return z
    
if __name__ == "__main__":
    ld = functions()
    epochs = 300
    batch_size = 256
    learning_rate = 0.001
    early_stop = 5
    cycle = 1
    model_name = 'alexnet'
    transfer = 'n'
    
    '''
    Available models:
        - resnet18;
        - resnet50;
        - resnet152;
        - vgg11;
        - googlenet;
        - alexnet;
        - efficientnet;
        - densenet;
    '''
    
    ## Loading the data
    train_images, test_images = data.load_images(batch_size, cycle)
    train_his, test_his = data.load_his(batch_size, cycle)
    hybrid = HybridModel(model_name, cycle)
    hybrid.cuda()
    summary(hybrid)
    
    if transfer == 'y':
        hybrid.load_state_dict(torch.load('./result/trained_hybrid_' + model_name + '.pkl'))
        test_stats = train.test_hybrid(hybrid, test_his, test_images)
            # Print test stats
        print('\nBest Validation Results: Average Loss: {:4.2f} | Accuracy: {:4.2f} | MAE: {:4.2f} | RMSE: {:4.2f}'.format(test_stats['loss'],
                                                                test_stats['accuracy'], test_stats['MAE'], test_stats['RMSE']))
    elif transfer == 'n':
        model, train_loss, val_loss = train.train_hybrid(hybrid, model_name, train_his, test_his, 
                                                     train_images, test_images, 
                                                     learning_rate, epochs) 
        
    elif transfer == 'full-train':
        for param in hybrid.parameters():
            param.requires_grad=True
        model, train_loss, val_loss = train.train_hybrid(hybrid, model_name, train_his, test_his,
                                                         train_images, test_images,
                                                         learning_rate, epochs)
        
        # Plot training and validation loss over time
        plt.plot(train_loss, label='training loss')
        plt.plot(val_loss, label='validation loss')
        plt.title('training and validation loss per epochs')
        plt.xlabel('epochs')
        plt.ylabel('average loss')
        plt.legend()

    
    
    
    
    
    