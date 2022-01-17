### LSTM ###
from train import train
import matplotlib.pyplot as plt
from torch import nn
import torch
from load_data import data
from torchsummary import summary

class MyModel(nn.Module): 
    def __init__(self, input_shape):
        super(MyModel, self).__init__()   
        
        self.fc = nn.Sequential(
            nn.Linear(input_shape, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
            )
        
    def forward(self, x):
        x = self.fc(x)
        return x

if __name__ == "__main__":
    epochs = 500
    batch_size = 256
    learning_rate = 0.001
    early_stop = 5
    transfer = 'n'
    train_loader, test_loader = data.load_datasets(batch_size)
    FCNN = MyModel(input_shape=12)
    FCNN.cuda()
    summary(FCNN)
    
    if transfer == 'n':
        model, train_loss, val_loss = train.train_lstm(train_loader=train_loader, 
                                            test_loader=test_loader,
                                            lr=learning_rate, epochs=epochs, model=FCNN, 
                                            early_stop=early_stop)
        
            # Plot training and validation loss over time
        plt.plot(train_loss, label='training loss')
        plt.plot(val_loss, label='validation loss')
        plt.title('training and validation loss per epochs')
        plt.xlabel('epochs')
        plt.ylabel('average loss')
        plt.legend()
        
    elif transfer == 'y':
        FCNN.load_state_dict(torch.load('./result/trained_lstm.pkl'))
        test_stats = train.test_lstm(FCNN, test_loader)
    
            # Print test stats
        print('\nBest Validation Results: Average Loss: {:4.2f} | Accuracy: {:4.2f} | MAE: {:4.2f} | RMSE: {:4.2f}'.format(test_stats['loss'],
                                                                    test_stats['accuracy'], test_stats['MAE'], test_stats['RMSE']))
    

