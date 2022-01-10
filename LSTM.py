### LSTM ###
from train import train
import matplotlib.pyplot as plt
from torch import nn
from load_data import data

class MyModel(nn.Module): 
    def __init__(self, input_shape):
        super(MyModel, self).__init__()   
        self.lstm = nn.LSTM(input_size=input_shape, hidden_size=256, num_layers=3)
        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()
        self.decoder = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
            )
        
    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        x = self.relu(x)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    epochs = 50
    batch_size = 256
    learning_rate = 0.001
    early_stop = 5
    train_loader, test_loader = data.load_datasets(batch_size)
    LSTM = MyModel(input_shape=8)
    LSTM.cuda()
    
    model, train_loss, val_loss = train.train_lstm(train_loader=train_loader, 
                                        test_loader=test_loader,
                                        lr=learning_rate, epochs=epochs, model=LSTM, 
                                        early_stop=early_stop)
    
    # Plot training and validation loss over time
    plt.plot(train_loss, label='training loss')
    plt.plot(val_loss, label='validation loss')
    plt.title('training and validation loss per epochs')
    plt.xlabel('epochs')
    plt.ylabel('average loss')
    plt.legend()

