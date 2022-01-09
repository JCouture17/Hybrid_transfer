### LSTM ###
from train import train, EarlyStopping
import matplotlib.pyplot as plt
import torch
from torch import nn
from time import time
from torch.optim import Adam, lr_scheduler
import copy    

from load_data import data

class MyModel(nn.Module): 
    def __init__(self, input_shape):
        super(MyModel, self).__init__()   
        self.lstm = nn.LSTM(input_size=input_shape, hidden_size=128, num_layers=3)
        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()
        self.decoder = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
            )
        
    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        x = self.relu(x)
        x = self.decoder(x)
        return x
    
def train_model(train_loader, test_loader, lr, epochs, model, early_stop=5, opt='Adam'):
    t0 = time()
    early_stopping = EarlyStopping(patience=early_stop)
    optimizer = Adam(model.parameters(), lr=lr)
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    # Loss Criterion
    criterion = nn.MSELoss()
    train_loss = []
    val_loss = []
    best_train, best_val = 0.0, 0.0
    for epoch in range(1, epochs + 1):
        t1 = time()
        # Train and Validate
        print('epoch:', epoch)
        train_stats = train.train_lstm(model, criterion, optimizer, train_loader)
        valid_stats = train.valid_lstm(model, criterion, test_loader)
        train_loss.append(train_stats['loss'])
        val_loss.append(valid_stats['loss'])
        # Keep best model
        if valid_stats['accuracy'] > best_val or (valid_stats['MAE']==best_val and train_stats['accuracy']>=best_train):
            best_train  = train_stats['accuracy']
            print('training accuracy =', float(train_stats['accuracy']), '%')
            best_val    = valid_stats['accuracy']
            print('validation accuracy = ', float(valid_stats['accuracy']), '%')
            print('RMSE = ', float(valid_stats['RMSE']))
            print('MAE = ', float(valid_stats['MAE']))
            best_model_weights = copy.deepcopy(model.state_dict())
        lr_decay.step()
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        print('Time taken for epoch = %ds' % (time() - t1))
    # Load best model and evaluate on test set
    model.load_state_dict(best_model_weights)
    test_stats = train.valid_lstm(model, criterion, test_loader)
    print("Total time = %ds" % (time() - t0))
    # print('\nBests Model Accuracies: Train: {:4.2f} | Val: {:4.2f} | Test: {:4.2f}'.format(best_train, best_val, test_stats['accuracy']))
    print('\nBest Validation Results: Average Loss: {:4.2f} | Accuracy: {:4.2f} | MAE: {:4.2f} | RMSE: {:4.2f}'.format(test_stats['loss'],
                                                                test_stats['accuracy'], test_stats['MAE'], test_stats['RMSE']))
    
    save_dir = "./result"
    torch.save(model.state_dict(), save_dir + '/trained_model.pkl') # Use this to save the model to a .pkl file
    print('Trained model saved to \'%s/trained_model.h5\'' % save_dir)
    
    return model, train_loss, val_loss


if __name__ == "__main__":
    epochs = 50
    batch_size = 256
    learning_rate = 0.0001
    early_stop = 5
    train_loader, test_loader = data.load_datasets(batch_size)
    LSTM = MyModel(input_shape=9)
    LSTM.cuda()
    
    model, train_loss, val_loss = train_model(train_loader=train_loader, 
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

