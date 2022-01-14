from load_data import data
from train import train
import matplotlib.pyplot as plt
    
if __name__ == "__main__":
    # User inputs
    epochs = 150
    batch_size = 512
    lr = 0.001
    early_stop = 5
    neural_net = 'alexnet'
    '''
    Available models:
        - resnet18;
        - resnet50;
        - resnet152;
        - vgg11;
        - googlenet;
        - alexnet
    '''
     
    # Load the data
    training, testing = data.load_data(batch_size)
    
    # Train the model on the training data
    model, train_loss, val_loss = train.train_transfer_network(training, testing, 
                    lr=lr, epochs=epochs, model_name = neural_net, 
                    early_stop=early_stop)
    
    # Plot training and validation loss over time
    plt.plot(train_loss, label='training loss')
    plt.plot(val_loss, label='validation loss')
    plt.title('training and validation loss per epochs')
    plt.xlabel('epochs')
    plt.ylabel('average loss')
    plt.legend()

