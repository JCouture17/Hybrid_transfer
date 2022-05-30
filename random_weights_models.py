from load_data import data
from train import train
import matplotlib.pyplot as plt
import torch
from torchsummary import summary
from torchvision import models
import torch.nn as nn

if __name__ == "__main__":
    # User inputs
    epochs = 200
    batch_size = 256
    lr = 0.001
    early_stop = 5
    cycle = 1
    neural_net = 'vgg11_randomWeights'
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
    
    # Load the data
    training, testing = data.load_images(batch_size, cycle)
    
    if neural_net == 'vgg11_randomWeights':
        model = models.vgg11(pretrained=False)
    elif neural_net == 'alexnet_randomWeights':
        model = models.vgg11(pretrained=False)
    
    try:
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1))
    # except:
    #     num_features = model.classifier.in_features
    #     # print("didn't get to first")
    except AttributeError:
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1))
    except TypeError:
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1))   
    except:
        num_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1))       

    if torch.cuda.is_available:
        model.cuda()
        
    Transfer_network = model
    summary(Transfer_network, (3,224,224))
    
    
    if transfer == 'y':
        Transfer_network.load_state_dict(torch.load('./result/trained_transfer_' + neural_net + '.pkl'))
        test_stats = train.test_transfer(Transfer_network, testing)
        # Print test stats
        print('\nBest Validation Results: Average Loss: {:4.2f} | Accuracy: {:4.2f} | MAE: {:4.2f} | RMSE: {:4.2f}'.format(test_stats['loss'],
                                                               test_stats['accuracy'], test_stats['MAE'], test_stats['RMSE']))
    elif transfer == 'n':
        trained_model, train_loss, val_loss = train.train_transfer_network(Transfer_network,
                                    training, testing, lr=lr, epochs=epochs, model_name=neural_net,
                                    early_stop=early_stop)
        # Plot training and validation loss over time
        plt.plot(train_loss, label='training loss')
        plt.plot(val_loss, label='validation loss')
        plt.title('training and validation loss per epochs')
        plt.xlabel('epochs')
        plt.ylabel('average loss')
        plt.legend()
