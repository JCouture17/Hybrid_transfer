from load_data import data
from train import train
import matplotlib.pyplot as plt
from models import transfer_model
import torch
from torchsummary import summary
    
if __name__ == "__main__":
    # User inputs
    epochs = 300
    batch_size = 512
    lr = 0.001
    early_stop = 5
    neural_net = 'resnet50'
    transfer = 'y'
    
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
    training, testing = data.load_data(batch_size)
    
    # Train the model on the training data
    Transfer_network = transfer_model.load_model(neural_net)
    summary(Transfer_network,(3,224,224))
    
    if transfer == 'y':
        Transfer_network.load_state_dict(torch.load('./result/trained_transfer_' + neural_net + '.pkl'))
        test_stats = train.test_transfer(Transfer_network, testing)
            # Print test stats
        print('\nBest Validation Results: Average Loss: {:4.2f} | Accuracy: {:4.2f} | MAE: {:4.2f} | RMSE: {:4.2f}'.format(test_stats['loss'],
                                                                test_stats['accuracy'], test_stats['MAE'], test_stats['RMSE']))
    elif transfer == 'n':
        model, train_loss, val_loss = train.train_transfer_network(Transfer_network,
                        training, testing, lr=lr, epochs=epochs, model_name = neural_net, 
                        early_stop=early_stop)

        # Plot training and validation loss over time
        plt.plot(train_loss, label='training loss')
        plt.plot(val_loss, label='validation loss')
        plt.title('training and validation loss per epochs')
        plt.xlabel('epochs')
        plt.ylabel('average loss')
        plt.legend()
        
        
        # Saliency Map before training
        batch_size=1
        training, testing = data.load_data(batch_size)
        image, targets = next(iter(training))
        Transfer_network = transfer_model.load_model(neural_net)
        image.cpu(), Transfer_network.cpu()
        image.requires_grad_()
        output = Transfer_network(image)
        output_idx = output.argmax()
        output_max = output[output_idx, 0]
        output_max.backward()
        
        saliency, _ = torch.max(image.grad.data.abs(), dim=1)
        saliency = saliency.reshape(224, 224)
        image = image.reshape(-1, 224, 224)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image.cpu().detach().numpy().transpose(1, 2, 0))
        ax[0].axis('off')
        ax[1].imshow(saliency.cpu(), cmap='hot')
        ax[1].axis('off')
        plt.tight_layout()
        fig.suptitle('The Image and Its Saliency Map Before Training')
        plt.show()
        
        # Saliency Map after training
        Transfer_network. load_state_dict(torch.load('./result/trained_transfer_' + neural_net + '.pkl'))
        image, targets = next(iter(training))
        image.cpu(), Transfer_network.cpu()
        image.requires_grad_()
        output = Transfer_network(image)
        output_idx = output.argmax()
        output_max = output[output_idx, 0]
        output_max.backward()
        
        saliency, _ = torch.max(image.grad.data.abs(), dim=1)
        saliency = saliency.reshape(224, 224)
        image = image.reshape(-1, 224, 224)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image.cpu().detach().numpy().transpose(1, 2, 0))
        ax[0].axis('off')
        ax[1].imshow(saliency.cpu(), cmap='hot')
        ax[1].axis('off')
        plt.tight_layout()
        fig.suptitle('The Image and Its Saliency Map After Training')
        plt.show()
        
        

