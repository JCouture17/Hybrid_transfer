import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
from torchsummary import summary
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from skimage import io as skio
from misc_functions import functions
from torchvision import transforms
from torch.utils.data import DataLoader

class CustomImageDataset:
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset[0])
    
    def __getitem__(self, index):
        x = self.dataset[0][index]
        if self.transform:
            x = self.transform(x)
        # y = self.dataset[1][index]
        return x
    

def load_data():
    from misc_functions import functions
    fcts = functions() 
    training_data = fcts.load('/home/jonathan/Documents/GitHub/Hybrid_transfer/Data/training_his.mat', 
                              'training_his').astype(np.float16)   
    testing_data = fcts.load('/home/jonathan/Documents/GitHub/Hybrid_transfer/Data/testing_his.mat',
                             'testing_his').astype(np.float16)
    training_targets = fcts.load('/home/jonathan/Documents/GitHub/Hybrid_transfer/Data/training_targets.mat', 'training_targets')
    testing_targets = fcts.load('/home/jonathan/Documents/GitHub/Hybrid_transfer/Data/testing_targets.mat', 'testing_targets')
    
    training_data = training_data.reshape([training_data.shape[0], training_data.shape[1], 1])
    testing_data = testing_data.reshape([testing_data.shape[0], testing_data.shape[1], 1])

    return training_data, training_targets, testing_data, testing_targets

def load_images():
    fcts = functions()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    train_data = skio.imread('/home/jonathan/Documents/GitHub/Hybrid_transfer/Data/training_data.tif')
    test_data = skio.imread('/home/jonathan/Documents/GitHub/Hybrid_transfer/Data/testing_dataset.tif')

    train_targets = fcts.load('/home/jonathan/Documents/GitHub/Hybrid_transfer/Data/training_targets.mat', 'training_targets').astype(np.int16)
    test_targets = fcts.load('/home/jonathan/Documents/GitHub/Hybrid_transfer/Data/testing_targets.mat', 'testing_targets').astype(np.int16)
    
    train_data = CustomImageDataset(dataset=(train_data, train_targets), transform=transform)
    test_data = CustomImageDataset(dataset=(test_data, test_targets), transform=transform)
    # Build dataloader
    train_loader = DataLoader(train_data, shuffle=False, batch_size=1)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=1)
    
    return train_loader, test_loader


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return(x)

if __name__ == "__main__":
    ld = functions()
    epochs = 150
    batch_size = 5
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
    
    ## Loading the transferred network and data
    train_data, test_data = load_images()
    
    model = models.alexnet(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False 
    num_features = model.classifier[1].in_features
    model.classifier = Identity()
    model.cuda()
    summary(model,(3,224,224))
    
    # Compute the Tl model output to combine the LSTM's output
    tl_output = torch.empty(1,num_features).cuda()
    for i, x in enumerate(train_data): 
        x = x.cuda()
        tl_output = torch.cat((tl_output, model(x)))
    tl_test = torch.empty(1, num_features).cuda()
    for i, x in enumerate(test_data):
        x = x.cuda()
        tl_test = torch.cat((tl_test, model(x)))
        
    tl_output = tl_output[1:].cpu().numpy()
    tl_test = tl_test[1:].cpu().numpy()
    # tl_output = tf.convert_to_tensor(tl_output)
    # tl_test = tf.convert_to_tensor(tl_test)g
    
    
    # Importing the LSTM model
    training_data, training_targets, testing_data, testing_targets = load_data()
    
    lstm = keras.Sequential()
    lstm.add(keras.Input(shape=(training_data.shape[1], 1)))
    lstm.add(layers.LSTM(512, activation='relu', return_sequences=True, name='lstm_1'))
    lstm.add(layers.LSTM(256, activation='relu', return_sequences=True, name='lstm_2'))
    lstm.add(layers.LSTM(128, activation='relu', return_sequences=True, name='lstm_3'))
    lstm.add(layers.Flatten())
    lstm.add(layers.Dropout(0.3))
    lstm.add(layers.Dense(256, activation='relu', name='dense_1'))
    # model.add(layers.Dense(512, activation='relu', name='dense_2'))
    lstm.add(layers.Dense(500, activation='relu', name='dense_3'))
    lstm.add(layers.Dense(1, name='output'))
    
    steps_per_epochs = np.ceil(training_data.shape[0] / batch_size)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, 
                                                    decay_steps=10*steps_per_epochs, decay_rate=0.95)
    opt = keras.optimizers.Adam(learning_rate=lr_schedule)
    lstm.compile(optimizer=opt, loss='mse', metrics=['mae', 'mape'])
    
    # Load in the trained LSTM model
    checkpoint_filepath = './chkpt/checkpoint.index'
    latest = tf.train.latest_checkpoint(os.path.dirname(checkpoint_filepath))
    lstm.load_weights(checkpoint_filepath)
    lstm.evaluate(testing_data, testing_targets) # obtain the LSTM's accuracy on its own
    lstm.pop() # removes the final output layer of the LSTM
    lstm.summary()
    
    # Compute the LSTM output to combine with the TL model
    lstm_output = np.array(lstm.predict(training_data))
    # lstm_output = tf.convert_to_tensor(lstm_output)
    lstm_test = np.array(lstm.predict(testing_data))
    # lstm_test = tf.convert_to_tensor(lstm_test)

    ## Adding a decoder layer
    parallel_data = np.concatenate((lstm_output, tl_output), 1)
    parallel_targets = training_targets
    parallel_test_data = np.concatenate((lstm_test, tl_test), 1)
    parallel_test_targets = testing_targets
    parallel_data = parallel_data.reshape([parallel_data.shape[0], parallel_data.shape[1], 1])
    parallel_test_data = parallel_test_data.reshape([parallel_test_data.shape[0], parallel_test_data.shape[1], 1])
    
    # Have to save and reload due to lack of memory space
    np.save('parallel_data', parallel_data)
    np.save('parallel_test_data', parallel_test_data)
    np.save('parallel_targets', parallel_targets)
    np.save('parallel_test_targets', parallel_test_targets)
    
    
    # import tensorflow as tf
    # from tensorflow import keras
    # from tensorflow.keras import layers
    # import numpy as np
    
    # parallel_data = np.load('parallel_data.npy')
    # parallel_test_data = np.load('parallel_test_data.npy')
    # parallel_targets = np.load('parallel_targets.npy')
    # parallel_test_targets = np.load('parallel_test_targets.npy')
    
    # epochs = 100
    # batch_size = 150
    # learning_rate = 0.001
    # early_stop = 5
    # steps_per_epochs = np.ceil(parallel_data.shape[0] / batch_size)
    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, 
    #                                                 decay_steps=10*steps_per_epochs, decay_rate=0.95)
    # opt = keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # decoder = keras.Sequential()
    # decoder.add(keras.Input(shape=(parallel_data.shape[1], 1)))
    # decoder.add(layers.Dropout(0.3))
    # decoder.add(layers.Dense(1024, activation='relu'))
    # decoder.add(layers.Dense(512, activation='relu'))
    # decoder.add(layers.Dense(128, activation='relu'))
    # decoder.add(layers.Dense(1))
    
    
    # checkpoint_save = './chkpt_hybrid/checkpoint_hybrid.index'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save, save_weights_only=True,
    #                                                 monitor='val_loss', mode='min', save_best_only='True')
    # decoder.compile(optimizer=opt, loss='mape', metrics=['mae', 'mse'])
    # hist2 = decoder.fit(parallel_data, parallel_targets, batch_size=batch_size, epochs=epochs,
    #                     validation_data = (parallel_test_data, parallel_test_targets),
    #                     shuffle=True, callbacks=[checkpoint])
    # latest = tf.train.latest_checkpoint(os.path.dirname(checkpoint_save))
    # decoder.load_weights(checkpoint_save)
    # decoder.evaluate(parallel_test_data, parallel_test_targets)
    
    # plt.plot(hist2.history['loss'], label='training loss')
    # plt.plot(hist2.history['val_loss'], label='validation loss')
    # plt.xlabel('epochs')
    # plt.ylabel('mean absolute percentage error loss')
    # plt.legend()
    
    
    
    
    
    
    