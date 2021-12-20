### LSTM ###
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
    
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

if __name__ == "__main__":
    epochs = 300
    batch_size = 512
    learning_rate = 0.001
    training_data, training_targets, testing_data, testing_targets = load_data()
    
    model = keras.Sequential()
    model.add(keras.Input(shape=(training_data.shape[1], 1)))
    model.add(layers.LSTM(512, activation='relu', return_sequences=True, name='lstm_1'))
    model.add(layers.LSTM(256, activation='relu', return_sequences=True, name='lstm_2'))
    model.add(layers.LSTM(128, activation='relu', return_sequences=True, name='lstm_3'))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(256, activation='relu', name='dense_1'))
    # model.add(layers.Dense(512, activation='relu', name='dense_2'))
    model.add(layers.Dense(500, activation='relu', name='dense_3'))
    model.add(layers.Dense(1, name='output'))
    
    checkpoint_filepath = './chkpt/checkpoint.index'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True,
                                                    monitor='val_loss', mode='min', save_best_only='True')
    
    steps_per_epochs = np.ceil(training_data.shape[0] / batch_size)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, 
                                                    decay_steps=10*steps_per_epochs, decay_rate=0.95)
    opt = keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(optimizer=opt, loss='mse', metrics=['mae', 'mape'])
    model.summary()
    hist = model.fit(training_data, training_targets, batch_size=batch_size, epochs=epochs,
              validation_data = (testing_data, testing_targets), shuffle=True, callbacks=[checkpoint])
    
    plt.plot(hist.history['loss'], label='training loss')
    plt.plot(hist.history['val_loss'], label='validation loss')
    plt.xlabel('epochs')
    plt.ylabel('mean squared error loss')
    plt.legend()
    
    latest = tf.train.latest_checkpoint(os.path.dirname(checkpoint_filepath))
    model.load_weights(checkpoint_filepath)
    model.evaluate(testing_data, testing_targets)
    