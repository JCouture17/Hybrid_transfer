### LSTM ###
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import os
    
def load_data():
    from misc_functions import functions
    fcts = functions() 
    training_data = fcts.load('./Matlab/train_test_his/training_data.mat', 'training').astype(np.float16)   
    testing_data = fcts.load('./Matlab/train_test_his/testing_data.mat', 'testing').astype(np.float16)
    training_targets = fcts.load('./Matlab/train_test_his/training_rul.mat', 'rul')
    testing_targets = fcts.load('./Matlab/train_test_his/testing_rul.mat', 'rul')
    
    training_data = training_data.reshape([training_data.shape[0], training_data.shape[1], 1])
    testing_data = testing_data.reshape([testing_data.shape[0], testing_data.shape[1], 1])

    return training_data, training_targets, testing_data, testing_targets

if __name__ == "__main__":
    epochs = 300
    batch_size = 256
    learning_rate = 0.001
    training_data, training_targets, testing_data, testing_targets = load_data()
    
    model = keras.Sequential()
    model.add(keras.Input(shape=(training_data.shape[1], 1)))
    model.add(layers.LSTM(512, activation='relu', return_sequences=True, name='lstm_1'))
    model.add(layers.LSTM(256, activation='relu', return_sequences=True, name='lstm_2'))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(256, activation='relu', name='dense_1'))
    model.add(layers.Dense(64, activation='relu', name='dense_2'))
    model.add(layers.Dense(1, name='dense_3'))
    
    checkpoint_filepath = './chkpt/checkpoint.index'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True,
                                                    monitor='val_loss', mode='min', save_best_only='True')
    
    steps_per_epochs = np.ceil(training_data.shape[0] / batch_size)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, 
                                                    decay_steps=10*steps_per_epochs, decay_rate=0.95)
    opt = keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=opt, loss='mape', metrics=['mae', 'mse'])
    model.summary()
    hist = model.fit(training_data, training_targets, batch_size=batch_size, epochs=epochs,
              validation_data = (testing_data, testing_targets), shuffle=True, callbacks=[checkpoint])
    
    plt.plot(hist.history['loss'], label='training loss')
    plt.plot(hist.history['val_loss'], label='validation loss')
    plt.xlabel('epochs')
    plt.ylabel('mean absolute percentage error loss')
    plt.legend()
    
    latest = tf.train.latest_checkpoint(os.path.dirname(checkpoint_filepath))
    model.load_weights(checkpoint_filepath)
    model.evaluate(testing_data, testing_targets)
    