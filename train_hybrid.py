## Train decoding layer 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import matplotlib.pyplot as plt
    
if __name__ == "__main__":
    
    parallel_data = np.load('parallel_data.npy')
    parallel_test_data = np.load('parallel_test_data.npy')
    parallel_targets = np.load('parallel_targets.npy')
    parallel_test_targets = np.load('parallel_test_targets.npy')
    
    epochs = 300
    batch_size = 50
    learning_rate = 0.001
    early_stop = 5
    steps_per_epochs = np.ceil(parallel_data.shape[0] / batch_size)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, 
                                                    decay_steps=2*steps_per_epochs, decay_rate=0.95)
    opt = keras.optimizers.Adam(learning_rate=lr_schedule)
    
    decoder = keras.Sequential()
    decoder.add(keras.Input(shape=(parallel_data.shape[1], 1)))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dropout(0.2))
    decoder.add(layers.Dense(512, activation='relu'))
    decoder.add(layers.Dense(128, activation='relu'))
    decoder.add(layers.Dense(1))
    decoder.summary()
    
    checkpoint_save = './chkpt_hybrid/checkpoint_hybrid.index'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save, save_weights_only=True,
                                                    monitor='val_mape', mode='min', save_best_only='True')
    decoder.compile(optimizer=opt, loss='mse', metrics=['mae', 'mape'])
    hist2 = decoder.fit(parallel_data, parallel_targets, batch_size=batch_size, epochs=epochs,
                        validation_data = (parallel_test_data, parallel_test_targets),
                        shuffle=True, callbacks=[checkpoint])
    latest = tf.train.latest_checkpoint(os.path.dirname(checkpoint_save))
    decoder.load_weights(checkpoint_save)
    decoder.evaluate(parallel_test_data, parallel_test_targets)
    
    plt.plot(hist2.history['loss'], label='training loss')
    plt.plot(hist2.history['val_loss'], label='validation loss')
    plt.xlabel('epochs')
    plt.ylabel('mean absolute percentage error loss')
    plt.legend()