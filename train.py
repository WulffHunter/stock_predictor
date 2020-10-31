import utils
import datetime
import os

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional

# Modes: vanilla, stacked, bidirectional

def define_model(feature_count, step_count, mode='vanilla'):
    model = Sequential()

    if mode == 'vanilla':
        model.add(LSTM(50, activation='relu', input_shape=(step_count, feature_count)))
    elif mode == 'stacked':
        model.add(LSTM(50, activation='relu', input_shape=(step_count, feature_count)))
        model.add(LSTM(50, activation='relu'))
    elif mode == 'bidirectional':
        model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(step_count, feature_count)))
    else:
        raise TypeError(f"Unknown layer type {mode}")

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    return model


def train(train_X, train_y,
          test_X, test_y,
          model_file,
          logs_root_dir,
          feature_count=1,
          epochs=200,
          model_mode='vanilla',
          tensorboard_on=True):
    train_X = train_X.reshape((train_X.shape[0], train_X.shape[1], feature_count))
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], feature_count))

    model = define_model(feature_count, train_X.shape[1], mode=model_mode)

    # generate the callbacks
    my_callbacks = []

    # Write to TensorBoard so we can visualize how our training is going
    if tensorboard_on:
        log_dir = logs_root_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        my_callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1))
    
    # Add a checkpoint saver
    checkpoint = ModelCheckpoint(model_file,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min')

    my_callbacks.append(checkpoint)

    model.fit(train_X, train_y,
              validation_data=(test_X, test_y),
              epochs=epochs,
              verbose=2,
              callbacks=my_callbacks)


def main(train_file: "The training data file",
         test_file: "The validation data file",
         model_file: "Model file name",
         epochs: ("Training: number of epochs", 'option', 'e') = 200,
         logs_root_dir: ("Name of the TensorBoard log directory", 'option', 'l') = 'logs/fit/',
         model_mode: ("The type of model to use. Can be Vanilla, Stacked, or Bidirectional", 'option', 'm') = 'vanilla',
):
    train_X, train_y = utils.get_X_y(train_file)
    test_X, test_y = utils.get_X_y(test_file)

    # print(type(model_file))

    train(train_X, train_y,
          test_X, test_y,
          model_file=model_file,
          logs_root_dir=logs_root_dir,
          feature_count=1,
          epochs=epochs,
          model_mode=model_mode,
          tensorboard_on=True)


if __name__ == "__main__":
    import plac

    plac.call(main)
