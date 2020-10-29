import utils

from numpy import array
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional

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


def train(X, y, feature_count=1, epochs=200, model_file, logs_root_dir, model_mode='vanilla'):
    reshaped_X = X.reshape((X.shape[0], X.shape[1], feature_count))

    model = define_model(feature_count, X.shape[1], mode=model_mode)

    # generate the callbacks
    my_callbacks = []

    # Write to TensorBoard so we can visualize how our training is going
    if tensorboard_on:
        log_dir = logs_root_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        my_callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1))
    
    # Add a checkpoint saver
    checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    my_callbacks.append(checkpoint)

    model.fit(X, y, epochs=200, verbose=2, callbacks=my_callbacks)


def main(in_file: "Input raw file in numpy binary format (use the first part of the name)",
         model_file: "Model file name",
         epochs: ("Training: number of epochs", 'option', 'e') = 200,
         logs_root_dir: ("Name of the TensorBoard log directory", 'option', 'l') = 'logs/fit/',
         model_mode: ("The type of model to use. Can be Vanilla, Stacked, or Bidirectional", 'option', 'm') = 'vanilla',
):
    X_y = utils.load_sequence(in_file + '.pkl')

    X = array(X_y[:, 0])
    y = array(X_y[:, 1])

    train(X, y,
          feature_count=1,
          epochs=200,
          model_file=model_file,
          logs_root_dir=logs_root_dir,
          model_mode=model_mode)


if __name__ == "__main__":
    import plac

    plac.call(main)
