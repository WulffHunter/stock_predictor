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


def train(X, y, feature_count=1, epochs=200, model_file, logs_root_dir):
    reshaped_X = X.reshape((X.shape[0], X.shape[1], feature_count))

    define_model(feature_count, X.shape[1], mode='vanilla')

    # generate the callbacks
    my_callbacks = []
    if tensorboard_on:
        log_dir = logs_root_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        my_callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1))
    checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    my_callbacks.append(checkpoint)

    model.fit(X, y, epochs=200, verbose=2)


def main(in_file: "Input raw file in numpy binary format (use the first part of the name)",
         model_file: "Model file name",
         epochs: ("Training: number of epochs", 'option', 'e') = 200,
         logs_root_dir: ("Name of the TensorBoard log directory", 'option', 'l') = 'logs/fit/'
):
    X = utils.load_sequence(in_file + '.X.pkl')
    y = utils.load_sequence(in_file + '.y.pkl')

    train(X, y, 1)


if __name__ == "__main__":
    import plac

    plac.call(main)
