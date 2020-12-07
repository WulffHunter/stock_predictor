import utils
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.linear_model import LinearRegression

def accuracy(y, y_pred):
    return sum(
        y_pred[i] == y[i] for i, prediction in enumerate(y_pred)
    ) / y_pred.shape[0]

def reshape_data(X, feature_count=1):
    return X.reshape((X.shape[0], X.shape[1], feature_count))

def get_set_predictions(model, set_to_predict, is_lr_model=False):
    y_pred = list()

    # y_pred = model.predict(set_to_predict, verbose=0)[0]

    for i in range(set_to_predict.shape[0]):
        if is_lr_model:
            X = np.array([set_to_predict[i, :]])

            y_pred.append(model.predict(X)[0][0])
        else:
            X = reshape_data(np.array([set_to_predict[i, :]]))

            y_pred.append(model.predict(X, verbose=0)[0][0])

    return np.array(y_pred)

def predict_from_seed(model, seed_set):
    y_pred = list()

    # Make the starting seed the earliest available number
    seed = list(seed_set[0, :])
    for i in range(seed_set.shape[0]):
        X = reshape_data(np.array([seed]))
        # Predict from the current seed set
        prediction = model.predict(X, verbose=0)[0][0]

        #Add the prediction to the list of predictions
        y_pred.append(prediction)

        # Remove the oldest element from the seed
        seed.pop(0)
        # Add the prediction as the newest element observed
        seed.append(prediction)

    return np.array(y_pred)


def main(model_file: "The filename of the trained model",
         train_file: "The filename of the data used for training the model",
         valid_file: "The filename of the data used for validating the model",
         test_file: "The filename for data that the model has not yet seen",
         plot_filename: ("The filename of the data plot", 'option', 'i') = None,
         use_blind: ("Use blind prediction", 'flag', 'b') = False,
         use_linear_regression: ("Train and plot a linear regression model", 'flag', 'lr') = False):
    train_x, train_y = utils.get_X_y(train_file)
    valid_x, valid_y = utils.get_X_y(valid_file)
    test_x, test_y = utils.get_X_y(test_file)

    model = load_model(model_file)

    train_pred = get_set_predictions(model, train_x)
    # train_accuracy = accuracy(train_y, train_pred)
    # train_precision = sklearn.metrics.precision_score(train_y, train_pred)
    train_r2 = sklearn.metrics.r2_score(train_y, train_pred)
    # print(f"Train set accuracy: {train_accuracy}")
    # print(f"Train set precision: {train_precision}")
    print(f"Train set R^2 score: {train_r2}")

    valid_pred = get_set_predictions(model, valid_x)
    # valid_accuracy = accuracy(valid_y, valid_pred)
    # valid_precision = sklearn.metrics.precision_score(valid_y, valid_pred)
    valid_r2 = sklearn.metrics.r2_score(valid_y, valid_pred)
    # print(f"Validation set accuracy: {valid_accuracy}")
    # print(f"Validation set precision: {valid_precision}")
    print(f"Validation set R^2 score: {valid_r2}")

    test_pred = get_set_predictions(model, test_x)
    # test_accuracy = accuracy(test_y, test_pred)
    # test_precision = sklearn.metrics.precision_score(test_y, test_pred)
    test_r2 = sklearn.metrics.r2_score(test_y, test_pred)
    # print(f"Test set accuracy: {test_accuracy}")
    # print(f"Test set precision: {test_precision}")
    print(f"Test set R^2 score: {test_r2}")

    if use_blind:
        blind_pred = predict_from_seed(model, test_x)
        # blind_accuracy = accuracy(test_y, blind_pred)
        # blind_precision = sklearn.metrics.precision_score(test_y, blind_pred)
        blind_r2 = sklearn.metrics.r2_score(test_y, blind_pred)
        # print(f"Blind prediction accuracy: {blind_accuracy}")
        # print(f"Blind set precision: {blind_precision}")
        print(f"Blind set R^2 score: {blind_r2}")

    if use_linear_regression:
        lr_model = LinearRegression()
        lr_model.fit(train_x, train_y)
        lr_pred = get_set_predictions(lr_model, test_x, True)
        lr_r2 = sklearn.metrics.r2_score(test_y, lr_pred)
        print(f"Linear Regression set R^2 score: {lr_r2}")

    # [array([91.44, 89.62, 96.37, 88.31, 83.62]) 81.5]
    # [array([89.62, 96.37, 88.31, 83.62, 81.5 ]) 80.25]
    # [array([96.37, 88.31, 83.62, 81.5 , 80.25]) 77.62]]

    if plot_filename is not None:
        plt.figure(figsize=(15, 5))

        plt.plot(train_y, color="blue", label='train real values')
        plt.plot(train_pred, color="red", label='train predictions')
        plt.savefig(plot_filename + ".train.png")
        plt.title('Train data predictions')
        plt.xlabel('Time [days]')
        plt.ylabel('Close Price')
        plt.legend(loc='best')
        plt.clf()

        plt.plot(valid_y, color="blue", label='validation real values')
        plt.plot(valid_pred, color="red", label='validation predictions')
        plt.savefig(plot_filename + ".valid.png")
        plt.title('Validation data predictions')
        plt.xlabel('Time [days]')
        plt.ylabel('Close Price')
        plt.legend(loc='best')
        plt.clf()

        plt.plot(test_y, color="blue", label='test real values') 
        plt.plot(test_pred, color="red", label='test predictions')

        if use_blind:
            plt.plot(blind_pred, color="green", label='blind predictions')
        
        if use_linear_regression:
            plt.plot(lr_pred, color="orange", label='linear regression predictions')

        plt.savefig(plot_filename + ".test.png")
        plt.title('Test data predictions')
        plt.xlabel('Time [days]')
        plt.ylabel('Close Price')
        plt.legend(loc='best')
        plt.clf()

        

if __name__ == "__main__":
    import plac

    plac.call(main)
