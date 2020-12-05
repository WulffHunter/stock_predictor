import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import datetime
import os
import matplotlib.pyplot as plt
from pickle import load
import utils
from sklearn.metrics import r2_score

def accuracy(y, y_pred):
    return sum(
        y_pred[i] == y[i] for i, prediction in enumerate(y_pred)
    ) / y_pred.shape[0]

def reshape_data(X, feature_count=1):
    return X.reshape((X.shape[0], X.shape[1], feature_count))

def get_set_predictions(model, set_to_predict):
    y_pred = list()

    y_pred = model.predict(set_to_predict)

    # for i in range(set_to_predict.shape[0]):
    #     X = reshape_data(np.array([set_to_predict[i, :]]))

    #     y_pred.append(model.predict(X)[0][0])

    return np.array(y_pred)

def predict_from_seed(model, seed_set):
    y_pred = list()

    # Make the starting seed the earliest available number
    seed = list(seed_set[0, :])
    for i in range(seed_set.shape[0]):
        X = reshape_data(np.array([seed]))
        # Predict from the current seed set
        prediction = model.predict(X)

        #Add the prediction to the list of predictions
        y_pred.append(prediction)

        # Remove the oldest element from the seed
        seed.pop(0)
        # Add the prediction as the newest element observed
        seed.append(prediction)

    return np.array(y_pred)


def main():
    # stock_filename = "./data/daily_" + stock + ".csv"
    train_filename = "./data/stitch_train.pkl"
    test_filename = "./data/stitched_test.pkl"
    valid_filename = "./data/stitched_val.pkl"
    #df = pd.read_pickle(train_filename)
    # print(df)
    
    # df = pd.DataFrame(df, columns=['close'])
    # df = df.reset_index()

    # print(df.head())
    # print(df.isna().values.any())
    # train, test = train_test_split(df, test_size=0.20)

    # X_train = np.array(train.index).reshape(-1, 1)
    # y_train = train['close']
    X_train, y_train = utils.get_X_y(train_filename)
    #print(X_train[0])

    model = LinearRegression()
    model.fit(X_train, y_train)

   

   


    # The coefficient
    #print('Slope: ', np.asscalar(np.squeeze(model.coef_)))
    #print('slope:', model.coef_)
    # # The Intercept
    print('Intercept: ', model.intercept_)

    # #original graph
    # # plt.figure(1, figsize=(16,10))
    # # plt.title('Linear Regression | Price vs Time')
    # # plt.scatter(X_train, y_train, edgecolor='w', label='Actual Price')
    # # plt.plot(X_train, model.predict(X_train), color='r', label='Predicted Price')
    # # plt.xlabel('Integer Date')
    # # plt.ylabel('Stock Price')
    # # plt.legend()
    # # plt.show()

    # # prediction
    # X_test = np.array(test.index).reshape(-1, 1)
    # y_test = test['close']

    X_test, y_test = utils.get_X_y(test_filename)
    valid_x, valid_y = utils.get_X_y(valid_filename)

    y_pred = model.predict(X_test)

    print(r2_score(y_test, y_pred))

    # plt.figure(1, figsize=(16,10))
    # plt.title('Linear Regression | Price vs Time')
    # plt.plot(X_test, y_pred, color='r', label='Predicted Price')
    # plt.scatter(X_test, y_test, edgecolor='w', label='Actual Price')

    # plt.xlabel('Integer Date')
    # plt.ylabel('Stock Price in $')

    # plt.show()



    train_pred = get_set_predictions(model, X_train)
    train_accuracy = accuracy(y_train, train_pred)
    # train_precision = sklearn.metrics.precision_score(y_train, train_pred)
    train_r2 = sklearn.metrics.r2_score(y_train, train_pred)
    print(f"Train set accuracy: {train_accuracy}")
    # print(f"Train set precision: {train_precision}")
    print(f"Train set R^2 score: {train_r2}")

    valid_pred = get_set_predictions(model, valid_x)
    valid_accuracy = accuracy(valid_y, valid_pred)
    # valid_precision = sklearn.metrics.precision_score(valid_y, valid_pred)
    valid_r2 = sklearn.metrics.r2_score(valid_y, valid_pred)
    print(f"Validation set accuracy: {valid_accuracy}")
    # print(f"Validation set precision: {valid_precision}")
    print(f"Validation set R^2 score: {valid_r2}")

    test_pred = get_set_predictions(model, X_test)
    test_accuracy = accuracy(y_test, test_pred)
    # test_precision = sklearn.metrics.precision_score(y_test, test_pred)
    test_r2 = sklearn.metrics.r2_score(y_test, test_pred)
    print(f"Test set accuracy: {test_accuracy}")
    # print(f"Test set precision: {test_precision}")
    print(f"Test set R^2 score: {test_r2}")

    blind_pred = predict_from_seed(model, X_test)
    blind_accuracy = accuracy(y_test, blind_pred)
    # blind_precision = sklearn.metrics.precision_score(y_test, blind_pred)
    blind_r2 = sklearn.metrics.r2_score(y_test, blind_pred)
    print(f"Blind prediction accuracy: {blind_accuracy}")
    # print(f"Blind set precision: {blind_precision}")
    print(f"Blind set R^2 score: {blind_r2}")



if __name__ == "__main__":
    main()