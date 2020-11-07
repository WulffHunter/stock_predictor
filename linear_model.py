import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import datetime
import os
import matplotlib.pyplot as plt


def main(stock):
    stock_filename = "./data/daily_" + stock + ".csv"
    df = pd.read_csv(stock_filename, index_col = 0)
    # print(df.head())
    
    df = pd.DataFrame(df, columns=['close'])
    df = df.reset_index()

    print(df.head())
    print(df.isna().values.any())
    train, test = train_test_split(df, test_size=0.20)

    X_train = np.array(train.index).reshape(-1, 1)
    y_train = train['close']
    model = LinearRegression()
    model.fit(X_train, y_train)

    # The coefficient
    print('Slope: ', np.asscalar(np.squeeze(model.coef_)))
    # The Intercept
    print('Intercept: ', model.intercept_)

    #original graph
    # plt.figure(1, figsize=(16,10))
    # plt.title('Linear Regression | Price vs Time')
    # plt.scatter(X_train, y_train, edgecolor='w', label='Actual Price')
    # plt.plot(X_train, model.predict(X_train), color='r', label='Predicted Price')
    # plt.xlabel('Integer Date')
    # plt.ylabel('Stock Price')
    # plt.legend()
    # plt.show()

    # prediction
    X_test = np.array(test.index).reshape(-1, 1)
    y_test = test['close']

    y_pred = model.predict(X_test)

    plt.figure(1, figsize=(16,10))
    plt.title('Linear Regression | Price vs Time')
    plt.plot(X_test, y_pred, color='r', label='Predicted Price')
    plt.scatter(X_test, y_test, edgecolor='w', label='Actual Price')

    plt.xlabel('Integer Date')
    plt.ylabel('Stock Price in $')

    plt.show()

if __name__ == "__main__":
    main("DOW")