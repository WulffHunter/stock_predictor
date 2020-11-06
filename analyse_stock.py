# Code only barely modified from https://www.kaggle.com/raoulma/ny-stock-price-prediction-rnn-lstm-gru/data?select=prices-split-adjusted.csv

import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt

def main(stock: "The training data file"):
    stock_filename = "./data/daily_" + stock + ".csv"

    df = pd.read_csv(stock_filename, index_col = 0)
    print(df.info())
    print(df.head())
    
    print(df.describe())


    plt.figure(figsize=(15, 5))
    plt.subplot(1,2,1)
    plt.plot(df.open.values, color='red', label='Open')
    plt.plot(df.close.values, color='green', label='Close')
    plt.plot(df.low.values, color='blue', label='Low')
    plt.plot(df.high.values, color='black', label='High')
    plt.title(stock + ' stock price')
    plt.xlabel('Time [days]')
    plt.ylabel('Price')
    plt.legend(loc='best')
    plt.show()

    plt.subplot(1,2,2)
    plt.plot(df.volume.values, color='black', label='Volume')
    plt.title(stock + ' stock volume')
    plt.xlabel('Time [days]')
    plt.ylabel('Volume')
    plt.legend(loc='best')
    plt.show()

    # function for min-max normalization of stock
    def normalize_data(df):
        min_max_scaler = sklearn.preprocessing.MinMaxScaler()
        df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))
        df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))
        df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))
        df['close'] = min_max_scaler.fit_transform(df['close'].values.reshape(-1,1))
        return df

    # Copy the stock
    df_stock = df.copy()
    df_stock.drop(['volume'],1,inplace=True)

    cols = list(df_stock.columns.values)
    print('df_stock.columns.values = ', cols)

    # normalize stock
    df_stock_norm = df_stock.copy()
    df_stock_norm = normalize_data(df_stock_norm)

    plt.figure(figsize=(15, 5))
    plt.plot(df_stock_norm.open.values, color='red', label='open')
    plt.plot(df_stock_norm.close.values, color='green', label='low')
    plt.plot(df_stock_norm.low.values, color='blue', label='low')
    plt.plot(df_stock_norm.high.values, color='black', label='high')
    #plt.plot(df_stock_norm.volume.values, color='gray', label='volume')
    plt.title(stock)
    plt.xlabel('Time [days]')
    plt.ylabel('Normalized price/volume')
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    import plac

    plac.call(main)
