# Code only barely modified from https://www.kaggle.com/raoulma/ny-stock-price-prediction-rnn-lstm-gru/data?select=prices-split-adjusted.csv

import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt

def main(stock: "The training data file",
         show_plots: ("Whether or not to display the plots to the users", 'flag', 's') = False,
         only_100_days: ("Only use 100 days worth of stock info for most of the figures", 'flag', '100') = False):
    stock_filename = "./data/daily_" + stock + ".csv"
    figures_dir = "figures/"

    if not os.path.exists(figures_dir):
        os.mkdir(figures_dir)

    df = pd.read_csv(stock_filename, index_col = 0)
    df = df.sort_index(axis=0, ascending=True)
    print(df.info())
    print(df.head())
    
    print(df.describe())

    df_100_days = df.head(100)

    curr_datafraome = df

    if only_100_days:
        curr_dataframe = df_100_days

    # Display figures regarding the opening, closing, low, and high per day
    plt.figure(figsize=(10, 5))
    plt.plot(curr_dataframe.open.values, color='red', label='Open')
    plt.plot(curr_dataframe.close.values, color='green', label='Close')
    plt.plot(curr_dataframe.low.values, color='blue', label='Low')
    plt.plot(curr_dataframe.high.values, color='black', label='High')
    plt.title(stock + ' stock price')
    plt.xlabel('Time [days]')
    plt.ylabel('Price')
    plt.legend(loc='best')
    plt.savefig(figures_dir + "/" + stock + "_changes.png")
    
    if show_plots:
        plt.show()

    plt.clf()

    # Display figures regarding the volume
    plt.plot(curr_dataframe.volume.values, color='black', label='Volume')
    plt.title(stock + ' stock volume')
    plt.xlabel('Time [days]')
    plt.ylabel('Volume')
    plt.legend(loc='best')
    plt.savefig(figures_dir + "/" + stock + "_volume.png")
    
    if show_plots:
        plt.show()

    plt.clf()

    # function for min-max normalization of stock
    def normalize_data(df):
        min_max_scaler = sklearn.preprocessing.MinMaxScaler()
        df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))
        df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))
        df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))
        df['close'] = min_max_scaler.fit_transform(df['close'].values.reshape(-1,1))
        return df

    # Copy the stock
    df_stock = curr_dataframe.copy()
    df_stock.drop(['volume'],1,inplace=True)

    cols = list(df_stock.columns.values)
    print('df_stock.columns.values = ', cols)

    # normalize stock
    df_stock_norm = df_stock.copy()
    df_stock_norm = normalize_data(df_stock_norm)

    # Display figures regarding the normalized price and volume
    plt.figure(figsize=(10, 5))
    plt.plot(df_stock_norm.open.values, color='red', label='open')
    plt.plot(df_stock_norm.close.values, color='green', label='low')
    plt.plot(df_stock_norm.low.values, color='blue', label='low')
    plt.plot(df_stock_norm.high.values, color='black', label='high')
    #plt.plot(df_stock_norm.volume.values, color='gray', label='volume')
    plt.title(stock)
    plt.xlabel('Time [days]')
    plt.ylabel('Normalized price/volume')
    plt.legend(loc='best')
    plt.savefig(figures_dir + "/" + stock + "_normalized.png")
    
    if show_plots:
        plt.show()

    plt.clf()

    # Display histograms of the stock information using all of the data
    df.hist(figsize=(12, 12))
    plt.title(stock + ' histograms')
    plt.savefig(figures_dir + "/" + stock + "_histograms.png")
    
    if show_plots:
        plt.show()

    # Display figures regarding the moving average
    ma_day = [10, 20, 50]

    df_stock_ma = curr_dataframe.copy()

    ma_cols = list()
    ma_cols.append('close')

    for ma in ma_day:
        column_name = f"MA for {ma} days"
        ma_cols.append(column_name)
        df_stock_ma[column_name] = df_stock_ma['close'].rolling(ma).mean()

    plt.figure(figsize=(10, 5))
    df_stock_ma[ma_cols].plot()
    plt.title(stock + ' moving averages')
    plt.savefig(figures_dir + "/" + stock + "_moving_average.png")
    
    if show_plots:
        plt.show()

    plt.clf()

    plt.figure(figsize=(10, 5))
    df_stock_ma['Daily Return'] = df_stock_ma['close'].pct_change()
    df_stock_ma['Daily Return'].plot(legend=True, linestyle='--', marker='o')
    plt.title(stock + ' daily return')
    plt.savefig(figures_dir + "/" + stock + "_daily_return.png")
    
    if show_plots:
        plt.show()

    plt.clf()

    df.hist(figsize=(12, 12))
    df_stock_ma['Daily Return'].hist()
    plt.title(stock + ' daily return histogram')
    plt.savefig(figures_dir + "/" + stock + "_daily_return_histogram.png")
    
    if show_plots:
        plt.show()

    plt.clf()

if __name__ == "__main__":
    import plac

    plac.call(main)
