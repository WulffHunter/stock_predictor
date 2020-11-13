import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as spc

def main(*stocks: "The stocks to calculate correlation groups on."):
    print(f"Clustering stocks {stocks}")

    stocks_dataframe = pd.DataFrame()

    for stock in stocks:
        stock_filename = "./data/daily_" + stock + ".csv"
        df = pd.read_csv(stock_filename, index_col=0)
        # Add the close value to the DataFrame of all stocks
        stocks_dataframe[stock] = pd.Series(df['close'].values)

    # https://stackoverflow.com/questions/52787431/create-clusters-using-correlation-matrix-in-python

    # Calculate correlations based on close
    corr = stocks_dataframe.corr().values

    pdist = spc.distance.pdist(corr)
    linkage = spc.linkage(pdist, method='complete')
    cluster_indexes = spc.fcluster(linkage, 0.5 * pdist.max(), 'distance')

    clusters = {}
    for index, stock in enumerate(stocks):
        if cluster_indexes[index] not in clusters:
            clusters[cluster_indexes[index]] = list()

        clusters[cluster_indexes[index]].append(stock)

    for key, value in sorted(clusters.items()):
        print (("Cluster {} --> {}").format(key, clusters[key]))

if __name__ == "__main__":
    import plac

    plac.call(main)
