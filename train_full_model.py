import os

import split
import make_train_test_split
import stitch
import train
import evaluate

def main(cluster: ("The cluster to train for", 'option', 'c') = None,
         model_mode: ("The type of model to use. Can be Vanilla, Stacked, or Bidirectional", 'option', 'm') = 'vanilla',
        ):

    if model_mode not in [ "vanilla", "stacked", "bidirectional" ]:
        raise TypeError(f"Unknown model type {model_mode}")

    clusters = {
        "1": [ "PYPL", "ZM" ],
        "2": [ "CSCO", "INTC" ],
        "3": [ "BBY", "TSLA", "XRX" ],
        "4": [ "V" ],
        "5": [ "ADBE", "AMZN", "ATVI", "CTAS", "FB", "MSFT", "NVDA" ],
        "6": [ "ALL", "DIS", "DOV", "HAS", "HD", "HSY", "JNJ", "MAR", "MCD", "MMM", "PEP", "WMT" ],
        "7": [ "AXP", "NFLX", "ORCL", "SBUX", "STX", "UAL", "VZ" ],
        "8": [ "GS" ],
        "9": [ "CMCSA", "CVS", "FDX", "GIS", "GPS", "IBM", "K", "MET", "T", "WHR" ],
        "10": [ "KHC", "UA" ],
        "11": [ "DOW", "FOXA" ],
        "12": [ "AAPL", "HPQ", "KO", "NKE", "TWTR" ]
    }

    stocks = None
    model_name = f"cluster_{cluster}"

    if cluster is not None:
        stocks = clusters[cluster]
    else:
        # Join all of the stocks into one list of stocks
        for clustername, stocklist in clusters.items():
            if stocks is None:
                stocks = stocklist
            else:
                stocks = [ *stocks, *stocklist ]

        cluster = "M_A"

    print("Training on the following stocks:", flush=True)
    print(stocks, flush=True)

    for stock in stocks:
        print("Downloading " + stock + "...", flush=True)

        # If the user doesn't already have the stock downloaded,
        # download it
        if not os.path.exists(f"./data/daily_{stock}.csv"):
            os.system(f"./download_data.sh {stock}")

    for stock in stocks:
        print("Restructuring " + stock + "...", flush=True)
        split.main(f"./data/daily_{stock}.csv", f"./data/{stock}", 5)

    for stock in stocks:
        print("Splitting " + stock + "...", flush=True)
        make_train_test_split.main(f"./data/{stock}.pkl", 20, 20, None)

    print("Stitching all of the stocks together...", flush=True)
    for data_type in [ "train", "val", "test" ]:

        all_stocks = list(map(
            lambda stock: f"./data/{stock}.pkl.{data_type}.pkl",
            stocks
        ))

        # all_stocks = ",".join(all_stocks)

        stitch.main(f"./data/{model_name}.{data_type}.pkl", *all_stocks)

    print("Training the model...", flush=True)
    epochs_count = 200

    train.main(f"./data/{model_name}.train.pkl",
               f"./data/{model_name}.val.pkl",
               f"./data/{model_name}_{epochs_count}epochs_{model_mode}.h5",
               epochs=epochs_count,
               logs_root_dir='logs/fit/',
               model_mode=model_mode)

    print("Training complete.", flush=True)
    
    print("Creating directory for evaluation figures...", flush=True)
    os.system(f"mkdir ./figures/")
    os.system(f"mkdir ./figures/{model_name}")
    print(f"Directory created. Find the figures in ./figures/{model_name} .", flush=True)

    print("Evaluating with 5 steps of blind prediction and linear regression...", flush=True)
    evaluate.main(f"./data/{model_name}_{epochs_count}epochs_{model_mode}.h5",
                  f"./data/{model_name}.train.pkl",
                  f"./data/{model_name}.val.pkl",
                  f"./data/{model_name}.test.pkl",
                  f"./figures/{model_name}/{model_name}_{epochs_count}epochs_{model_mode}",
                  True,
                  5,
                  True)

    print("Model pipeline complete.", flush=True)

if __name__ == "__main__":
    import plac

    plac.call(main)
