# CPS803 Stock Predictor

> By Jared Rand, Diwei Guan, and Gil Litvak

## How to get set up
Please note that all of the code is intended for a Unix or Unix-like environment. Additionally, this project requires the user to have Python 3 installed.

### Creating the venv
Run the following command in your favourite terminal:

```
source create_venv.sh
```

### Activating the venv
Run the following command in your favourite terminal:

```
source stock_predictor/bin/activate
```

### Downloading data
To download the dataset of a single stock, use the command `./download_data.sh` followed by the symbol of the stock you're trying to download. For example, in order to download the Apple Inc. stock (AAPL), run the following command:

```
./download_data.sh AAPL
```

This will automatically download the CSV file `daily_AAPL.csv` to a `./data` folder.

### Analyzing a stock

To analyze a downloaded stock, run the command `analyse_stock.py` with a stock symbol. For example, one could run:

```
python3 analyse_stock.py AAPL
```

All images would appear in the `./figures` directory. One could force the analysis to only use the first 100 days of stock information with the flag `-100`.

### Restructuring your data into model input and output

An important part of training any model is restructuring the data into the model's input (`X`) and output (`y`). To restructure the data, run the command `split.py` with the name of the CSV to be restructured and the desired name of the output file. For example, the command:

```
python3 split.py ./data/daily_AAPL.csv ./data/AAPL
```

would result in the creation of a file `./data/AAPL.pkl` that would contain both the input and output to the model.

### Creating a train/validation/test split

To create a train/validation/test split, use the command `make_train_test_split.py`. Splitting the AAPL stock into 60% training data, 20% validation data, and 20% test data would look like the following:

```
python3 make_train_test_split.py ./data/AAPL.pkl -v 20 -t 20
```

Where the `-v` and `-t` options represent the percentage of validation and test data respectively. This command will output the following files:

```
./data/AAPL.pkl.train.pkl
./data/AAPL.pkl.val.pkl
./data/AAPL.pkl.test.pkl
```

### Creating training data from multiple stocks

Once several stocks have been split into train, validation, and test sets, the datasets of each category can be combined into one using the `stitch.py` command. For example:

```
python3 stitch.py ./data/train_stitched.pkl ./data/AAPL.pkl.train.pkl ./data/F.pkl.train.pkl ./data/MSFT.pkl.train.pkl ./data/TSLA.pkl.train.pkl ./data/V.pkl.train.pkl
```

will stitch together the train files of the `AAPL`, `F`, `MSFT`, `TSLA`, and `V` files into a single `./data/train_stitched.pkl` file.

### Training a model

One can train a `vanilla` (single-LSTM), `stacked`, or `bidirectional` recurrent neural network model with the `train.py` command. For example:

```
python3 train.py ./data/train_stitched.pkl ./data/val_stitched.pkl ./models/stitched_100epoch_stacked.h5 -e 100 -m stacked
```

would train the model on `./data/train_stitched.pkl` and validate with `./data/val_stitched.pkl`. The model would be saved as `./models/stitched_100epoch_stacked.h5`. The `-e` option denotes how many epochs to run training for, while the `-m` command tells the script to train a model as described above.

### Evaluating a model

One can evaluate a model with the `evaluate.py` script:

```
python3 evaluate.py ./models/stitched_100epoch_stacked.h5 ./data/train_stitched.pkl ./data/val_stitched.pkl ../data/test_stitched.pkl -i ./figures/stitched_100epoch_stacked_images -lr -b -bi 5
```

runs evaluation on `./models/stitched_100epoch_stacked.h5`. The script outputs the R^2 scores for `./data/train_stitched.pkl`, `./data/val_stitched.pkl`, and `./data/test_stitched.pkl`. The `-i` option tells the script to save plots of the model predictions at `./figures/stitched_100epoch_stacked_images` (e.g. a file might be saved as `./figures/stitched_100epoch_stacked_images.test.png`). The `-lr` flag tells the script to compare against linear regression. The `-b` command tells the model to run blind prediction, and the `-bi` option tells the script to only run blind prediction for `5` iterations.

### Computing clusters of stocks

Once all stocks of interest have been downloaded, they can be split into clusters using the command `calculate_correlations.py` and passing in their stock symbols:

```
python3 calculate_correlations.py AAPL F V MSFT TSLA
```

To aid in this, one could print the symbols of all of the stocks in their `./data/` directory using the command:

```
./print_all_stocks.sh
```


## Automation

To make replicating results easier, automation scripts have been included:

```
python3 automate_split.py
```
runs the `split.py` and `make_train_test_split.py` scripts on all of the files in the `./data/` directory, and

```
python3 evaluate_automation.py
```
evaluates all of the cluster models in the `./cluster_data/` directory.