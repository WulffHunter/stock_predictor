#!/bin/bash

# To run this script: ./download_data.sh "DOW"

# Create the data directory if it doesn't exist
mkdir -p ./data/

# Default the symbol to the Dow-Jones index
SYMBOL=${1:-"DOW"}

# Download the CSV into the `data` directory
wget -P ./data/ "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=${SYMBOL}&outputsize=full&apikey=250U66DFYADEDPQM&datatype=csv"