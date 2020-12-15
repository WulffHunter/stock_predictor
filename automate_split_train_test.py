import os
validate_stocks = "./data/"
train_stocks = "./data/"
test_stocks = "./data/"
files = os.listdir("data")
for f in files:
    if "test" in f:
        test_stocks += f + " ./data/"
    elif "val" in f:
        validate_stocks += f + " ./data/"
    elif "train" in f:
        train_stocks += f + " ./data/"
print(train_stocks)
print(test_stocks)
print(validate_stocks)