import os, sys

csv_files = []
pkl_files = []

files = os.listdir("data")
for f in files:
    if f.endswith(".csv"):
        csv_files.append(f)
for f in csv_files:
    os.system(f"python3 split.py ./data/{f} ./data/{f[6:len(f)-4]}")
new_files = os.listdir("data")
for f in new_files:
    if f.endswith(".pkl"):
        pkl_files.append(f)
for f in pkl_files:
    os.system(f"python3 make_train_test_split.py ./data/{f} -t 20 -v 20")