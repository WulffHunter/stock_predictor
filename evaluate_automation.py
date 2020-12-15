import os
models = os.listdir("cluster_models")
for model in models:
    number = model.split("_")[0]
    print(number)
    os.system(f"python3 evaluation.py ./cluster_models/{model} ./cluster_data/{number}_train.pkl ./cluster_data/{number}_val.pkl ./cluster_data/{number}_test.pkl -i ./plot_figures/{number}")