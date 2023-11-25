import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from network import Network
import numpy as np

def load_column_names(file_path):
    with open(file_path, "r") as f:
        column_names = f.readlines()
    names = [name.split(":")[0] for name in column_names[33:]]
    names.append("spam")
    return names

def normalize_data(data: pd.DataFrame):
    
    df = []
    for column in data.columns:
        mn = data[column].min()
        mx = data[column].max()
        df.append(data[column].map(lambda x: (x - mn) / (mx - mn)))

    return pd.DataFrame(df)

def load_data(file_path, column_names):
    data = pd.read_csv(file_path, names=column_names)
    return normalize_data(data)

def main():
    column_names = load_column_names("./spam/spambase.names")
    data = load_data("./spam/spambase.data", column_names)

    network = Network(data, data.columns[:-1], [data.columns[-1]], [114, 1])
    network.train_network(1000, 0.1)

if __name__ == '__main__':
    main()