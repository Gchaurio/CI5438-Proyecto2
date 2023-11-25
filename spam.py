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

def load_data(file_path, column_names):
    X = pd.read_csv(file_path, names=column_names)
    y = X["spam"]
    X.drop("spam", axis=1, inplace=True)
    return X, y

def train_test_split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

def create_neural_network(input_size):
    hidden_layer_size = 57  # Replace 'capa' with the actual value
    output_size = 1
    return Network(input_size, hidden_layer_size, output_size)

def train_neural_network(neural_network, X_train, y_train, iterations, learning_rate, tolerance):
    return neural_network.train(X_train.values, y_train.values, iterations, learning_rate, tolerance)

def plot_network_error(neural_network, errors, save=True, filename="mse.png", title="Network Error for Spam Dataset"):
    if save:
        neural_network.plot_error(len(errors), errors, save, filename, title)

def evaluate_model(neural_network, X_test, y_test):
    y_pred = neural_network.feedforward(X_test)
    y_pred_class = [round(i[0]) for i in y_pred]
    neural_network.calculate_precision(y_pred_class, y_test)
    neural_network.calculate_false_positive_and_false_negative(y_pred_class, y_test)
    neural_network.show_info()

def main():
    column_names = load_column_names("./spam/spambase.names")
    X, y = load_data("./spam/spambase.data", column_names)

    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    iterations, learning_rate = 10000, 0.1, 

    print(X_train.columns, pd.DataFrame(y_train).columns)

    network = Network(X_train.join(y_train), X_train.columns, pd.DataFrame(y_train).columns, [114, 57, 1])
    network.train_network(1000, 0.1)

if __name__ == "__main__":
    main()
