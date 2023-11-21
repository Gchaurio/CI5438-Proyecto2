from neuron import Neuron
import pandas as pd
from sklearn.model_selection import train_test_split

class Network(object):

    def __init__(self, data: pd.DataFrame, ind, dep, neuron_layers):

        self.amount_neurons_layers = neuron_layers
        self.n_layers = len(neuron_layers)
        self.ind = ind
        self.dep = dep
        self.data = data
        self.network = None
    
    def get_training_test(self, data: pd.DataFrame):

        x_train, x_test, y_train, y_test = train_test_split(data[self.ind], data[self.dep], test_size=0.2, random_state=42)

        return x_train, x_test, y_train, y_test


    def form_network(self):

        network = []

        for i in range(self.n_layers):
            layer = []
            for j in range (self.amount_neurons_layers[i]):
                layer.append(Neuron([0] * len(self.ind)))
            network.append(layer)
        
        self.network = network

    def get_info(self):

        for i in self.network:
            print(i)
            print(len(i))

        






