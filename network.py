from neuron import Neuron
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from random import random

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
                layer.append(Neuron(np.array([0.0] * len(self.ind))))
            network.append(layer)
        
        self.network = network

    def get_info(self):

        for i in self.network:
            print(i)
            print(len(i))

    def evaluate(self, values):
        x = values
        for layer in self.amount_neurons_layers:
            for neuron in layer:
                neuron.values = x

            x = [neuron.get_activation_value() for neuron in layer]

        return x

    def train_network(self, iters, learning_rate):

        self.w = [random() for i in range(len(self.dep))]
        x_train, self.x_test, y_train, self.y_test = self.get_training_test(self.data)
        data = x_train
        data.join(y_train)

        for iteration in range(iters):
            
            for _, row in data.iterrows():
                
                values = row[self.ind]
                result = row[self.dep]

                h = self.evaluate(values)
                delta_j = [neuron.activation_function_derivate() for neuron in self.amount_neurons_layers[-1]]
                
                for i in range(len(delta_j)):
                    delta_j[i] = delta_j[i] * (result[i] - h[i])

                deltas = [delta_j]

                # Backpropagation
                for i in range(len(self.amount_neurons_layers)-1,-1,-1):

                    layer = self.amount_neurons_layers[i]
                    next_layer = self.amount_neurons_layers[i+1]
                    
                    delta = []

                    for neuron in layer:
                        d = 0.0
                        for k in range(len(next_layer)):
                            d += sum(next_layer[k].weights)*deltas[-1][k]

                        delta.append(d*neuron.activation_function_derivate())

                    deltas.append(delta)

                # Weights update
                for layer in self.amount_neurons_layers:
                    for delta in reversed(deltas):
                        for i in range(len(layer)):
                            neuron = layer[i]
                            for j in range(len(neuron.weights)):
                                neuron.weights[j] += (learning_rate * neuron.get_activation_value() * delta[i])