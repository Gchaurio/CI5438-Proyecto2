import pandas as pd 
import numpy as np

class Neuron(object):
    
    def __init__(self, weights):

        self.weights = weights
        self.values = None

    def calculate_values(self):

        return np.dot(self.values, self.weights)

    def activation_function(self, z):

        return 1 / (1 + np.exp(-z))
    
    def activation_function_derivate(self):

        return self.get_activation_value()*(1-self.get_activation_value())
    
    def get_activation_value(self):

        return self.activation_function(self.calculate_values())


