import pandas as pd 
import numpy as np

class Neuron(object):
    
    def __init__(self, weights):

        self.weights = weights
        self.values = None
        self.activation_value = None

    
    # Calcula valores 
    def calculate_values(self):
        
        return np.dot(self.values, self.weights)

    # Funcion logistica
    def activation_function(self, z):

        return 1 / (1 + np.exp(-z))
    
    # Derivada de funcion logistica
    def activation_function_derivate(self):

        return self.get_activation_value()*(1-self.get_activation_value())
    
    # Retorna el valor de activacion de la neurona
    def get_activation_value(self):

        self.activation_value = self.activation_function(self.calculate_values())

        return self.activation_value
    



