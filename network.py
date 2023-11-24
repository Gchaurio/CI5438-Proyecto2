from neuron import Neuron
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from random import random
import matplotlib.pyplot as plt
import os


class Network(object):

    def __init__(self, data: pd.DataFrame, ind, dep, neuron_layers):

        self.amount_neurons_layers = neuron_layers
        self.n_layers = len(neuron_layers)
        self.ind = ind
        self.dep = dep 
        self.data = data
        self.network = None
        self.test_y = None
        self.test_x = None
        self.error_medio = []
        self.error_maximo = []
        self.error_minimo = []
        self.form_network()
    
    def get_training_test(self, data: pd.DataFrame):

        x_train, x_test, y_train, y_test = train_test_split(data[self.ind], data[self.dep], test_size=0.2, random_state=42)

        self.y_test = y_test
        self.x_test = x_test

        return x_train, x_test, y_train, y_test
    
    def test(self):
        '''
        Funcion que evalua los datos del conjunto de pruebas
        luego de entrenar el modelo
        '''
        df = pd.DataFrame(columns=['pred', 'result'])
        
        # Se combinan los datos de prueba en un solo DataFrame
        data = pd.concat([self.x_test,self.y_test],axis=1)

        for _, row in data.iterrows():

            values = row[self.ind]
            result = row[self.dep]

            # Evaluacion
            prediction = self.evaluate(values)

            max_index = np.argmax(prediction)

            result_positive_index = np.argmax(result)

            if max_index == result_positive_index:
                df = df.append({
                'pred': prediction[max_index],
                'result': result[result_positive_index],
            }, ignore_index=True)
            else: 
                df = df.append({
                'pred': "Incorrect classification",
                'result': result[result_positive_index],
            }, ignore_index=True)

        return df


    def form_network(self):

        network = []

        for i in range(self.n_layers):
            layer = []
            for j in range (self.amount_neurons_layers[i]):
                
                # Generate an array of random numbers between 0 and 1 with 3 decimal places
                weights = np.random.uniform(low=0.0, high=1.0, size= len(self.ind))
                # Round the weights to 3 decimal places
                weights = np.round(weights, decimals=3)

                # Append the weights to the layer
                layer.append(Neuron(weights))
                
            network.append(layer)
        
        self.network = network

    def evaluate(self, values):

        i = 0
        for layer in self.network:
            if i == 0:
                for neuron in layer:
                    neuron.values = values
                x = [neuron.get_input_layer_value() for neuron in layer]
                i +=1
            else:
                for neuron in layer:
                    neuron.values = values
                x = [neuron.get_activation_value() for neuron in layer]

        return x
    
    # def error_data(self):

    #     '''
    #     Grafico del error de cada iteracion
    #     '''
    #     k = len(self.error_medio)
    #     plt.xlabel("Iteraciones")
    #     plt.ylabel("Error")
    #     plt.plot(self.convergencia, self.errores[self.convergencia], c='red', marker='o')
    #     plt.plot(range(int(k)), self.errores[:int(k)])
    #     plt.savefig(os.path.join("graficos", name))
    #     plt.show()


    def train_network(self, iters, learning_rate):

        x_train, self.x_test, y_train, self.y_test = self.get_training_test(self.data)
        
        data = x_train
        data = data.join(y_train)

        self.error_medio = []
        self.error_maximo = []
        self.error_minimo = []
    
        for iteration in range(iters):

            print(iteration)
        
            for _, row in data.iterrows():

                values = row[self.ind]
                result = row[self.dep]

                h = self.evaluate(values)

                error = result - h

                self.error_medio.append(np.mean(error))
                self.error_maximo.append(np.max(error))
                self.error_minimo.append(np.min(error))

                delta_j = [neuron.activation_function_derivate() for neuron in self.network[-1]]
                
                
                for i in range(len(delta_j)):

                    delta_j[i] = delta_j[i] * (result[i] - h[i])

                deltas = [delta_j]
                
                # Backpropagation
                for i in range(len(self.network)-1,0,-1):
                    print(i)
                    layer = self.network[i-1]
                    next_layer = self.network[i]
                    
                    delta = []

                    for neuron in layer:
                        d = 0.0
                        for k in range(len(next_layer)):
                            d += sum(next_layer[k].weights)*deltas[-1][k]

                        delta.append(d*neuron.activation_function_derivate())

                    deltas.append(delta)
                
                # Weights update
                for l in range(len(self.network)):
                    layer = self.network[l]
                    delta = deltas[-(1+l)]
                    for i in range(len(layer)):
                        neuron = layer[i]
                        for j in range(len(neuron.weights)):
                            neuron.weights[j] += (learning_rate * neuron.get_activation_value() * delta[i])



                            