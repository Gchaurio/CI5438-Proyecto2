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
        self.n_layers = len(neuron_layers) + 1
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

        ind = len(self.ind)
        input_layer = [Neuron(np.array([random()] * ind)) for i in range(self.amount_neurons_layers[0])]
        network = [input_layer]

        for i in range(1, self.n_layers):
            layer = [Neuron(np.array([random()] * len(network[-1]))) for n in range(self.amount_neurons_layers[i-1])]
            network.append(layer)

        self.network = network

    def evaluate(self, values):
        x = values
        for layer in self.network:
            for neuron in layer:
                neuron.values = x

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

                error = [result[i] - h[i] for i in range(len(result))]

                self.error_medio.append(np.mean(error))
                self.error_maximo.append(np.max(error))
                self.error_minimo.append(np.min(error))

                delta_j = [neuron.activation_function_derivate() for neuron in self.network[-1]]
                delta_j = [delta_j[i] * error[i] for i in range(len(delta_j))]

                deltas = [delta_j]
                
                # Backpropagation
                for l in range(len(self.network)-1,0,-1):
                    layer = self.network[l-1]
                    next_layer = self.network[l]
                    
                    delta = []

                    for i in range(len(layer)):
                        neuron = layer[i]
                        d = 0.0
                        for k in range(len(next_layer)):
                            d += (next_layer[k].weights[i])*deltas[-1][k]

                        # delta[i] = 
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



                            