import pandas as pd
from network import Network
import matplotlib.pyplot as plt

dep = ['Iris-setosa','Iris-versicolor','Iris-virginica']
ind = ['sepal_length','sepal_width','petal_length','petal_width']
df = pd.read_csv("iris.csv")
#posible_class = len(df['species'].unique())
n = Network(df, ind, dep, [10, 10, len(dep)])
n.form_network()
n.train_network(500, 0.01)