import pandas as pd
from network import Network

dep = ['Iris-setosa','Iris-versicolor','Iris-virginica']
ind = ['sepal_length','sepal_width','petal_length','petal_width']
df = pd.read_csv("iris.csv")
n = Network(df, ind, dep, [len(ind), 5, 4, len(dep)])
n.train_network(3000, 0.1)
