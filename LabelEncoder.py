import pandas as pd
import numpy as np
from urllib.request import urlopen
from sklearn.preprocessing import LabelEncoder

url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

data=pd.read_csv(url, names=["sepal_length", "sepal_width", "petal_length", "petal_width","class"])
print(data)

X=data.drop('class', axis=1)
Y=data['class']
print(Y)

transform=LabelEncoder()
Y=transform.fit_transform(Y)
print(Y)
