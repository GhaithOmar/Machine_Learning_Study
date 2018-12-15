import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
df = pd.read_csv('ecommerce_data.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(df['time_of_day'].unique())

onehotencoder = OneHotEncoder(categorical_features=[0])
X[:, -1] = onehotencoder.fit_transform(X[:, -1]).reshape(-1, 1)
print(X[:5])
