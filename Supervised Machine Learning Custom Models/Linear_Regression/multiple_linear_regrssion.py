import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from linear_regression_class import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('..\\Data\\50_Startups.csv')

#df['Ones']=  1
df['NewYork'] = df['State'].apply(lambda x: 1 if x == 'New York' else 0)
X = df[['R&D Spend', 'Administration', 'Marketing Spend', 'NewYork']].values
Y = df['Profit'].values


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.25)


ml = LinearRegression()

ml.fit(X_train, Y_train)

yhat = ml.predict(X_test)

print(ml.score(X_test, Y_test))
