import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('Salary_Data.csv')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
#y = y.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3)

# print(X_train)
# print(y_train)

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train, y_train)

# Predicting the Test set results
y_pred = reg.predict(X_test)

print(f'R_squared = {reg.score(X_test,y_test)}')
# Visualising the Training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, reg.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, reg.predict(X_test), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
