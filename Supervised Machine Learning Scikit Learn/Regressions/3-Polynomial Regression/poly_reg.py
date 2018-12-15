import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Position_Salaries.csv')

X = df.iloc[:, 1:2].values
Y = df.iloc[:, 2].values


# fiiting linear regression

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X, Y)

# Fitting Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)

X_poly = poly_reg.fit_transform(X)


lr2 = LinearRegression()
lr2.fit(X_poly, Y)

####################################################
# Predection
#######################################################


yhat1 = lr.predict(X)
yhat2 = lr2.predict(poly_reg.fit_transform(X))

print(f"Score for linear regression: {lr.score(X,Y)}")
print(f"Score for polynomial_regression: {lr2.score(poly_reg.fit_transform(X),Y)}")

plt.scatter(X, Y, color='red')
plt.plot(X, yhat1, color='blue', label='Linear Regression')
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.plot(X, yhat2, color='green',label='polynomial_regression')
plt.legend()
plt.show()

print(lr2.predict(poly_reg.fit_transform(6.5)))
