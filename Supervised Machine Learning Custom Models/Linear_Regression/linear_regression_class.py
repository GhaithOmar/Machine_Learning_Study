import numpy as np


class LinearRegression():
    """ This is a custom Linear Regression class applying the theory,
            and using the following equations:
            w = (X.T * X)^-1 * X.T*Y
            # w is the weight
            # X.T mean transpose(X)

            yhat = X * w
            # yhat is the prediction

            Rsquared = 1 - SSres/SStot
            SSres = (Y-Yhat)^2
            SStot = (Y-mean(Y))

            Notes:
            X shape is N X D
            Y shape is N X 1

            The methods that you can use:
            LinearRegression.fit(X,Y)
            # this method will train the model and store the final weight
            # will automaticly store X and Y and it will add a column of ones as the first column in X
            # then will calculate the final w that will be used for predicting using the equation above

            LogisticRegression.predict(X)
            # this method use the (w) that optained from .fit
            # .fit necessary to use predict
            # and will predict the value of X using both the stored weight (w)  and the yhat equation above

            LogisticRegression.score(X,Y)
            # pass the X you want to predict or test and the actual value Y  ,after training the model using .fit
            # it will predict the value of X and then use R squared method to calculate the accuracy
    """

    def fit(self, X, Y):
        N = X.shape[0]
        X = np.append(arr=np.ones((N, 1)).astype(int), values=X, axis=1)
        self.w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))

    def predict(self, X):
        N = X.shape[0]
        X = np.append(arr=np.ones((N, 1)).astype(int), values=X, axis=1)
        return X.dot(self.w)

    def score(self, X, Y):
        Y_hat = self.predict(X)
        d1 = Y - Y_hat
        d2 = Y - Y.mean()
        SSres = d1.dot(d1)
        SStot = d2.dot(d2)
        return 1 - (SSres / SStot)
