import numpy as np


class LogisticRegression():
    """ This is a custom Logistic Regression class applying the theory,
            and using the following equations:
            sigma = 1/(1+exp(-z))
            z = w.T * X  (where w.T is the weight vector transposed and X is the matrix )
            in python z :
            z = X.dot(w)
            also using gradient descent to solve the weight
            w = w - learning_rate *dj/dw
            dj/dw = X.T(Y-T)
            #Y is the predicited values
            #T is the actual values

            Notes:
            X shape is N X D
            Y shape is N X 1
            T shape is N X 1

            The methods that you can use:
            LogisticRegression.fit(X,Y)
            # this method will train the model and store the final weight
            # will automaticly store X and Y and use the scale method to scale X because we using Gradient Desecent model
            # also it will add a column of ones
            # then will calculate the final w that will be used for predicting using both Gradient Desecent model and sigmoid function
            # Note in the fit you can also pass how many times you want gradient descent to run and the learning rate
            LogisticRegression.scale(X):
            # this method will scale any passed matrix and add a column of ones as the first column
            # then it will return the new modified matrix with the shape of (N X (D+1))

            LogisticRegression.cross_entropy(T,Yn)
            # this method accepet to argument the first is the acutal value and the second is the predicted value
            # will calculate and return the entropy cost using the following equation:
                    J = -[T log(Yn)  + (1-T)log(1-Yn)]

            LogisticRegression.sigmoid(z)
            # this method calculate the sigmoid

            LogisticRegression.predict(X)
            # this method use the (w) that optained from .fit
            # .fit necessary to use predict
            # and will predict the value of X using both weight (w) and the sigmoid function and the passed (X)

            LogisticRegression.score(X,Y)
            # pass the X you want to predict or test and the actual value Y  ,after training the model using .fit
            # it will predict the value of X and then test if the predicted value == the acutal one and sum all the True and divid on the total number
            # at the end it will return the classification rate
    """

    def scale(self, X):
        N, D = X.shape
        for j in range(D):
            std = X[:, j].std()
            m = X[:, j].mean()
            if std:
                for i in range(N):
                    X[i, j] = (X[i, j] - m) / std
        X = np.append(arr=np.ones((N, 1)).astype(int), values=X, axis=1)

        return(X)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cross_entropy(self, T, Yn):
        E = 0
        N = T.shape[0]
        for i in range(N):
            if T[i] == 1:
                E -= np.log(Yn[i])
            else:
                E -= np.log(1 - Yn[i])
        return E

    def fit(self, X, Y, times=500, lr=0.01):
        self.X = self.scale(X)
        self.Y = Y
        N, D = self.X.shape
        self.w = np.random.randn(D) / np.sqrt(D)
        self.costs = []
        z = self.X.dot(self.w)
        Yn = self.sigmoid(z)
        for t in range(times):
            z = self.X.dot(self.w)
            self.w = self.w - lr * self.X.T.dot((Yn - self.Y))
            Yn = self.sigmoid(z)
            if t % 10 == 0:
                self.costs.append(self.cross_entropy(self.Y, Yn))

    def predict(self, X):
        X = self.scale(X)
        return np.round(self.sigmoid(X.dot(self.w)))

    def score(self, X, Y):
        Yhat = self.predict(X)
        return np.mean(Y == Yhat)
