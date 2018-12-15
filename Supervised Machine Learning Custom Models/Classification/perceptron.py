import numpy as np
import matplotlib.pyplot as plt
from util import get_data as mnist
from datetime import datetime


class Perceptron:
    """ Perceptron is linear binary classifier
            Target is {-1,+1} not {0,1}
            equation
            prediction = sign(w.T*X + b)
            if w.T*X + b ==0   this point on the hyperplane
            if w.T*X + b >0    predict +1
            if w.T*X + b <0    predict -1

            Methods:

            Perceptron.fit(X,Y,learning_rate=optinal, epochs=optinal)


            Perceptron.predict(X)


            Perceptron.score(X)
     """

    def fit(self, X, Y, learning_rate=1.0, epochs=1000):
        D = X.shape[1]
        self.w = np.random.randn(D)
        self.b = 0

        N = len(Y)
        costs = []

        for epoch in range(epochs):
            # first we should get a prediction so we know what misclassified
            Yhat = self.predict(X)
            incorrect = np.nonzero(Y != Yhat)[0]
            if len(incorrect) == 0:
                break

            i = np.random.choice(incorrect)
            # we chose one random incorrect sample and then use the update rule
            self.w += learning_rate * Y[i] * X[i]
            # biase term since X0=1 it's equlivent to this
            self.b += learning_rate * Y[i]

            c = len(incorrect) / float(N)  # incorrect rate

            costs.append(c)
        print(
            f'final w: {self.w}, final b: {self.b}, epochs: {epoch +1 } / {epochs}')
        plt.plot(costs)
        plt.show()

    def predict(self, X):
        return np.sign(X.dot(self.w) + self.b)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)


if __name__ == '__main__':
    X, y = mnist()
    idx = np.logical_or(y == 0, y == 1)
    X = X[idx]
    y = y[idx]
    y[y == 0] = -1
    # because perceptron take target -1,1 so we need to change all the 0 to -1

    Ntrain = len(y) // 2

    X_train, y_train = X[:Ntrain], y[:Ntrain]
    X_test, y_test = X[Ntrain:], y[Ntrain:]

    classifier = Perceptron()
    t0 = datetime.now()
    classifier.fit(X_train, y_train)
    print(f"Training time is : {datetime.now()-t0}")

    t0 = datetime.now()
    print(f"Train accuracy is : {classifier.score(X_train,y_train)}")
    print(f"Time to compute train accuracy is {datetime.now()-t0}")

    t0 = datetime.now()
    print(f"Test accuracy is : {classifier.score(X_test,y_test)}")
    print(f"Time to compute Test accuracy is {datetime.now()-t0}")
