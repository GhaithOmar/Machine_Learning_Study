import numpy as np
import matplotlib.pyplot as plt

from util import getData
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


class ANN(object):

    def __init__(self, M):
        self.M = M

    def fit(
            self,
            X,
            Y,
            X_test=None,
            Y_test=None,
            learning_rate=1e-6,
            epoch=10000,
            reg=0,
            show_fig=False):

        N, D = X.shape
        K = len(set(Y))
        T = self.y2indicator(Y)
        self.W1 = np.random.randn(D, self.M) / np.sqrt(D)
        self.b1 = np.zeros(self.M)
        self.W2 = np.random.randn(self.M, K) / np.sqrt(self.M)
        self.b2 = np.zeros(K)
        test = 0
        costs = []
        best_validation_error = 1

        for i in range(epoch):
            pY, Z = self.forward(X)

            # gradinet ascent
            pY_T = pY - T

            self.W2 -= learning_rate * (Z.T.dot(pY_T) + reg * self.W2)
            self.b2 -= learning_rate * (pY_T.sum() + reg * self.b2)

            # relu
            # dZ = pY_Y.dot(self.W2.T) * (Z>0)

            # tanh
            dZ = pY_T.dot(self.W2.T) * (1 - Z * Z)
            self.W1 -= learning_rate * (X.T.dot(dZ) + reg * self.W1)
            self.b1 -= learning_rate * (dZ.sum(axis=0) + reg * self.b1)

            if i % 10 == 0:
                pY_valid, _ = self.forward(Xvalid)
                c = self.cost2(Yvalid, pY_valid)
                e = self.error_rate(Yvalid, np.argmax(pY_valid, axis=1))
                costs.append(c)
                print(f'i: {i}', f'Cost = {c}', f'Error rate = {e}')

                if best_validation_error > e:
                    best_validation_error = e
        print(f'best test error rate = {e}')

        if show_fig:
            plt.plot(costs)
            plt.title('Cost plot')
            plt.show()

    def forward(self, X):
        # relu
        # Z = Z * (Z>0)
        # tanh
        Z = np.tanh(X.dot(self.W1) + self.b1)
        return self.softmax(Z.dot(self.W2) + self.b2), Z

    def predict(self, X):
        pY, _ = self.forward(X)
        return np.argmax(pY, axis=1)

    def score(self, X, Y):
        pY = self.predict(X)
        return 1 - error_rate(Y, pY)

    def cost2(self, T, Y):
        N = len(T)
        return -np.log(Y[np.arange(N), T]).mean()

    def error_rate(self, targets, predictions):
        return np.mean(targets != predictions)

    def y2indicator(self, y):
        N = len(y)
        K = len(set(y))
        ind = np.zeros((N, K))
        for i in range(N):
            ind[i, y[i]] = 1
        return ind

    def relu(self, x):
        return x * (x > 0)

    def sigmoid(self, A):
        return 1 / (1 + np.exp(-A))

    def softmax(self, A):
        expA = np.exp(A)
        return expA / expA.sum(axis=1, keepdims=True)
