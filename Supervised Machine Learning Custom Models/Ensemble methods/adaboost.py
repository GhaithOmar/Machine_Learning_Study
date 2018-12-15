import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


class AdaBoost:
    # This AdaBoost model is Binary classifier
    def __init__(self, M):
        self.M = M

    def fit(self, X, Y):
        self.models = []
        self.alphas = []

        N, _ = X.shape  # get N number of sample
        W = np.ones(N) / N  # we identify W to be uniform distribution 1 over N

        for m in range(self.M):
            tree = DecisionTreeClassifier(max_depth=1)  # create decision Stump
            tree.fit(X, Y, sample_weight=W)
            # we get the prediction because we want to calculate our weighted
            # error
            P = tree.predict(X)
            # W  is always a probability distribution because we normailize it
            # in the w update
            err = W.dot(P != Y)
            alpha = 0.5 * (np.log(1 - err) - np.log(err))

            W = W * np.exp(-alpha * Y * P)  # vectorized form
            W = W / W.sum()  # normalize so it sums to 1
            self.models.append(tree)
            self.alphas.append(alpha)

    def predict(self, X):
        # Not like SKlearn API
        # we want accuracy and exponential loss for plotting purposes
        N, _ = X.shape
        FX = np.zeros(N)
        for alpha, tree in zip(self.alphas, self.models):
            FX += alpha * tree.predict(X)
        return np.sign(FX), FX

    def score(self, X, Y):

        P, FX = self.predict(X)
        L = np.exp(-Y * FX).mean()  # loss normalize by the number of samples
        # we take the mean rather than the sum
        # this insures both the loss and accuracy  in the same scale somewhere
        # between 0,1
        return np.mean(P == Y), L


if __name__ == '__main__':
    from mushroom_data import get_data
    X, Y = get_data()
    Y[Y == 0] = -1  # make the targets -1,+1
    Ntrain = int(0.8 * len(X))
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    T = 200
    train_errors = np.empty(T)
    test_losses = np.empty(T)
    test_errors = np.empty(T)
    for num_tress in range(T):
        if num_tress == 0:
            train_errors[num_tress] = None
            test_errors[num_tress] = None
            test_losses[num_tress] = None
            continue
        if num_tress % 20 == 0:
            print(num_tress)

        model = AdaBoost(num_tress)

        model.fit(Xtrain, Ytrain)
        acc, loss = model.score(Xtest, Ytest)
        acc_train, _ = model.score(Xtrain, Ytrain)
        train_errors[num_tress] = 1 - acc_train
        test_errors[num_tress] = 1 - acc
        test_losses[num_tress] = loss
        if num_tress == T - 1:
            print(f'final accur{acc}')
            print(f'final train error: {1-acc_train}')
            print(f'final test error: {1-acc}')
    plt.plot(test_errors, label='test errors')
    plt.plot(test_losses, label='test_losses')
    plt.legend()
    plt.show()

    plt.plot(train_errors, label='train errors')
    plt.plot(test_errors, label='test errors')
    plt.legend()
    plt.show()
