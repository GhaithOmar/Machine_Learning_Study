import numpy as np
from sortedcontainers import SortedList
# if 2 close points are the same distance away, one will be overwritten
from util import get_data, get_donut, get_xor
from datetime import datetime


class KNN(object):
    """ This is a K-Nearest Neighbor class, it makes a prediction using closest known data point from the training set.
            Method that avilable:
            when you iniate the class you should pass K
            K is the number of closest neighbors you want to use to predict

            KNN.fit(X,Y)
            this method only store X and Y as training data

            KNN.predict(X)
            this method that do all the work, it calculate the distance between the current point in the matrix provided and all the point in the training
            matrix that we stored using the fit method. Then compares the distance and take the lowest, the number of points taken depends on the passed
            K when the class initiated.
            Then calculate the number of votes for each class, the highest will be the predictied value for the current point

            KNN.score(X,Y)
            # pass the X you want to predict or test and the actual value Y  ,after training the model using .fit
            # it will predict the value of X and then test if the predicted value == the acutal one and sum all the True and divid on the total number
            # at the end it will return the classification rate
    """

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        # we create Y_hat as the same size of x since we need prediction for
        # every inputs
        y = np.zeros(len(X))
        for i, x in enumerate(
                X):  # (i) will be the index of the ROWs  on the test set and x will be the (i)th row
            sl = SortedList()
            for j, xt in enumerate(
                    self.X):  # (j) will be the index of the ROWs  on the training set and xt will be the (j)th row
                diff = x - xt				# we calculate the difference between each element in both train and test set
                d = diff.dot(diff)			# here we take the square distance
                if len(
                        sl) < self.k:		# if len sorted list is less than self.k we can add  the current point without checking any thing
                    # this mean the index J in the Y training set
                    sl.add((d, self.y[j]))
                else:
                    # if len sorted lis is > than self.k, we should check the value in the end at our sorted list with difference
                    # if the distance or the difference between x and xt is
                    # lower than the last value we should delete the last value
                    # in our sorted list
                    if d < sl[-1][0]:
                        del sl[-1]
                        # and here we add the new distance
                        sl.add((d, self.y[j]))
            votes = {}  # we create a dict to collect all the votes

            # we don't care in the vote case about the distance because now we
            # want to predict so we need the value of y (second element is the
            # class)
            for _, v in sl:
                # votes contain the class as the key and the count as value
                votes[v] = votes.get(v, 0) + 1
            max_votes = 0
            max_votes_class = -1
            for v, count in votes.items():  # we loop through the votes
                if count > max_votes:  # if this vote is grater than our max_votes we make this vote our max_votes and we store the count of that vote
                    max_votes = count
                    max_votes_class = v
                y[i] = max_votes_class  # we set yi to the corresponding class
        return y

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)


if __name__ == '__main__':
    X, y = get_data(2000)
    #X,y = get_xor()
    #X,y = get_donut()
    Ntrain = 1000
    X_train, y_train = X[:Ntrain], y[:Ntrain]
    X_test, y_test = X[Ntrain:], y[Ntrain:]
    for k in (1, 2, 3, 4, 5):
        knn = KNN(k)
        print(f"\nThis is for K = {k} \n\n")
        t0 = datetime.now()
        knn.fit(X_train, y_train)
        print(f'Training time: {datetime.now()-t0}')
        knn.predict(X_test)
        t0 = datetime.now()
        print(f"Train accuracy: {knn.score(X_train,y_train)}")
        print(
            f"Time to compile: {datetime.now()-t0}",
            f'Train size: {len(y_train)}')
        t0 = datetime.now()
        print(f"test accuracy: {knn.score(X_test,y_test)}")
        print(
            f"Time to compile: {datetime.now()-t0}",
            f'Test size: {len(y_train)}')
