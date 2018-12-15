import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from soft_k_means import plot_k_means
from datetime import datetime 

def get_data(limit=None):
	print('Reading in and transforming data....')
	df = pd.read_csv('..\\Large Files\\train.csv').values
	X = df[:, 1:]/255.0
	Y = df[:,0]

	if limit is not None:
		X,Y = X[:limit], Y[:limit]
	return X, Y 


def purity(Y,R):
	N,K = R.shape 
	p = 0

	for k in range(K):
		best_target = -1  # we don'tstrictly need to store this
		max_intersection = 0
		for j in range(K):
			# we need only the row that corresponding to this target label
			# which is j and thats the first index
			# the second index is which cluster k we currently looking at
			intersection = R[Y==j, k].sum()
			if intersection > max_intersection:
				max_intersection =intersection
				best_target = j
		p += max_intersection

	return p/N

def DBI(X, M, R):
	# lower is better 

	K, D = M.shape

	#get sigmas first

	sigma = np.zeros(k)
	for k in range(K):
		diffs = X - M[k] #should be NXD

		squared_distances = (diffs*diffs.sum(axis=1))

		weighted_squared_distances = R[:, k] * squared_distances
		sigma[k] = np.sqrt(weighted_squared_distances).mean()

	# calculate the Davies-Bouldin Index
	dbi = 0
	for k in range(K):
		max_ratio=0
		for j in range(K):
			if k != j:
				numerator = sigma[k] + sigma[j]
				denominator = np.linalg.norm(M[k] - M[j])
				ratio = numerator/denominator
				if ratio > max_ratio:
					max_ratio = ratio
		dbi += max_ratio
	return dbi/k 


def main():
	X,Y = get_data(1000)

	print("Number of data points:", len(Y))

	M, R = plot_k_means(X, len(set(Y)))

	print(f'Purity: {purity(Y,R)}')

	print(f'DBI: {DBI(X,M,R)}')


if __name__ == '__main__':
	main()
