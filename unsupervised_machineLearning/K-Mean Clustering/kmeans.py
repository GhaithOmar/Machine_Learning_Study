import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

df = pd.read_csv('Mall_Customers.csv')
X = df.iloc[:,3:].values

# Using the elbow method to find the optimal number of cluster
from sklearn.cluster import KMeans
wcss = [] 

for i in range(1,11):
	kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
	kmeans.fit(X)
	# inertia is in clusters sum of squares
	wcss.append(kmeans.inertia_)	

plt.plot(range(1,11), wcss)
plt.title("The Elbow method")
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

plt.show()

# Applying K-means to the mall Dataset
kmeans = KMeans(n_clusters=5, init= 'k-means++', max_iter=300, n_init=10, random_state=0)
#fit_predict method is going to tell us the cluster in which each client belong

y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters

plt.scatter(X[:,0],X[:,1], s=100, c=y_kmeans)
plt.show()