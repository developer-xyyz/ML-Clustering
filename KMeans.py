#Ahnaf Ahmad
#1001835014

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def euclidean_distance(x1, x2):
    squared_distance = np.sum((x1 - x2) ** 2)
    distance = np.sqrt(squared_distance)
    return distance

class KMeans_Model:

    def __init__(self, K=2, max_iters=100):
        self.K = K
        self.max_iters = max_iters
       
        self.clusters = []
        for _ in range(self.K):
            self.clusters.append([])

        self.centroids = []


    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        random_samples= np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = []
        for idx in random_samples:
            self.centroids.append(self.X[idx])

        for _ in range(self.max_iters):
            self.clusters = self.create_clusters(self.centroids)

            centroids_old = self.centroids
            self.centroids = self.get_centroids(self.clusters)

            if self.converged(centroids_old, self.centroids):
                break

        return self.get_cluster_labels(self.clusters)


    def get_cluster_labels(self, clusters):
        labels = np.zeros(self.n_samples)
        for i in range(self.K):
            labels[clusters[i]] = i
        return labels

    def create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        labels = []
        for sample in self.X:
            closest_centroid = self.closest_centroid(sample, centroids)
            labels.append(closest_centroid)
        for idx in range(len(labels)):
            label = labels[idx]
            clusters[label].append(idx)
        return clusters

    def closest_centroid(self, sample, centroids):
        distances = []
        for point in centroids:
            distances.append(euclidean_distance(sample, point))
        closest_idx = np.argmin(distances)
        return closest_idx


    def get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx in range(self.K):
            cluster = clusters[cluster_idx]
            cluster_sum = np.zeros(self.n_features)
            for sample_idx in cluster:
                cluster_sum += self.X[sample_idx]
            cluster_mean = cluster_sum / len(cluster)
            centroids[cluster_idx] = cluster_mean
        return centroids


    def converged(self, centroids_old, centroids):
        distances = []
        for i in range(self.K):
            distances.append(euclidean_distance(centroids_old[i], centroids[i]))
        return sum(distances) == 0
    
    def plot(self):
        fig, ax = plt.subplots(figsize=(6, 4))

        for i in range(len(self.clusters)):
            index = self.clusters[i]
            point = self.X[index].T
            ax.scatter(*point)

        for i in range(len(self.centroids)):
            point = self.centroids[i]
            ax.scatter(*point, marker="x", color="black", linewidth=5)

        plt.show()

class KMeans:
    def __init__(self, datafile,k=2, ):
        self.datafile = datafile
        self.k = k

    def run(self):
        data = pd.read_csv(self.datafile, header=None, sep=',')
        X = data.values

        k = KMeans_Model(self.k,max_iters=1)
        labels = k.predict(X)
        k.plot()
        plt.show()

        k = KMeans_Model(self.k,max_iters=75)
        labels = k.predict(X)
        k.plot()
        plt.show()

        k = KMeans_Model(self.k,max_iters=150)
        labels = k.predict(X)
        k.plot()
        plt.show()

k = KMeans('ClusteringData.txt',2)
k.run()

k2 = KMeans('ClusteringData.txt',5)
k2.run()

k6 = KMeans('ClusteringData.txt',8)
k6.run()