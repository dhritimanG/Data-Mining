import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import  TfidfVectorizer
from __future__ import division

class Dbscan:
    def __init__(self, epsilon, min_pts):
        self.epsilon = epsilon
        self.min_pts = min_pts
        self.dataset = None
        self.visited = set()
        self.clusters = dict()
        self.noise = set()
        self.assigned = set()
        self.cluster_datasets = None
        self.n = None
        
    def fit(self, dataset):
        self.dataset = dataset
        self.n = dataset.shape[0]
        c = 0
        
        for p in range(self.n):
            print p
            if p in self.visited:
                continue
            else:
                self.visited.add(p)
                neighbors = self.range_query(dataset, p, self.epsilon)
                if len(neighbors) < self.min_pts:
                    self.noise.add(p)
                else:
                    c += 1
                    self.clusters[c] = set()
                    self.expand_cluster(p, neighbors, c, self.epsilon, self.min_pts)
            
    def expand_cluster(self, p, neighbors, c, epsilon, min_points):
        self.clusters[c].add(p)
        self.assigned.add(p)
        for q in neighbors:
            if q in self.noise:
                self.noise.remove(q)
                
            if q not in self.visited:
                self.visited.add(q)
                new_neighbors = self.range_query(self.dataset, q, self.epsilon)
                
                if len(new_neighbors) >= self.min_pts:
                    neighbors.extend(new_neighbors)
                
                if q not in self.assigned:
                    self.clusters[c].add(q)
                    
        self.build_clusters(self.dataset, self.clusters, self.noise)
        
    def range_query(self, dataset, p, epsilon):
        dist = euclidean_distances(dataset, np.array(np.matrix(dataset[p])))
        neighbors = list(np.where(dist <= epsilon)[0])
        
        return neighbors
        
    def build_clusters(self, dataset, clusters, noise):
        cluster_datasets = []
        
        for cluster in clusters.keys():
            indexes = list(clusters[cluster])
            tmp_dataset = dataset[indexes[0]]
            for i in range(1,len(indexes)):
                tmp_dataset = np.vstack((tmp_dataset, dataset[indexes[i]]))
                
            cluster_datasets.append(tmp_dataset)
            
        if len(noise) > 0:
            indexes = list(noise)
            tmp_dataset = dataset[indexes[0]]
            for i in range(1,len(indexes)):
                tmp_dataset = np.vstack((tmp_dataset, dataset[indexes[i]]))
                
            cluster_datasets.append(tmp_dataset)
        
        self.cluster_datasets = cluster_datasets
        
    def plotGraphs(self, cluster_datasets):
        color = 10*['r', 'b', 'g', 'y', 'c', 'k', 'o']
        
        for i in range(len(cluster_datasets)):
            plt.scatter(np.array(self.cluster_datasets[i][:,0]), np.array(self.cluster_datasets[i][:,1]), color = color[i])
            
    def calc_purity_gini(self, centroids_dict, labels, n, n_clusters):
        purity_cnt = 0
        gini = 0
        for i in centroids_dict:
            k = list(centroids_dict[i])
            y = []
            for i in range(len(k)):
                y.append(labels[k[i]])
            counts_dict = dict(Counter(y))
            tmp = max(counts_dict, key = counts_dict.get)
            purity_cnt += counts_dict[tmp]
        
            values = np.array(list(counts_dict.values()))
            values = np.square(values/np.sum(values))
            gini += 1 - np.sum(values)
            
        purity = purity_cnt/n
        gini_coef = gini/n_clusters
        print("Purity:", purity, "Gini:", gini_coef)


def load_fashion(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

    
train_X, train_y = load_fashion('/Users/dhritiman/Documents/GitHub/fashion-mnist/data/fashion', kind='train')
test_X, test_y = load_fashion('/Users/dhritiman/Documents/GitHub/fashion-mnist/data/fashion', kind='t10k')
train_X = train_X/255
test_X = test_X/255

model = Dbscan(epsilon = 5, min_pts = 3)
model.fit(test_X)

model.calc_purity_gini(model.clusters, test_y, model.n, len(model.clusters))
    
############################################################