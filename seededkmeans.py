
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np

class SeededKmeans(object):
    """
    seeds: tuple of datapoints with labels
    """
    def __init__(self, seeds=([], []), n_clusters=10, max_iter=3000, tolerance = 0.00001):
        self.seeds = seeds
        self.K = n_clusters
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.centroids = None
                                     
    def initialize_centroids(self): # initialize centroids by label
        idx = 0
        self.seed_dict = {}
        temp_labels = np.array(self.labels_)
        for i in np.sort(np.unique(self.labels_)):
            self.seed_dict[idx] = self.examples_[np.where(self.labels_ == i)[0], :]
            temp_labels[np.where(self.labels_ == i)[0]] = idx
            idx += 1

        # initialize random centroids
        random_seeds = np.random.permutation(self.X.shape[0])[:self.K]
        if random_seeds.size >= self.K:
            # In the normal process
            self.centroids = self.X[random_seeds, :]
        else:
            # In order to run extreme experiments with all seeds and no input datapoints
            self.centroids = np.random.rand(self.K, self.examples_.shape[1])

        for i in range(idx):
            self.centroids[i, :] = np.mean(self.seed_dict[i], axis=0)
                                                                                                                             

    def run_normal_kmeans(self): # run like normal after initialization of seeds
        done = False
        iter = 0
        while not done:
            self.assign_clusters()
            initial_centroids = np.array(self.centroids)
            self.calc_centroids()
            iter += 1
        
            convergence = max(self.euclidean(initial_centroids, self.centroids))
            if iter >= self.max_iter or convergence <= self.tolerance:
                done = True
                self.assign_clusters()

    def assign_clusters(self): # assign closest centroid to each instance
        self.cluster_assignments, distances = pairwise_distances_argmin_min(self.X, self.centroids)

    def calc_centroids(self): # recalculate centroids
        for i in range(self.K):
            if i in self.cluster_assignments:
                self.centroids[i,:] = np.mean(self.X[np.where(self.cluster_assignments == i)[0], :], axis=0)
            

    def euclidean(self, v1, v2):
        dist = (v1 - v2)**2
        return np.sqrt(np.sum(dist, 1))

    def check_data(self, X):
        self.X = X
    
    def check_seeds(self, seeds):
        self.examples_ = seeds[0]  # examples (x)
        self.labels_ = seeds[1]


    def fit(self, X):
        print("1")
        self.check_data(X)
        print("2")
        self.check_seeds(self.seeds)
        print("3")
        self.initialize_centroids()
        print("4")
        # append seeds
        if self.X.size == 0:
            self.X = self.examples_
        else:
            self.X = np.vstack((self.X, self.examples_))
        # run kmeans like usual
        print("5")
        self.run_normal_kmeans()
        print("6")

    def predict(self, X):
        self.check_data(X)
        self.assign_clusters()
        return self.cluster_assignments

