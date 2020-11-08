import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from seededkmeans import SeededKmeans
class ConstrainedKmeans(SeededKmeans):
    """
    Constrained KMeans:
    Semi-supervised clustering algorithm described in the paper Sugato Basu, Arindam Banerjee, and R. Mooney. Semi-
    supervised clustering by seeding with constrains. In Proceedings of 19th International Conference on Machine Learning (ICML-2002)
    Parameters
    -----------------------
    n_clusters : number of clusters to look for
    max_iter: maximum number of iterations
    seed_datapoints = (Sx,Sy): tupple defining the labelled datapoints to build the semi-supervised clustering
        algorithm.
        Sx: NxM numpy matrix containing the M attributes of the N seed datapoints.
        Sy: N numpy array with labels of the N datapoints to train the semisupervised method.
    append_seeds: if True, the seeds will be added to the dataset in the fit method. If the seeds are already in the X
        input, this variable must be set to False.
    """

    def __init__(self, seeds=([], []), n_clusters=10, max_iter=100, tolerance=1e-5, verbose=False):
        super(ConstrainedKmeans, self).__init__(seeds, n_clusters=n_clusters, max_iter=max_iter,
                                                tolerance=tolerance)

    def _assign_clusters(self):
        # This method assigns the closest centroid to each instance
        self.cluster_assignments, distances = pairwise_distances_argmin_min(self.X, self.centroids)

        # Reassign the seed instances.
        self.cluster_assignments[self.seeds_indexes] = self.seeds_initial_assignment

    def _fit(self):
        # Starts fitting
        self.initialize_centroids()
        if self.X.size == 0:
            # Just for the extreme case with all seeds and no data.
            self.X = self.examples_
            self.seeds_initial_assignment = self.labels_
            self.seeds_indexes = range(self.X.shape[0])
        else:
            self.X = np.vstack((self.X, self.examples_))
            self.seeds_initial_assignment = self.labels_
            self.seeds_indexes = range(self.X.shape[0] - self.examples_.shape[0], self.X.shape[0])

        # run normal kmeans
        self.run_normal_kmeans()

    def predict(self, X):
        self.check_data(X)
        self.cluster_assignments, distances = pairwise_distances_argmin_min(self.X, self.centroids)

        return self.cluster_assignments