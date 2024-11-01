import numpy as np
from scipy.spatial.distance import minkowski, cosine

class KNearestNeighbors:
    def __init__(self, k=5, distance_metric='minkowski', kernel='uniform', p=2):
        self.k = k
        self.distance_metric = distance_metric
        self.kernel = kernel
        self.p = p

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        k = 0
        predictions = []
        for x in X_test:
            k = k + 1
            #счётчик чтоб было не скучно ждать выполнения
            print(k)
            distances = self._compute_distances(x)
            nearest_indices = distances.argsort()[:self.k]
            nearest_distances = distances[nearest_indices]
            kernel_weights = self._apply_kernel(nearest_distances)
            nearest_targets = self.y_train[nearest_indices]
            weighted_sum = np.sum(kernel_weights * nearest_targets)
            total_weight = np.sum(kernel_weights)
            prediction = np.round(weighted_sum / total_weight).astype(int)
            predictions.append(prediction)
        return predictions

    def _compute_distances(self, x):
        x = x.flatten()
        if self.distance_metric == 'minkowski':
            return np.array([minkowski(x, x_train.flatten(), self.p) for x_train in self.X_train])
        elif self.distance_metric == 'cosine':
            return np.array([cosine(x, x_train) for x_train in self.X_train])

    def _apply_kernel(self, distances):
        if self.kernel == 'uniform':
            return np.ones(len(distances))
        elif self.kernel == 'gaussian':
            return np.exp(-0.5 * (distances ** 2))
        elif self.kernel == 'custom':
            a, b = 1, 2
            return (1 - np.abs(distances) ** a) ** b