import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import algoritm


class LOWESSAnomalyDetector:
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def _compute_weights(self, x):
        """
        Compute weights for each training point based on the distance to the test point x.
        Using a Gaussian kernel for weighting.
        """
        distances = np.array([euclidean(x, x_train) for x_train in self.X_train])
        weights = np.exp(-0.5 * (distances / self.bandwidth) ** 2)
        return weights / np.sum(weights)

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            weights = self._compute_weights(x)
            weighted_avg = np.dot(weights, self.y_train)  # Weighted sum of target values
            predictions.append(weighted_avg)
        return np.array(predictions)

    def detect_anomalies(self, X_test, y_test, threshold=1.5):
        predictions = self.predict(X_test)
        residuals = np.abs(predictions - y_test)
        anomalies = residuals > threshold
        return anomalies, residuals


data = pd.read_csv('processed_manga_data.csv', header=0)
limited_data = data.head(2000)

target_column = 'status'
X = limited_data.drop(columns=[target_column])
y = limited_data[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = algoritm.KNearestNeighbors(k=5, distance_metric='minkowski', kernel='uniform', p=2)
knn.fit(X_train.values, y_train.values)

baseline_predictions = knn.predict(X_test.values)
baseline_mse = mean_squared_error(y_test.values, baseline_predictions)

detector = LOWESSAnomalyDetector(bandwidth=1.0)
detector.fit(X_train.values, y_train.values)

anomalies, residuals = detector.detect_anomalies(X_test.values, y_test.values, threshold=1.5)

predictions = detector.predict(X_test.values)
weighted_mse = mean_squared_error(y_test.values, predictions)

print(f"Baseline MSE (before weighting): {baseline_mse:.3f}")
print(f"Weighted MSE (after LOWESS weighting): {weighted_mse:.3f}")

print(f"Number of anomalies detected: {np.sum(anomalies)}")