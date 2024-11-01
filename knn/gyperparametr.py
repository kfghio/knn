import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform
import algoritm
import random


def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def random_search_knn(X_train, y_train, X_val, y_val, param_distributions, n_iter=10):
    best_score = 0
    best_params = {}
    results = []

    for i in range(n_iter):
        k = random.choice(param_distributions['n_neighbors'])
        p = random.choice(param_distributions['p'])
        weights = random.choice(param_distributions['weights'])

        #Для выявления наилучших параметров запускаю
        #knn = algoritm.KNearestNeighbors(k=k, distance_metric='minkowski', kernel=weights, p=p)
        #Строю графики(на всякий)
        #Затем на основе лучшего результата фиксирую параметры
        #Строю окончательный график
        knn = algoritm.KNearestNeighbors(k=k, distance_metric='minkowski', kernel='uniform', p=1)
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_val)
        score = calculate_accuracy(y_val, y_pred)
        results.append((k, p, weights, score))

        if score > best_score:
            best_score = score
            best_params = {'n_neighbors': k, 'p': p, 'weights': weights}

    return best_params, best_score, results


data = pd.read_csv('processed_manga_data.csv', header=0)
limited_data = data.head(2000)

target_column = 'status'
X = limited_data.drop(columns=[target_column])
y = limited_data[target_column]


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


param_distributions = {
    'n_neighbors': list(range(1, 20)),
    'p': [1, 2],
    'weights': ['uniform', 'gaussian', 'custom']
}


best_params, best_score, search_results = random_search_knn(X_train.values, y_train.values, X_val.values, y_val.values, param_distributions, n_iter=10)
print("Лучшие параметры:", best_params)
print("Лучшая точность:", best_score)


neighbors = [result[0] for result in search_results]
accuracies = [result[3] for result in search_results]

plt.figure(figsize=(10, 6))
plt.scatter(neighbors, accuracies, c='blue', marker='o')
plt.xlabel("Количество соседей (k)")
plt.ylabel("Точность на валидационном множестве")
plt.title("Зависимость точности от числа соседей при случайном поиске гиперпараметров")
plt.show()
