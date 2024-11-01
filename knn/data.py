import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, r2_score
import algoritm


data = pd.read_csv('processed_manga_data.csv', header=0)
limited_data = data.head(5000)

target_column = 'status'
X = limited_data.drop(columns=[target_column])
y = limited_data[target_column]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


knn = algoritm.KNearestNeighbors(k=5, distance_metric='minkowski', kernel='gaussian', p=2)
knn.fit(X_train.values, y_train.values)  # Преобразуем в NumPy массивы


y_pred = knn.predict(X_test.values)

accuracy = accuracy_score(y_test, y_pred)

print(f'accuracy: {accuracy:.2f}')