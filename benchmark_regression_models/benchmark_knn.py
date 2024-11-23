# KNN
# 7 features x 3 waves x 14 channels + 7 features from alpha waves x 7 asymmetry channels = 343 features

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import psutil
import os
import sys

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # in MB

def get_size(obj, seen=None):
    """Recursively calculate size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

# Generate synthetic training data
np.random.seed(42)
X_train = np.random.rand(10000, 343)  # 10000 samples, 343 features
y_train = np.random.randint(0, 10, 10000)  # Labels between 0 and 9

# Generate synthetic test data
X_test = np.random.rand(1000, 343)  # 1000 test samples
y_test = np.random.randint(0, 10, 1000)  # True labels for test data

# Benchmark model creation and training
start_time = time.time()
start_memory = get_memory_usage()

knn_model = KNeighborsClassifier(n_neighbors=1, metric='manhattan')
knn_model.fit(X_train, y_train)

end_time = time.time()
end_memory = get_memory_usage()

training_time = end_time - start_time
memory_usage = end_memory - start_memory

print("----------------Start----------------")
print(f"Model creation and training time: {training_time:.4f} seconds")
print(f"Memory usage for training: {memory_usage:.2f} MB")

# Benchmark prediction time
start_time = time.time()
y_pred = knn_model.predict(X_test)
end_time = time.time()

prediction_time = end_time - start_time
prediction_time_per_sample = prediction_time / len(X_test)

print("------------------------------------")
print(f"Prediction time for {len(X_test)} samples: {prediction_time:.4f} seconds")
print(f"Average prediction time per sample: {prediction_time_per_sample:.6f} seconds")

# Benchmark prediction time for a single sample
single_sample = np.random.rand(1, 343)
start_time = time.time()
prediction = knn_model.predict(single_sample)
end_time = time.time()

single_prediction_time = end_time - start_time
print("------------------------------------")
print(f"Prediction time for a single sample: {single_prediction_time:.6f} seconds")

# Additional information
print("------------------------------------")
print(f"Number of features: {X_train.shape[1]}")
print(f"Number of training samples: {X_train.shape[0]}")
print(f"Number of test samples: {X_test.shape[0]}")
print(f"Model size in memory: {get_size(knn_model) / 1024 / 1024:.2f} MB")