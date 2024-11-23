# RF
# 2 features x 2 waves x 14 channels + 3 features from alpha waves x 7 asymmetry channels = 77 features


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # in MB

# Generate synthetic training data
np.random.seed(42)
X_train = np.random.rand(10000, 77)  # 10000 samples, 77 features
y_train = np.random.randint(0, 10, 10000)  # Labels between 0 and 9

# Generate synthetic test data
X_test = np.random.rand(1000, 77)  # 1000 test samples, 77 features
y_test = np.random.randint(0, 10, 1000)  # True labels for test data

# Benchmark model creation and training
start_time = time.time()
start_memory = get_memory_usage()

rf_model = RandomForestClassifier(n_estimators=500, random_state=42)
rf_model.fit(X_train, y_train)

end_time = time.time()
end_memory = get_memory_usage()

training_time = end_time - start_time
memory_usage = end_memory - start_memory

print("----------------Start----------------")
print(f"Model creation and training time: {training_time:.4f} seconds")
print(f"Memory usage for training: {memory_usage:.2f} MB")

# Benchmark prediction time
start_time = time.time()
y_pred = rf_model.predict(X_test)
end_time = time.time()

prediction_time = end_time - start_time
prediction_time_per_sample = prediction_time / len(X_test)

print("------------------------------------")
print(f"Prediction time for {len(X_test)} samples: {prediction_time:.4f} seconds")
print(f"Average prediction time per sample: {prediction_time_per_sample:.6f} seconds")

# Benchmark prediction time for a single sample
single_sample = np.random.rand(1, 77)
start_time = time.time()
prediction = rf_model.predict(single_sample)
end_time = time.time()

single_prediction_time = end_time - start_time
print("------------------------------------")
print(f"Prediction time for a single sample: {single_prediction_time:.6f} seconds")

# Additional information
print("------------------------------------")
print(f"Number of features: {X_train.shape[1]}")
print(f"Number of training samples: {X_train.shape[0]}")
print(f"Number of test samples: {X_test.shape[0]}")
print(f"Number of trees in the forest: {rf_model.n_estimators}")
print(f"Model size in memory: {rf_model.__sizeof__() / 1024 / 1024:.2f} MB")