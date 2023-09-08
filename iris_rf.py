import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import time

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
model = RandomForestClassifier(n_estimators=100)

# Define the number of folds for cross-validation
k = 7

# Perform k-fold cross-validation on the training data
kf = KFold(n_splits=k, shuffle=True, random_state=42)
start_train_time = time.time()
scores = cross_val_score(model, X_train, y_train, cv=kf)
end_train_time = time.time()

# Print the accuracy scores for each fold
for fold, score in enumerate(scores):
    print(f"Fold {fold + 1}: Accuracy = {score}")

# Calculate the training time
train_time = end_train_time - start_train_time

# Fit the Random Forest model on the entire training data
start_test_time = time.time()
model.fit(X_train, y_train)
end_test_time = time.time()

# Predict the labels for the test data
y_pred = model.predict(X_test)

# Calculate the testing time
test_time = end_test_time - start_test_time

# Print the classification report
print(classification_report(y_test, y_pred, target_names=iris.target_names))
print(f"Training Time: {train_time:.4f} seconds")
print(f"Testing Time: {test_time:.4f} seconds")
