import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import time
import warnings

# Suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)



from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create an MLP regressor
mlp_regressor = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

# Define a range of K values to test
k_values = [5, 6, 7, 8, 9, 10]

for num_folds in k_values:
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    ensemble_predictions = np.zeros_like(y_train)  # To store ensemble predictions

    train_times = []
    test_times = []

    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Train the MLP regressor on the training fold and measure training time
        start_time = time.time()
        mlp_regressor.fit(X_train_fold, y_train_fold)
        train_times.append(time.time() - start_time)

        # Predict on the validation fold and measure test time
        start_time = time.time()
        val_predictions = mlp_regressor.predict(X_val_fold)
        test_times.append(time.time() - start_time)

        # Accumulate predictions for the ensemble model
        ensemble_predictions[val_index] += val_predictions

    # Average the predictions for the ensemble model
    ensemble_predictions /= num_folds

    # Test the ensemble model on the separate test dataset and measure test time
    start_time = time.time()
    test_predictions = mlp_regressor.predict(X_test)
    test_time = time.time() - start_time

    # Calculate mean squared error on the test set for the ensemble model
    mse_ensemble_test = mean_squared_error(y_test, test_predictions)

    print(f"K={num_folds}:")
    print(f"  Mean Squared Error on Test Set: {mse_ensemble_test:.2f}")
    print(f"  Average Training Time: {np.mean(train_times):.4f} seconds")
    print(f"  Average Test Time: {np.mean(test_times):.4f} seconds")
    print(f"  Total Test Time: {test_time:.4f} seconds")
