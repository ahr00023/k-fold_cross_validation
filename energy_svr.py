import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import time

# Load the dataset
data = pd.read_excel('https://github.com/JielingChen/building_energy_efficiency_prediction/raw/main/ENB2012_data.xlsx')
data.columns = ['relative_compactness', 'surface_area', 'wall_area', 'roof_area', 'overall_height', 'orientation', 
                'glazing_area', 'glazing_area_distribution', 'heating_load', 'cooling_load']

y = data['heating_load']
X = data.drop(['heating_load', 'cooling_load'], axis=1)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k_list=[5,6,7,8,9,10]
for k_value in k_list:
    
    # Initialize K-Fold Cross-Validation
    kf = KFold(n_splits=k_value, shuffle=True, random_state=42)
    
    # Create an SVR model
    svr = SVR(kernel='linear')  # You can experiment with different kernels
    
    mse_scores = []  # To store the mean squared error scores for each fold
    training_times = []  # To store training times for each fold
    testing_times = []  # To store testing times for each fold
    
    for train_index, test_index in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Record the start time for training
        start_train = time.time()
        
        # Fit the SVR model to the training data
        svr.fit(X_train, y_train)
        
        # Record the end time for training
        end_train = time.time()
        
        # Record the start time for testing
        start_test = time.time()
    
        # Predict on the test data
        test_predictions = svr.predict(X_test)
        
        # Record the end time for testing
        end_test = time.time()
    
        # Calculate the mean squared error for this fold
        mse_fold = mean_squared_error(y_test, test_predictions)
        mse_scores.append(mse_fold)
        
        # Calculate training time and testing time for this fold
        training_time = end_train - start_train
        testing_time = end_test - start_test
        training_times.append(training_time)
        testing_times.append(testing_time)
    
    # Calculate the mean of the MSE scores, training times, and testing times from all folds
    mean_mse = np.mean(mse_scores)
    mean_training_time = np.mean(training_times)
    mean_testing_time = np.mean(testing_times)
    
    print("K-value:", k_value)
    print("Mean MSE from K-Fold Cross Validation:", mean_mse)
    print("Mean Training Time:", mean_training_time, "seconds")
    print("Mean Testing Time:", mean_testing_time, "seconds")
