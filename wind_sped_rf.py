import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import time
import pandas as pd
df=pd.read_csv("data/regression/dataset_wind/wind_dataset.csv")
df = df.fillna(df.mean())  # Fill with mean values

df.head()
df['DATE']=pd.to_datetime(df['DATE'])
df['date_year']=df['DATE'].dt.year  #year
df['date_month_no']=df['DATE'].dt.month  #month
df['date_day']=df['DATE'].dt.day  # day
df=df.drop(columns=['DATE'])
X = df.drop(columns=['WIND'], axis=1)   ## Features
y = df['WIND']   ## target



# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k_list=[5,6,7,8,9,10]
for k_value in k_list:
    
    # Initialize K-Fold Cross-Validation
    kf = KFold(n_splits=k_value, shuffle=True, random_state=42)
    
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)# You can experiment with different kernels
    
    mse_scores = []  # To store the mean squared error scores for each fold
    training_times = []  # To store training times for each fold
    testing_times = []  # To store testing times for each fold
    
    for train_index, test_index in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Record the start time for training
        start_train = time.time()
        
      
        model.fit(X_train, y_train)
        
        # Record the end time for training
        end_train = time.time()
        
        # Record the start time for testing
        start_test = time.time()
    
        # Predict on the test data
        test_predictions = model.predict(X_test)
        
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
