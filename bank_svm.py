
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

df = pd.read_csv('data/classification/dataset_5_Bank/bank/bank-full.csv',sep=';')


df['y']=df['y'].map({'yes':1,'no':0})
df['default']=df['default'].map({'yes':1,'no':0})
df['housing']=df['housing'].map({'yes':1,'no':0})
df['loan']=df['loan'].map({'yes':1,'no':0})

cat=df.select_dtypes(include=object)
cal_columns=cat.columns
numeric=df.select_dtypes(include=np.number)
numeric_colums=numeric.columns
numeric_colums
numeric_colums
scaler=MinMaxScaler()
scaler.fit(df[numeric_colums])
df[numeric_colums]=scaler.transform(df[numeric_colums])
enc=OneHotEncoder(sparse=False,handle_unknown='ignore')
enc.fit(df[cal_columns])
ohe_columns=list(enc.get_feature_names_out(cal_columns))
df[ohe_columns]=enc.transform(df[cal_columns])
df.drop(columns=cal_columns,axis=1,inplace=True)
df.columns

X = df.drop(columns='y')
y = df['y']



# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(len(X_train))
# Create an SVM classifier

model = SVC()


# Define the number of folds for cross-validation
k = 10
# Perform k-fold cross-validation on the training data
kf = KFold(n_splits=k, shuffle=True, random_state=42)
start_train_time = time.time()
scores = cross_val_score(model, X_train, y_train, cv=kf)


# Print the accuracy scores for each fold
for fold, score in enumerate(scores):
    print(f"Fold {fold + 1}: Accuracy = {score}")

# Calculate the training time


# Fit the Random Forest model on the entire training data

model.fit(X_train, y_train)
end_train_time = time.time()
train_time = end_train_time - start_train_time

start_test_time = time.time()
# Predict the labels for the test data
y_pred = model.predict(X_test)
end_test_time = time.time()
# Calculate the testing time
test_time = end_test_time - start_test_time



# Print the classification report
print(classification_report(y_test, y_pred))
print(f"Training Time: {train_time:.4f} seconds")
print(f"Testing Time: {test_time:.4f} seconds")

f1 = f1_score(y_test, y_pred, average='weighted')

# Print the F1-score
print(f"F1-Score: {f1}")