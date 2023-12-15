import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import time
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
# Step 1: Load and preprocess the dataset
train_data = pd.read_csv("data/classification/dataset_3_Titanic/train.csv")
test_data = pd.read_csv("data/classification/dataset_3_Titanic/test.csv")

# Drop irrelevant columns (PassengerId, Name, Ticket, Cabin)
train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Convert categorical variables (Sex and Embarked) to numerical using LabelEncoder
label_encoder = LabelEncoder()
train_data['Sex'] = label_encoder.fit_transform(train_data['Sex'])
test_data['Sex'] = label_encoder.transform(test_data['Sex'])
train_data['Embarked'] = label_encoder.fit_transform(train_data['Embarked'].fillna('S'))
test_data['Embarked'] = label_encoder.transform(test_data['Embarked'].fillna('S'))

# Fill missing values in the Age column with the mean age
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)

# Fill missing values in the Fare column with the mean fare
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)

# Step 2: Split the dataset into training and testing sets
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the SVM model on the training set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVC(kernel='linear', C=1.0, random_state=42)


k = 5

# Perform k-fold cross-validation on the training data
kf = KFold(n_splits=k, shuffle=True, random_state=42)

scores = cross_val_score(model, X_train_scaled, y_train, cv=kf)


# Fit the SVM model on the entire training data

start_train_time = time.time()
model.fit(X_train_scaled, y_train)
end_train_time = time.time()

# Calculate the training time
train_time = end_train_time - start_train_time


start_test_time = time.time()
# Predict the labels for the test data
y_pred = model.predict(X_test_scaled)
end_test_time = time.time()
# Calculate the testing time
test_time = end_test_time - start_test_time

# Print the classification report
print(classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived']))
print(f"Training Time: {train_time:.4f} seconds")
print(f"Testing Time: {test_time:.4f} seconds")

# Convert predicted numeric labels back to original categorical labels
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_test_labels = label_encoder.inverse_transform(y_test)

# Calculate the F1-score
f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')
print(f"F1-Score: {f1}")