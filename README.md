# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the placement dataset and preprocess it by handling missing values and encoding categorical data.

2.Split the dataset into training and testing sets.

3.Train a Logistic Regression model using the training data.

4.Predict student placement on test data and evaluate using accuracy and confusion matrix. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Mahith M
RegisterNumber: 212225220061 
*/
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Rachitha U
RegisterNumber:  212225220078
*/


# Logistic Regression – Student Placement Prediction

# Step 1: Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load the dataset
data = pd.read_csv("Placement_Data.csv")

# Display first few rows
print(data.head())

# Step 3: Data Preprocessing

# Drop serial number column (not useful for prediction)
data.drop('sl_no', axis=1, inplace=True)

# Fill missing salary values with 0
data['salary'].fillna(0, inplace=True)

# Convert categorical columns into numerical values
le = LabelEncoder()
categorical_columns = [
    'gender', 'ssc_b', 'hsc_b', 'hsc_s',
    'degree_t', 'workex', 'specialisation', 'status'
]

for col in categorical_columns:
    data[col] = le.fit_transform(data[col])

# Step 4: Split features and target
X = data.drop('status', axis=1)
y = data['status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 6: Prediction
y_pred = model.predict(X_test)

# Step 7: Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

## Output:
<img width="1135" height="532" alt="image" src="https://github.com/user-attachments/assets/35b9ce97-e45e-4b75-8583-3135a25ae9dd" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
