# this is the source code for credit scoring model 
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the Dataset
data = pd.read_csv('/content/credit_data.csv')  # Ensure the correct path if needed

# Step 2: Handle Missing Values
data['Income'].fillna(data['Income'].mean(), inplace=True)  # Fill missing income with the mean

# Step 3: Encode Categorical Variables
data = pd.get_dummies(data, columns=['Gender', 'Job_Type'], drop_first=True)

# Step 4: Split the Data into Features and Target
X = data.drop('Loan_Default', axis=1)  # Features
y = data['Loan_Default']  # Target variable

# Step 5: Split the Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train a Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Predict the Test Set Results
y_pred = model.predict(X_test)

# Step 8: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the evaluation results
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)

# Step 9: Make Loan Decisions Based on Predictions
print("\nLoan Decisions:")
for i in range(len(y_test)):
    probability = model.predict_proba(X_test)[i, 1]  # Probability of default
    decision = "Deny Loan" if probability >= 0.5 else "Grant Loan"  # Threshold for loan approval
    print(f"Individual {i+1}: Loan Default Probability = {probability:.2f}, Decision: {decision}")
