
------------------------------------------------------------
Student Name -Konapala Tarakananda
Roll No. - 2511AI41
Subject - APR
Submitted to - Dr.chandranath adak
------------------------------------------------------------

Dataset: salaries_data.csv


Run this script directly:
    
------------------------------------------------------------

"""
# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load the dataset
# Make sure Salary_Data.csv is in the same folder as this script
data = pd.read_csv("Salary_Data.csv")

# Display first few rows
print("First 5 rows of the dataset:")
print(data.head())

# Step 3: Separate features (X) and target (y)
X = data[['YearsExperience']]  # independent variable
y = data['Salary']             # dependent variable

# Step 4: Split data into training and testing sets (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predict salaries for test data
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
print("\nModel Performance:")
print(f"Intercept (b0): {model.intercept_:.2f}")
print(f"Coefficient / Slope (b1): {model.coef_[0]:.2f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

# Step 8: Visualize Training Data with Regression Line
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.plot(X_train, model.predict(X_train), color='red', linewidth=2, label='Regression Line')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Simple Linear Regression (Training Set)")
plt.legend()
plt.show()

# Step 9: Visualize Test Data with Regression Line
plt.scatter(X_test, y_test, color='green', label='Test Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Regression Line')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Simple Linear Regression (Test Set)")
plt.legend()
plt.show()

