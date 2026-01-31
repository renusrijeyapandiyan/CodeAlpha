# ==========================================
# Sales Prediction using Python
# File: sales_prediction.py
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
file_path = r"C:\Users\renus\OneDrive\Documents\myfolder\codeAlpha\Sales Prediction using Python\Advertising.csv"
df = pd.read_csv(file_path)

print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nDataset Description:")
print(df.describe())

# -----------------------------
# 2. Data Cleaning
# -----------------------------
# Drop duplicates
df.drop_duplicates(inplace=True)

# Fill missing numeric values with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# -----------------------------
# 3. Data Transformation
# -----------------------------
# Scale numeric features (TV, Radio, Newspaper)
numeric_cols = ['TV', 'Radio', 'Newspaper']
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# If categorical columns exist, apply one-hot encoding
# df = pd.get_dummies(df, drop_first=True)  # Uncomment if you have categorical columns

# -----------------------------
# 4. Feature Selection
# -----------------------------
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# 5. Regression Model
# -----------------------------
X = df.drop('Sales', axis=1)
y = df['Sales']

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict sales
y_pred = model.predict(X_test)

# -----------------------------
# 6. Model Evaluation
# -----------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error (MSE): {mse:.3f}")
print(f"R2 Score: {r2:.3f}")

# -----------------------------
# 7. Analyze Advertising Impact
# -----------------------------
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Impact'])
print("\nAdvertising Impact on Sales:")
print(coeff_df.sort_values(by='Impact', ascending=False))

# -----------------------------
# 8. Visualize Predictions
# -----------------------------
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

# -----------------------------
# 9. Save Predictions (Optional)
# -----------------------------
pred_df = X_test.copy()
pred_df['Actual_Sales'] = y_test
pred_df['Predicted_Sales'] = y_pred
pred_df.to_csv("Sales_Predictions.csv", index=False)
print("\nPredictions saved as 'Sales_Predictions.csv'")
