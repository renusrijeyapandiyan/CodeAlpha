# Car Price Prediction with Machine Learning

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# 1. Load the dataset
# -------------------------------
data = pd.read_csv("car data.csv")

print("Dataset Loaded Successfully")
print(data.head())

# -------------------------------
# 2. Data Preprocessing
# -------------------------------

# Drop car name (not useful for prediction)
data.drop("Car_Name", axis=1, inplace=True)

# Convert categorical columns to numerical
data = pd.get_dummies(data, drop_first=True)

# -------------------------------
# 3. Split features and target
# -------------------------------
X = data.drop("Selling_Price", axis=1)
y = data["Selling_Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 4. Train Linear Regression Model
# -------------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)

lr_predictions = lr.predict(X_test)

print("\nLinear Regression Results:")
print("MAE:", mean_absolute_error(y_test, lr_predictions))
print("MSE:", mean_squared_error(y_test, lr_predictions))
print("R2 Score:", r2_score(y_test, lr_predictions))

# -------------------------------
# 5. Train Random Forest Regressor
# -------------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_predictions = rf.predict(X_test)

print("\nRandom Forest Results:")
print("MAE:", mean_absolute_error(y_test, rf_predictions))
print("MSE:", mean_squared_error(y_test, rf_predictions))
print("R2 Score:", r2_score(y_test, rf_predictions))

# -------------------------------
# 6. Visualization
# -------------------------------
plt.scatter(y_test, rf_predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Price (Random Forest)")
plt.show()
