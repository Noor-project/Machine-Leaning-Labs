import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
BASE = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = None

search_dirs = [BASE, os.path.dirname(BASE), os.getcwd()]

def find_csv():
    for d in search_dirs:
        if os.path.isdir(d):
            for name in sorted(os.listdir(d)):
                if name.lower().endswith('.csv') and 'position' in name.lower():
                    return os.path.join(d, name)
    return None

csv_path = CSV_PATH or find_csv()

if csv_path is None or not os.path.exists(csv_path):
    print("Could not find Position_Salaries.csv. Folders searched:")
    for d in search_dirs:
        found = [n for n in os.listdir(d) if n.lower().endswith('.csv')] if os.path.isdir(d) else []
        print(f"  {d}  ->  CSV files: {found if found else 'none'}")
    raise SystemExit("Copy the CSV into one of these folders, or set CSV_PATH above.")

# Load dataset
data = pd.read_csv(csv_path)

# Independent variable (Level) and dependent variable (Salary)
X = data[["Level"]].values
y = data["Salary"].values

poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

# Train polynomial regression model
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# Predictions
y_pred = poly_model.predict(X_poly)

# Plot results
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, y_pred, color="red", label="Polynomial Fit (deg=4)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.title("Polynomial Regression")
plt.legend()
plt.show()

# Find out its R2 score and RMSE
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print("R2 Score:", r2)
print("RMSE:", rmse)