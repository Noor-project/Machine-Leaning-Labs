import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# 1. LOAD AND EXPLORE THE DATASET
# ============================================================
BASE = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = None
search_dirs = [BASE, os.path.dirname(BASE), os.getcwd()]

def find_csv():
    for d in search_dirs:
        if os.path.isdir(d):
            for name in sorted(os.listdir(d)):
                low = name.lower()
                if low.endswith('.csv') and ('insurance' in low or 'medical' in low):
                    return os.path.join(d, name)
    return None

csv_path = CSV_PATH or find_csv()

if csv_path is None or not os.path.exists(csv_path):
    print("Could not find the insurance CSV. Folders searched:")
    for d in search_dirs:
        found = [n for n in os.listdir(d) if n.lower().endswith('.csv')] if os.path.isdir(d) else []
        print(f"  {d}  ->  CSV files: {found if found else 'none'}")
    raise SystemExit("Copy the CSV into one of these folders, or set CSV_PATH above.")

data = pd.read_csv(csv_path)

print("Shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())
print("\nInfo:")
data.info()
print("\nSummary statistics:")
print(data.describe())
print("\nMissing values:")
print(data.isnull().sum())
print("\nsmoker counts:\n", data['smoker'].value_counts())
print("\nregion counts:\n", data['region'].value_counts())

# Features and target (features exactly as listed in the activity; 'sex' is not included)
features = ['age', 'bmi', 'children', 'smoker', 'region']
X = data[features]
y = data['charges']

# ============================================================
# 2. PREPROCESS
# ============================================================
# 2.1 Encode categorical features (region, smoker) with one-hot encoding.
# drop_first=True removes one redundant column per feature; for smoker this
# leaves a single 0/1 column (smoker_yes).
X = pd.get_dummies(X, columns=['smoker', 'region'], drop_first=True, dtype=int)
print("\nColumns after encoding:", X.columns.tolist())

# Split BEFORE scaling so the scaler learns only from the training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2.2 Scale numerical features (fit on train only, apply to test)
num_cols = ['age', 'bmi', 'children']
scaler = StandardScaler()
X_train = X_train.copy()
X_test = X_test.copy()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# ============================================================
# 3. TRAIN LINEAR REGRESSION
# ============================================================
model = LinearRegression()
model.fit(X_train, y_train)

print("\nCoefficients:")
for name, coef in zip(X_train.columns, model.coef_):
    print(f"  {name:20s} {coef:10.2f}")
print(f"  {'intercept':20s} {model.intercept_:10.2f}")

# ============================================================
# 4. EVALUATE (RMSE and R2 on the TEST set)
# ============================================================
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("\nRMSE:", rmse)
print("R2 Score:", r2)

# ============================================================
# 5. PLOT PREDICTED VS ACTUAL COSTS
# ============================================================
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue', label='Predictions')
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, color='red', linewidth=2, label='Perfect prediction')
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Predicted vs Actual Insurance Costs')
plt.legend()
plt.grid(True)
plt.show()