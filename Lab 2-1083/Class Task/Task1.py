import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = None
search_dirs = [BASE, os.path.dirname(BASE), os.getcwd()]

def find_csv():
    for d in search_dirs:
        if os.path.isdir(d):
            for name in sorted(os.listdir(d)):
                low = name.lower()
                if low.endswith('.csv') and ('head' in low or 'brain' in low):
                    return os.path.join(d, name)
    return None

csv_path = CSV_PATH or find_csv()

if csv_path is None or not os.path.exists(csv_path):
    print("Could not find headbrain.csv. Folders searched:")
    for d in search_dirs:
        found = [n for n in os.listdir(d) if n.lower().endswith('.csv')] if os.path.isdir(d) else []
        print(f"  {d}  ->  CSV files: {found if found else 'none'}")
    raise SystemExit("Copy the CSV into one of these folders, or set CSV_PATH above.")

df = pd.read_csv(csv_path)
print(df.head())

# Computing X and Y
X = df['Head Size(cm^3)'].values
y = df['Brain Weight(grams)'].values

X_sum = np.sum(X)
X_squared_sum = np.sum(X * X)
n = len(X)
print("Sum of X:", X_sum)
print("Sum of X squared:", X_squared_sum)
print("Length of X:", n)

n = len(X)

# calculating cross-deviation and deviation about x
numer = n * np.sum(X * y) - np.sum(X) * np.sum(y)
denom = n * np.sum(X * X) - (np.sum(X)) ** 2
w1 = numer / denom

# calculating regression coefficients
w0 = (np.sum(y) - w1 * (np.sum(X))) / n
print("w1 =", w1)
print("w0 =", w0)
max_x = np.max(X)
min_x = np.min(X)

# Calculating line values x and y
x1 = np.linspace(min_x, max_x)
y1 = w0 + w1 * x1

# Ploting Line
plt.plot(x1, y1, color='red', label='Regression Line')

# Ploting Scatter Points
plt.scatter(X, y, c='green', label='Scatter Plot')

plt.xlabel('Head Size in cm3')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()        # this line was cut off in the screenshot; the plot needs it

rmse = 0
for i in range(n):
    y_pred = w0 + w1 * X[i]
    rmse += (y[i] - y_pred) ** 2
rmse = np.sqrt(rmse / n)
print("RMSE=", rmse)

ss_tot = 0
ss_res = 0
y_mean = np.mean(y)
for i in range(n):
    y_pred = w0 + w1 * X[i]
    ss_tot += (y[i] - y_mean) ** 2
    ss_res += (y[i] - y_pred) ** 2
r2 = 1 - (ss_res / ss_tot)
print("R2 Score=", r2)
slope, intercept = np.polyfit(X, y, 1)
print("\nCheck with np.polyfit -> slope:", slope, "| intercept:", intercept)