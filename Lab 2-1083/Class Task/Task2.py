import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = None

search_dirs = [BASE, os.path.dirname(BASE), os.getcwd()]

def find_csv():
    for d in search_dirs:
        if os.path.isdir(d):
            for name in sorted(os.listdir(d)):
                if name.lower().endswith('.csv') and 'admission' in name.lower():
                    return os.path.join(d, name)
    return None

csv_path = CSV_PATH or find_csv()

if csv_path is None or not os.path.exists(csv_path):
    print("Could not find the admissions CSV. Folders searched:")
    for d in search_dirs:
        found = [n for n in os.listdir(d) if n.lower().endswith('.csv')] if os.path.isdir(d) else []
        print(f"  {d}  ->  CSV files: {found if found else 'none'}")
    raise SystemExit("Copy the CSV into one of these folders, or set CSV_PATH above.")

print("Using dataset:", csv_path)
data = pd.read_csv(csv_path)
data.columns = data.columns.str.strip()   # real files often have trailing spaces ('LOR ', 'Chance of Admit ')
print("Columns:", data.columns.tolist())

y = data['Chance of Admit']
df = data.drop(columns=['Serial No.', 'Chance of Admit'], errors='ignore')   # features only

simple_lr = LinearRegression()
simple_lr.fit(df[['GRE Score']], y)

plt.scatter(df['GRE Score'], y, color='green', label='Actual Data', alpha=0.5)
plt.plot(df['GRE Score'], simple_lr.predict(df[['GRE Score']]), color='red', linewidth=3, label='Regression Line')
plt.xlabel('GRE Score')
plt.ylabel('Admission chance')
plt.title('GRE Score vs Admission chance')
plt.legend()
plt.show()

x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
lr = LinearRegression()
lr.fit(x_train, y_train)

coefficients = lr.coef_
features = df.columns

plt.figure(figsize=(8, 5))
plt.barh(features, coefficients, color='teal')
plt.xlabel('Coefficient Value (Weight)')
plt.title('Feature Importance in Multivariate Linear Regression')
plt.axvline(x=0, color='black', linewidth=0.8)
plt.show()
pred = lr.predict(x_test)

rmse = np.sqrt(metrics.mean_squared_error(y_test, pred))
print("RMSE:", rmse)
print("R2 Score:", metrics.r2_score(y_test, pred))