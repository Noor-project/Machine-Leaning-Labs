import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# 1. LOAD DATASET AND CHOOSE FEATURES
# ============================================================
BASE = os.path.dirname(os.path.abspath(__file__))

# Set CSV_PATH ONLY if the automatic search below fails. Use a raw string, e.g.
# CSV_PATH = r"C:\Users\btwit\Documents\Semester 7\Machine Learning\Lab\Machine-Leaning-Labs\Lab 3-1083\1000_Companies.csv"
CSV_PATH = None

# Otherwise look for a CSV with "compan" in its name in: the script's folder,
# the folder above it, and the folder VS Code is running from.
search_dirs = [BASE, os.path.dirname(BASE), os.getcwd()]

def find_csv():
    for d in search_dirs:
        if os.path.isdir(d):
            for name in sorted(os.listdir(d)):
                if name.lower().endswith('.csv') and 'compan' in name.lower():
                    return os.path.join(d, name)
    return None

csv_path = CSV_PATH or find_csv()

if csv_path is None or not os.path.exists(csv_path):
    print("Could not find 1000_Companies.csv. Folders searched:")
    for d in search_dirs:
        found = [n for n in os.listdir(d) if n.lower().endswith('.csv')] if os.path.isdir(d) else []
        print(f"  {d}  ->  CSV files: {found if found else 'none'}")
    raise SystemExit("Copy the CSV into one of these folders, or set CSV_PATH above.")

data = pd.read_csv(csv_path)
data.columns = data.columns.str.strip()
print("Shape:", data.shape)
print("Columns:", data.columns.tolist())
print(data.head())
print("\nMissing values:", data.isnull().sum().sum())

numeric_features = ['R&D Spend', 'Administration', 'Marketing Spend']
categorical_features = ['State']
target = 'Profit'

print("\nCorrelation of each numeric feature with Profit:")
print(data[numeric_features + [target]].corr()[target].drop(target).round(3))

# One-hot encode State (three cities with no order). drop_first=True removes one redundant column.
X = pd.get_dummies(data[numeric_features + categorical_features],
                   columns=categorical_features, drop_first=True, dtype=int)
y = data[target]
state_cols = [c for c in X.columns if c not in numeric_features]
print("\nFeatures after encoding:", X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================================
# 2 and 3. LINEAR REGRESSION AND POLYNOMIAL (DEGREE 2, 3, 4)
# ============================================================
# Numeric features are scaled and (for degree > 1) expanded to polynomial terms.
# The 0/1 State columns pass through unchanged, since squaring a 0/1 column adds nothing.
# The scaler is fit on the TRAIN data only (it sits inside the pipeline).
def build_model(degree):
    if degree == 1:
        numeric_steps = StandardScaler()
    else:
        numeric_steps = make_pipeline(StandardScaler(),
                                      PolynomialFeatures(degree=degree, include_bias=False))
    prep = ColumnTransformer([('num', numeric_steps, numeric_features)],
                             remainder='passthrough')
    return Pipeline([('prep', prep), ('reg', LinearRegression())])

rows = []
for degree in [1, 2, 3, 4]:
    m = build_model(degree)
    m.fit(X_train, y_train)
    test_pred = m.predict(X_test)
    n_features = m.named_steps['reg'].coef_.shape[0]
    rows.append({
        'Model': 'Linear' if degree == 1 else f'Poly deg {degree}',
        'Features': n_features,
        'Train R2': r2_score(y_train, m.predict(X_train)),
        'Test R2': r2_score(y_test, test_pred),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, test_pred)),
    })

results = pd.DataFrame(rows).set_index('Model')
print("\nComparison (all features):")
print(results.round(4))
print("\nBest test R2:", results['Test R2'].idxmax())

# ============================================================
# 4. PLOT FITTED CURVES FOR DIFFERENT DEGREES
# ============================================================
# A curve can only be drawn against ONE input, so the left plot uses the numeric feature
# most correlated with Profit. Those curves come from single-feature models, so their
# scores differ from the table above. The right plot shows the table's train vs test R2.
plot_feature = data[numeric_features + [target]].corr()[target].drop(target).abs().idxmax()
print("\nCurves plotted against:", plot_feature)

Xp_train = X_train[[plot_feature]]
grid = pd.DataFrame({plot_feature: np.linspace(X[plot_feature].min(), X[plot_feature].max(), 300)})

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

axes[0].scatter(X[plot_feature], y, color='gray', alpha=0.3, s=12, label='Actual data')
colors = {1: 'green', 2: 'orange', 3: 'red', 4: 'purple'}
for d in [1, 2, 3, 4]:
    steps = [StandardScaler()]
    if d > 1:
        steps.append(PolynomialFeatures(degree=d, include_bias=False))
    steps.append(LinearRegression())
    m = make_pipeline(*steps)
    m.fit(Xp_train, y_train)
    label = 'Linear (deg 1)' if d == 1 else f'Polynomial (deg {d})'
    axes[0].plot(grid[plot_feature], m.predict(grid), color=colors[d], linewidth=2, label=label)
axes[0].set_xlabel(plot_feature)
axes[0].set_ylabel('Profit')
axes[0].set_title(f'Fitted curves ({plot_feature} vs Profit)')
axes[0].legend()
axes[0].grid(True)

degrees = [1, 2, 3, 4]
axes[1].plot(degrees, results['Train R2'].values, marker='o', color='blue', label='Train R2')
axes[1].plot(degrees, results['Test R2'].values, marker='o', color='red', label='Test R2')
axes[1].set_xticks(degrees)
axes[1].set_xlabel('Polynomial degree (1 = linear)')
axes[1].set_ylabel('R2 score')
axes[1].set_title('Train vs Test R2 by degree (all features)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()