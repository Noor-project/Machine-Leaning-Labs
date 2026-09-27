import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# 1. LOAD DATASET AND CHOOSE FEATURES
# ============================================================
BASE = os.path.dirname(os.path.abspath(__file__))

# Set CSV_PATH ONLY if the automatic search below fails. Use a raw string, e.g.
# CSV_PATH = r"C:\Users\btwit\Documents\...\car_data.csv"
CSV_PATH = None

# If the automatic column matching below picks the wrong columns (or none), set these
# to your dataset's exact column names, e.g.
# TARGET_COLUMN = 'Selling_Price'
# FEATURE_COLUMNS = ['Kms_Driven', 'Year']
TARGET_COLUMN = None
FEATURE_COLUMNS = None

# Otherwise look for a CSV with "car" in its name in: the script's folder,
# the folder above it, and the folder VS Code is running from.
search_dirs = [BASE, os.path.dirname(BASE), os.getcwd()]

def find_csv():
    for d in search_dirs:
        if os.path.isdir(d):
            for name in sorted(os.listdir(d)):
                if name.lower().endswith('.csv') and 'car' in name.lower():
                    return os.path.join(d, name)
    return None

csv_path = CSV_PATH or find_csv()

if csv_path is None or not os.path.exists(csv_path):
    print("Could not find the car price CSV. Folders searched:")
    for d in search_dirs:
        found = [n for n in os.listdir(d) if n.lower().endswith('.csv')] if os.path.isdir(d) else []
        print(f"  {d}  ->  CSV files: {found if found else 'none'}")
    raise SystemExit("Copy the CSV into one of these folders, or set CSV_PATH above.")

raw = pd.read_csv(csv_path)
print("Dataset:", os.path.basename(csv_path), "| shape:", raw.shape)
print("Columns:", raw.columns.tolist())

# The slide names the features Kms_Driven, Year, Engine, Horsepower, but different
# versions of this dataset name (or lack) these columns. Match them case-insensitively.
cols_lower = {c.strip().lower(): c for c in raw.columns}

def pick(aliases):
    for a in aliases:
        if a in cols_lower:
            return cols_lower[a]
    return None

target = TARGET_COLUMN or pick(['price', 'selling_price', 'sellingprice', 'car_price', 'sale_price'])

if FEATURE_COLUMNS:
    feature_map = {c: c for c in FEATURE_COLUMNS}
else:
    aliases = {
        'Kms_Driven': ['kms_driven', 'km_driven', 'kilometers_driven', 'mileage', 'odometer'],
        'Year': ['year', 'model_year', 'manufacture_year'],
        'Engine': ['engine', 'engine_size', 'enginesize', 'engine_cc'],
        'Horsepower': ['horsepower', 'max_power', 'power', 'hp'],
    }
    feature_map = {}
    for nice_name, alias_list in aliases.items():
        found = pick(alias_list)
        if found:
            feature_map[nice_name] = found
        else:
            print(f"Note: no column found for '{nice_name}', skipping it.")

if target is None or not feature_map:
    raise SystemExit("Could not match the target/feature columns automatically.\n"
                     "Set TARGET_COLUMN and FEATURE_COLUMNS near the top using the column list printed above.")

def to_num(s):
    """Turn columns like '1248 CC' or '74 bhp' or '45,000' into numbers."""
    if pd.api.types.is_numeric_dtype(s):
        return s
    cleaned = s.astype(str).str.replace(',', '', regex=False).str.extract(r'(\d+\.?\d*)')[0]
    return pd.to_numeric(cleaned, errors='coerce')

data = pd.DataFrame({name: to_num(raw[col]) for name, col in feature_map.items()})
data['Price'] = to_num(raw[target])
before = len(data)
data = data.dropna()
print(f"\nTarget column : {target}")
print(f"Features used : {list(feature_map.values())}")
print(f"Rows kept     : {len(data)} of {before} (rows with missing values dropped)")

features = list(feature_map.keys())
print("\nCorrelation of each feature with Price:")
print(data.corr()['Price'].drop('Price').round(3))

X = data[features]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================================
# 2 and 3. LINEAR REGRESSION AND POLYNOMIAL (DEGREE 2, 3, 4)
# ============================================================
# Each model is a pipeline: scale -> (polynomial expansion) -> linear regression.
# Scaling first keeps the polynomial terms numerically stable (Year^4 would be huge).
# The pipeline fits the scaler on the TRAIN data only.
models = {'Linear': make_pipeline(StandardScaler(), LinearRegression())}
for d in [2, 3, 4]:
    models[f'Poly deg {d}'] = make_pipeline(
        StandardScaler(), PolynomialFeatures(degree=d, include_bias=False), LinearRegression()
    )

rows = []
for name, m in models.items():
    m.fit(X_train, y_train)
    train_r2 = r2_score(y_train, m.predict(X_train))
    test_pred = m.predict(X_test)
    rows.append({
        'Model': name,
        'Train R2': train_r2,
        'Test R2': r2_score(y_test, test_pred),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, test_pred)),
    })

results = pd.DataFrame(rows).set_index('Model')
print("\nComparison (all chosen features):")
print(results.round(4))
print("\nBest test R2:", results['Test R2'].idxmax())

# ============================================================
# 4. PLOT FITTED CURVES FOR DIFFERENT DEGREES
# ============================================================
# A curve can only be drawn against ONE input, so the plot uses the single feature most
# correlated with Price. These curves come from single-feature models, so their scores
# differ from the multi-feature table above.
plot_feature = data.corr()['Price'].drop('Price').abs().idxmax()
print("\nCurves plotted against:", plot_feature)

Xp_train = X_train[[plot_feature]]
grid = pd.DataFrame({plot_feature: np.linspace(X[plot_feature].min(), X[plot_feature].max(), 300)})

plt.figure(figsize=(9, 6))
plt.scatter(X[plot_feature], y, color='gray', alpha=0.4, label='Actual data')

colors = {1: 'green', 2: 'orange', 3: 'red', 4: 'purple'}
for d in [1, 2, 3, 4]:
    steps = [StandardScaler()]
    if d > 1:
        steps.append(PolynomialFeatures(degree=d, include_bias=False))
    steps.append(LinearRegression())
    m = make_pipeline(*steps)
    m.fit(Xp_train, y_train)
    label = 'Linear (deg 1)' if d == 1 else f'Polynomial (deg {d})'
    plt.plot(grid[plot_feature], m.predict(grid), color=colors[d], linewidth=2, label=label)

plt.xlabel(plot_feature)
plt.ylabel('Price')
plt.title(f'Fitted curves for different degrees ({plot_feature} vs Price)')
plt.legend()
plt.grid(True)
plt.show()