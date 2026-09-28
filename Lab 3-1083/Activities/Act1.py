import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, roc_auc_score)

# ============================================================
# 1. LOAD DATASET
# ============================================================
BASE = os.path.dirname(os.path.abspath(__file__))

# Set CSV_PATH ONLY if the automatic search below fails. Use a raw string, e.g.
# CSV_PATH = r"C:\Users\btwit\Documents\Semester 7\Machine Learning\Lab\Machine-Leaning-Labs\Lab 3-1083\headbrain.csv"
CSV_PATH = None

# Otherwise look for a CSV with "headbrain" (or "head" / "brain") in its name in: the
# script's folder, the folder above it, and the folder VS Code is running from.
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

data = pd.read_csv(csv_path)
data.columns = data.columns.str.strip()
print("Shape:", data.shape)
print(data.head())
print("\nMissing values:", data.isnull().sum().sum())
print("\nClass distribution of Gender:")
print(data['Gender'].value_counts().sort_index())

# ============================================================
# 1 (cont.) PREPROCESS: encode categorical variables
# ============================================================
# Target: Gender has two values (1 and 2). Encode as 0 / 1 so the model has a 0/1 label.
# Class 0 = Gender 1, class 1 = Gender 2.
y = data['Gender'].map({1: 0, 2: 1})

# Age Range is categorical (1 and 2, not a quantity), so one-hot encode it.
# drop_first=True leaves one 0/1 column: 'Age Range_2'.
X = pd.get_dummies(data[['Age Range', 'Head Size(cm^3)', 'Brain Weight(grams)']],
                   columns=['Age Range'], drop_first=True, dtype=int)
print("\nFeatures after encoding:", X.columns.tolist())

# Split first (stratify keeps the class balance the same in train and test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the numeric features (fit on train only, apply to test)
num_cols = ['Head Size(cm^3)', 'Brain Weight(grams)']
scaler = StandardScaler()
X_train = X_train.copy()
X_test = X_test.copy()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# ============================================================
# 2. TRAIN LOGISTIC REGRESSION
# ============================================================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]     # probability of class 1, needed for the ROC curve

# ============================================================
# 3. EVALUATE
# ============================================================
print("\nAccuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))
print("ROC AUC  :", roc_auc_score(y_test, y_prob))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ============================================================
# 4. PLOT CONFUSION MATRIX AND ROC CURVE
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=['Gender 1', 'Gender 2'], cmap='Blues', ax=axes[0]
)
axes[0].set_title('Confusion Matrix')

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
axes[1].plot(fpr, tpr, color='blue', linewidth=2, label=f'Logistic Regression (AUC = {auc:.3f})')
axes[1].plot([0, 1], [0, 1], color='red', linestyle='--', label='Random guessing')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()