import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Dataset (approximate values, for the exercise only)
df = pd.DataFrame({
    'Province': ['Punjab', 'Sindh', 'Khyber Pakhtunkhwa', 'Balochistan'],
    'Population': [110.0, np.nan, 35.5, 12.3],       # millions, Sindh missing
    'Literacy Rate': [66.3, 61.6, np.nan, 43.6],     # %, KP missing
    'Region': ['East', 'South', 'North', 'West']
})

print("Original data:")
print(df)
print("\nMissing values per column:")
print(df.isnull().sum())

# 1. Handle missing values
# Fill Sindh's missing Population with the median (done first, while all 4 rows exist)
df['Population'] = df['Population'].fillna(df['Population'].median())
# Drop the row (KP) whose Literacy Rate is missing
df = df.dropna(subset=['Literacy Rate']).reset_index(drop=True)

print("\nAfter handling missing values:")
print(df)

# 2. Encode Region
# Label Encoding
le = LabelEncoder()
df['Region_Label'] = le.fit_transform(df['Region'])

# One-Hot Encoding
region_onehot = pd.get_dummies(df['Region'], prefix='Region', dtype=int)
df_encoded = pd.concat([df, region_onehot], axis=1)

print("\nAfter Label + One-Hot Encoding:")
print(df_encoded)

# 3. Scatter plot: Population vs Literacy Rate (label each province)
plt.figure(figsize=(7, 5))
plt.scatter(df['Population'], df['Literacy Rate'], color='blue')
for i in range(len(df)):
    plt.annotate(df['Province'][i], (df['Population'][i], df['Literacy Rate'][i]),
                 textcoords="offset points", xytext=(5, 5))
plt.xlabel('Population (millions)')
plt.ylabel('Literacy Rate (%)')
plt.title('Population vs Literacy Rate')
plt.grid(True)
plt.show()

# 4. Outlier detection in Literacy Rate (IQR method)
Q1 = df['Literacy Rate'].quantile(0.25)
Q3 = df['Literacy Rate'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df['Literacy Rate'] < lower) | (df['Literacy Rate'] > upper)]
print(f"\nIQR limits: {lower:.2f} to {upper:.2f}")
print("Outlier provinces:")
print(outliers if not outliers.empty else "None detected")
print("\nLowest literacy:", df.loc[df['Literacy Rate'].idxmin(), 'Province'])