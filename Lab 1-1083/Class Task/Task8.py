import pandas as pd
import numpy as np

# Create a sample dataset with missing values
data = {
    'A': [1, 2, None, 4, 5],
    'B': [6, None, 8, 9, 10],
    'C': [11, 12, 13, None, 15]
}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

# Method 1: Drop rows with missing values
df_dropna_rows = df.dropna(axis=0)

# Method 2: Drop columns with missing values
df_dropna_columns = df.dropna(axis=1)

# Method 3: Fill missing values with the median
df_fill_median = df.fillna(df.median())

print("\nDataFrame after dropping rows with missing values:")
print(df_dropna_rows)
print("\nDataFrame after dropping columns with missing values:")
print(df_dropna_columns)
print("\nDataFrame after filling missing values with the median:")
print(df_fill_median)

# Handling Categorical Attributes
# Manual Encoding Example
mapping = {'h': 1, 'u': 2, 't': 3}
features = pd.DataFrame({'Type': ['h', 'u', 't', 'h', 'u']})
features['Type'] = features['Type'].map(mapping)

print("\nEncoded Types:")
print(features.groupby('Type').size())