import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Data
features_df = pd.DataFrame({'Regionname': ['North', 'South', 'East', 'West', 'North']})

# Apply Label Encoding
le = LabelEncoder()
features_df['Region'] = le.fit_transform(features_df['Regionname'])

print("\nLabel Encoded Regions:")
print(features_df.value_counts())

# One-hot encoding
features_df['Method'] = ['Method1', 'Method2', 'Method1', 'Method3', 'Method2']
df_one_hot = pd.get_dummies(features_df['Method'], dtype=int)

print("\nOne Hot Encoded DataFrame:")
print(df_one_hot)