import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Name": ["Ahsan", "Hira", "Bilal", "Zara", "Salman", "Mahnoor"],
    "Age": [25, 27, 35, 29, None, 40],
    "Salary": [50000, None, 75000, 2000000, 60000, 90000],
    "Department": ["IT", "Finance", "IT", "HR", "Finance", "IT"]
}
df = pd.DataFrame(data)

print("Original data:")
print(df)
print("\nMissing values:")
print(df.isnull().sum())

# 1. Handle missing values
# Median, not mean, for both columns. Salary contains an extreme value (2,000,000),
# which would drag the mean far up and give a badly inflated fill value.
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Salary'] = df['Salary'].fillna(df['Salary'].median())

print("\nAfter handling missing values:")
print(df)

# 2. Encode Department (nominal, no order -> one-hot)
df_encoded = pd.get_dummies(df, columns=['Department'], dtype=int)
print("\nAfter one-hot encoding Department:")
print(df_encoded)

# 3. Boxplot of Salary
plt.figure(figsize=(6, 5))
plt.boxplot(df['Salary'])
plt.ylabel('Salary')
plt.title('Boxplot of Salary')
plt.show()

Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = df[(df['Salary'] < lower) | (df['Salary'] > upper)]

print(f"\nIQR limits: {lower:.0f} to {upper:.0f}")
print("Outliers:")
print(outliers[['Name', 'Salary', 'Department']])

# 4. Comment on Zara's salary
print("""
Comment on Zara's salary:
Statistically, Zara's salary (2,000,000) is an outlier. It is far above the upper
IQR limit and about 20x the typical salary in this dataset. Statistical detection
only says the value is unusual, not that it is wrong. Before removing it, check
the source: it could be a data-entry error (extra zeros) or a real senior/executive
salary. If it is an error, correct or remove it. If it is genuine, keep it and use
robust methods (median, log transform, RobustScaler) so it does not distort the model.
""")