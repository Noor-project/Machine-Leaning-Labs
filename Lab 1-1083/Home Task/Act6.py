import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
n = 100

# Generate random data for 100 students
df = pd.DataFrame({
    'ID': range(1, n + 1),
    'Math': np.random.randint(40, 100, n).astype(float),
    'Science': np.random.randint(40, 100, n).astype(float),
    'English': np.random.randint(40, 100, n).astype(float)
})

# Deliberately plant one unusually weak student so the boxplot has something to find
df.loc[7, ['Math', 'Science', 'English']] = [5, 8, 6]

# Grade (A/B/C) based on the average of the three subjects
avg = df[['Math', 'Science', 'English']].mean(axis=1)
df['Grade'] = np.where(avg >= 75, 'A', np.where(avg >= 60, 'B', 'C'))

# Introduce missing marks (5 per subject) to practise handling them
for col in ['Math', 'Science', 'English']:
    idx = np.random.choice(n, 5, replace=False)
    df.loc[idx, col] = np.nan

print("Missing values before handling:")
print(df.isnull().sum())

# 1. Handle missing marks (fill with mean)
marks = ['Math', 'Science', 'English']
df[marks] = df[marks].fillna(df[marks].mean())
print("\nMissing values after handling:")
print(df.isnull().sum())

# 2. Encode Grade (ordinal: C < B < A)
grade_map = {'C': 0, 'B': 1, 'A': 2}
df['Grade_Encoded'] = df['Grade'].map(grade_map)
print("\nGrade encoding sample:")
print(df[['ID', 'Grade', 'Grade_Encoded']].head())

# 3. Histogram of Math scores (10 bins)
plt.figure(figsize=(7, 5))
plt.hist(df['Math'], bins=10, edgecolor='black')
plt.xlabel('Math Score')
plt.ylabel('Number of Students')
plt.title('Histogram of Math Scores')
plt.show()

# 4. Boxplot of total score to find unusual students
df['Total'] = df[marks].sum(axis=1)

plt.figure(figsize=(6, 5))
plt.boxplot(df['Total'])
plt.ylabel('Total Score')
plt.title('Boxplot of Total Scores')
plt.show()

Q1 = df['Total'].quantile(0.25)
Q3 = df['Total'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = df[(df['Total'] < lower) | (df['Total'] > upper)]
print(f"\nIQR limits for Total: {lower:.1f} to {upper:.1f}")
print("Students with unusual total scores:")
print(outliers[['ID', 'Math', 'Science', 'English', 'Total', 'Grade']])

# 5. Summary of findings
summary = df.groupby('Grade')['Total'].agg(['count', 'mean']).round(2)
print("\nAverage total score per grade:")
print(summary)
print("\nBest performing grade overall:", summary['mean'].idxmax())