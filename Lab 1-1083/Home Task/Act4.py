import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# Files are looked up next to this script. Change BASE if your files live elsewhere.
BASE = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE, 'students.csv')
json_path = os.path.join(BASE, 'attendance.json')
xlsx_path = os.path.join(BASE, 'extra.xlsx')

# ---------------------------------------------------------------
# Sample files are created ONLY if yours are missing.
# If you have the real files, they are used instead.
# ---------------------------------------------------------------
if not os.path.exists(csv_path):
    pd.DataFrame({'Name': ['Ali', 'Sara', 'Ahmed', 'Zara', 'Hina', 'Usman'],
                  'Marks': [78, 85, 62, 91, 55, 70]}).to_csv(csv_path, index=False)
if not os.path.exists(json_path):
    with open(json_path, 'w') as f:
        json.dump([{'Name': 'Ali', 'Attendance': 88}, {'Name': 'Sara', 'Attendance': 92},
                   {'Name': 'Ahmed', 'Attendance': 65}, {'Name': 'Zara', 'Attendance': 95},
                   {'Name': 'Hina', 'Attendance': 58}, {'Name': 'Usman', 'Attendance': 74}], f)
if not os.path.exists(xlsx_path):
    pd.DataFrame({'Name': ['Ali', 'Sara', 'Ahmed', 'Zara', 'Hina', 'Usman'],
                  'Bonus': [2, 3, 1, 5, 0, 2]}).to_excel(xlsx_path, index=False)

# 1. Load all three files
df_csv = pd.read_csv(csv_path)
df_json = pd.read_json(json_path)
df_excel = pd.read_excel(xlsx_path)   # needs: pip install openpyxl

print("students.csv:\n", df_csv)
print("\nattendance.json:\n", df_json)
print("\nextra.xlsx:\n", df_excel)

# 2. Merge into a single DataFrame: Name, Marks, Attendance, Bonus
df = df_csv.merge(df_json, on='Name').merge(df_excel, on='Name')
df = df[['Name', 'Marks', 'Attendance', 'Bonus']]
print("\nMerged DataFrame:\n", df)

# 3. Scatter plot: Marks vs Attendance, highlight attendance < 70%
low = df['Attendance'] < 70
plt.figure(figsize=(7, 5))
plt.scatter(df.loc[~low, 'Attendance'], df.loc[~low, 'Marks'], color='blue', label='Attendance >= 70%')
plt.scatter(df.loc[low, 'Attendance'], df.loc[low, 'Marks'], color='red', label='Attendance < 70%')
for _, row in df.iterrows():
    plt.annotate(row['Name'], (row['Attendance'], row['Marks']),
                 textcoords="offset points", xytext=(5, 5))
plt.axvline(70, color='gray', linestyle='--')
plt.xlabel('Attendance (%)')
plt.ylabel('Marks')
plt.title('Marks vs Attendance')
plt.legend()
plt.grid(True)
plt.show()

# 4. One-hot encoding on Name
df_onehot = pd.get_dummies(df, columns=['Name'], dtype=int)
print("\nOne-hot encoded on Name:\n", df_onehot)