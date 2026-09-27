import pandas as pd
from sklearn.decomposition import PCA

# Sample Dataset
data = pd.DataFrame({
    'Height_cm': [150, 160, 170, 180, 190],
    'Height_inch': [59, 63, 67, 71, 75],   # highly correlated with Height_cm
    'Weight': [50, 60, 70, 80, 90],
    'Date': ['2026-01-05', '2026-02-10', '2026-03-15', '2026-04-01', '2026-05-20']
})

# 1. Feature Selection
# Remove features that are highly correlated with another feature (redundant information)
correlation = data[['Height_cm', 'Height_inch', 'Weight']].corr()
print("Correlation Matrix:\n", correlation)
# Height_cm and Height_inch are almost perfectly correlated, so drop one
data = data.drop('Height_inch', axis=1)

# 2. Feature Engineering
# Create new, more useful features from existing ones
# Example A: combine Height and Weight into BMI
data['BMI'] = data['Weight'] / ((data['Height_cm'] / 100) ** 2)
# Example B: extract Day of Week from the Date column
data['Date'] = pd.to_datetime(data['Date'])
data['Day_of_Week'] = data['Date'].dt.day_name()
print("\nData after Feature Engineering:\n", data)

# 3. Dimensionality Reduction (PCA)
# Reduce number of numeric features while preserving most of the important information
numeric_features = data[['Height_cm', 'Weight', 'BMI']]
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(numeric_features)

print("\nData after PCA (reduced to 2 components):\n", reduced_data)