import pandas as pd
import numpy as np

# Create sample data with numpy
np.random.seed(42)
data = {
    'ID': range(1, 101),
    'Score': np.random.randint(50, 100, 100),
    'Height': np.random.normal(170, 10, 100).round(1),
    'Weight': np.random.normal(70, 5, 100).round(1)
}

df_numpy = pd.DataFrame(data)
print("Numpy Generated Data:")
print(df_numpy.head())