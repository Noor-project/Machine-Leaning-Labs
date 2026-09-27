import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

data = pd.DataFrame({'Age': [22, 25, 47, 35, 60], 'Income': [25000, 32000, 95000, 60000, 150000]})

# Standardization: rescales data to mean 0 and standard deviation 1
standard_scaler = StandardScaler()
data_standard = standard_scaler.fit_transform(data)

# Normalization (Min-Max Scaling): rescales data into a fixed range, usually 0 to 1
minmax_scaler = MinMaxScaler()
data_minmax = minmax_scaler.fit_transform(data)