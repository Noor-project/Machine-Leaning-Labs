import pandas as pd

data = pd.DataFrame({'Income': [25000, 27000, 26000, 28000, 500000, 24000]})

Q1 = data['Income'].quantile(0.25)
Q3 = data['Income'].quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = data[(data['Income'] < lower_limit) | (data['Income'] > upper_limit)]
print(outliers)
data_cleaned = data[(data['Income'] >= lower_limit) & (data['Income'] <= upper_limit)]
print(data_cleaned)
import numpy as np

data['Income_Log'] = np.log(data['Income'])
print(data)