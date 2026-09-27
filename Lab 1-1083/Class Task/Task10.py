from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

data = pd.DataFrame({'Education': ['High School', 'Bachelors', 'Masters', 'PhD',]})
# Manually defining the order
order = [['High School', 'Bachelors', 'Masters', 'PhD']]

oe = OrdinalEncoder(categories=order)
data['Education_Encoded'] = oe.fit_transform(data[['Education']])

print(data)