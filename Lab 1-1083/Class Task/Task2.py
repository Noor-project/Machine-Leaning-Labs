import pandas as pd 
#Load CSV from local file 
df_csv =pd. read_csv ('Lab 1-1083/cars.csv')  
print("CSV Data:" )
print(df_csv.head())
#Load CSV from online source 
online_csv_url = 'https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv'
df_online_csv = pd. read_csv(online_csv_url) 
print("\nOn1ine CSV Data: ")
print(df_online_csv.head())
