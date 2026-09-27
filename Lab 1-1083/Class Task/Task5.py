import pandas as pd 
#Load text file with custom delimiter 
df_text = pd. read_csv( 'Lab 1-1083/data.txt',delimiter='\t') 
print("Text File Data:") 
print(df_text . head())  