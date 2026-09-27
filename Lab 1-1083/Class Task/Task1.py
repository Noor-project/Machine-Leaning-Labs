import pandas as pd 
#Create DataFrame from dictionary 
data = { 
'Name' : 
['Alice', 'Bob' ,'Charlie', 'Diana '],
'Age': [25, 30, 35, 28],   
'City':['Lahore , ',' Faisalabad ','Karachi', 'Multan'], 

'Salary': [50000, 60000, 70000, 55000] 
}
df_manual=pd. DataFrame(data) 
print("Manua1 DataFrame: ")
print (df_manual )
