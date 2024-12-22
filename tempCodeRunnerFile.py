import pandas as pd

file_path = 'Data/ObesityDataSet.csv'

data = pd.read_csv(file_path)

print(data.head())

data.describe()