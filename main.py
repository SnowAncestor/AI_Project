import pandas as pd

file_path = 'data/ObesityDataSet.csv'

data = pd.read_csv(file_path)

print(data.head())

data.describe()