import pandas as pd

file_path = 'Data/ObesityDataSet.csv'

data = pd.read_csv(file_path)

#حذف الداتا المكررة
#inplace بمعنى ستعدل على الداتا الأصلية نفسها ولن ينشئ نسخة جديدة
data.drop_duplicates(inplace=True)

