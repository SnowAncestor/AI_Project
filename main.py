import pandas as pd

# تحديد المسار للملف داخل مجلد data
file_path = 'data/ObesityDataSet.csv'

# قراءة البيانات باستخدام pandas
data = pd.read_csv(file_path)

# عرض البيانات
print(data.head())
