from sklearn.preprocessing import LabelEncoder
import pandas as pd

file_path = 'Data/ObesityDataSet.csv'

data = pd.read_csv(file_path)

#حذف الداتا المكررة
#inplace بمعنى ستعدل على الداتا الأصلية نفسها ولن ينشئ نسخة جديدة
data.drop_duplicates(inplace=True)

#تحويل القيم النصية الى ارقام
#انشاء اوبجكت من label encoder
label_encoder = LabelEncoder()
#الأعمدة التي تم تحويلها لأرقام
columns_to_encode = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS",
    "NObeyesdad",
]

#تحويل كل عمود وقيمه الفريدة لأرقام
#مثال لو كان لدي [a,b,c,d]
#سيقوم بجعله [0,1,2,3] للقيم التي غير متكررة
#fit :يعمل على إيجاد القيم الفريدة الغير مكررة
#transform : يحول القيم التي يجدها لأرقام
for column in columns_to_encode:
    data[column] = label_encoder.fit_transform(data[column])
        # معرفة القيم كيف تحولت لأرقام
    print(f"\nColumn: {column}")
    print("Text values and their corresponding numeric codes:")
    for text, num in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
        print(f"{text} -> {num}")
        
# عرض البيانات المعدلة
print("\nData after Label Encoding:")
print(data.head())  # عرض أول 5 صفوف من البيانات بعد الترميز


