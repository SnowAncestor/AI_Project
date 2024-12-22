from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



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

X = data_name.drop("NObeyesdad", axis=1)  #تم حذف الليبل الذي من المفترض ان يتنبئ به النموذج
y = data_name["NObeyesdad"]  #الهدف الذي نريد التنبؤ به وهو مخزن في y
#x: هكذا يتعلم النموذج ان هذه القيم هي القيم التي من المفترض ان يحصل على واحد او اكثر منها ليتنبئ
#y: وهكذا يعرف أي قيمة ينبغي عليه التنبؤ بها

#x : هي الميزات التي سيتعلم بناءً عليها النموذج
#y : هو الهدف الذي ينبغي للنموذج التنبؤ به
#هنا سيختبر الميزات والهدف اولًا ولكننا فصلناهم كي نعرف أيهم هو الهدف الذي نحن بحاجة للتنبؤ به
#حددنا نسبة 15% للاختبار
#وبما اننا حددنا نسبة الاختبار فالباقي سيذهب للتعلم وهو 85% وهكذا تم التوزيع
#startify : تم استخدامها لسبب بسيط وهو ضمان أن توزيع الفئات التدريب والنتائج تكون متوازنة، فمثلا
#لو كان لدي 30 بالمئة من النتائج تخرج على أن الشخص سمنة مفرطة، فمن غير المنطقي توزيعه على 50 50
#بل من الأفضل توزيعه على 30 30 لأنه حينها ستكون النتيجة أكثر توازن
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y
)

#الآن اختيار الخوارزمية المناسبة او تجربة عدة خوارزميات 





#fit module


#معرفة ال accurcy


#اصدار النموذج

