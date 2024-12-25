from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

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

X = data.drop("NObeyesdad", axis=1)  #تم حذف الليبل الذي من المفترض ان يتنبئ به النموذج
Y = data["NObeyesdad"]  #الهدف الذي نريد التنبؤ به وهو مخزن في y
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
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, stratify=Y)

#الآن اختيار الخوارزمية المناسبة او تجربة عدة خوارزميات 

#max_depth هنا وضعناها لأننا بدونها الشجرة ستتفرع بشكل مستمر وحينها قد تكون النتيجة غير دقيقة او
#overfitting
model = DecisionTreeClassifier(max_depth=3)
model=model.fit(X_train, Y_train)


#cross validation : الطريقة الي تعلمناها عشان نخلي توزيع البيانات متوازن
#بهذه الطريقة، يتم تقسيم البيانات إلى 10 أقسام، ويتم تدريب النموذج على 9 أقسام واختباره على القسم العاشر
#shuffle: معناه انه بكل مرة سيتم تغيير عينة الاختبار وعينات التدريب بشكل عشوائي
cv = StratifiedKFold(n_splits=10, shuffle=True)


#هنانستخدمه للتدريب ونجرب فيه
cross_val_scores = cross_val_score(model, X, Y, cv=cv, scoring='accuracy')

print("\nCross-Validation Accuracy Scores:")
print(cross_val_scores)
print(f"Mean Accuracy from Cross-Validation: {cross_val_scores.mean():.4f}")

# استخدام النموذج للتنبؤ على بيانات الاختبار
Y_pred = model.predict(X_test)


#precision : ببساطة بيظهر لك مدى دقة النموذج في التنبؤ بالنتائج الصحيحة
#هنا بيتحقق كم نتيجة جابها صح مقارنة بالنتائج الخاطئة
#هنا ممكن يصنف شيء صح وهو خطأ
precision = precision_score(Y_test, Y_pred, average='weighted')

#recall : بيظهر لك مدى قدرة النموذج على العثور على جميع النتائج الصحيحة ولكن الفرق بينه وبين السابق انه
#يقارن بالنسبة لكم نتيجة صحيحة تم تصنيفها خاطئة والمفترض ان تكون صحيحة
#هنا ممكن يصنف شيء خطأ بس هو صح
recall = recall_score(Y_test, Y_pred, average='weighted')

#بالنسبة لهذا النوع فهو خليط بين الإثنين السابقين، وهدفه هو مقارنة النتائج الصحيحة بالنتائج الخاطئة
#يعني بالأول كان بيجيبلك كم نتيجة صح مقارنة بكم نتيجة خطا
#والثاني كم نتيجة صح جابها مقارنة بكم نتيجة خطأ المفترض تكون صح
#اما هذا مقارنة بين الاثنين الصح والخطأ
f1 = f1_score(Y_test, Y_pred, average='weighted')


#الشرح بشكل عام، هسا النتيجة الي بالأول هي نسبة تحديده للصح من بين الأخطاء، يعني كل ما كانت اعلى يعني هو تنبئ بنسبة عالية صح مقارنة بالخطأ
#الثاني كل ما كان اعلى يعني تنبئ بنسبة عالية صح مقارنة بالنتائج الصحيحة اللي مفترض تكون صح
#الثالث هو مزيج بين الاثنين السابقين

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')

print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))

