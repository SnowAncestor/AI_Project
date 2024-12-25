from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier

def load_and_preprocess_data(file_path):
    try:
        data = pd.read_csv(file_path)
        #حذف الداتا المكررة
        #inplace بمعنى ستعدل على الداتا الأصلية نفسها ولن ينشئ نسخة جديدة
        data.drop_duplicates(inplace=True)
        #تحويل القيم النصية الى ارقام
        #انشاء اوبجكت من label encoder
        label_encoder = LabelEncoder()
        columns_to_encode = [
            "Gender", "family_history_with_overweight", "FAVC", "CAEC",
            "SMOKE", "SCC", "CALC", "MTRANS", "NObeyesdad"
        ]
        #تحويل كل عمود وقيمه الفريدة لأرقام
        #مثال لو كان لدي [a,b,c,d]
        #سيقوم بجعله [0,1,2,3] للقيم التي غير متكررة
        #fit :يعمل على إيجاد القيم الفريدة الغير مكررة
        #transform : يحول القيم التي يجدها لأرقام
        for column in columns_to_encode:
            data[column] = label_encoder.fit_transform(data[column])
            
        return data
    except Exception as e:
        raise Exception(f"Error in data preprocessing: {str(e)}")


def train_model(data):
    X = data.drop("NObeyesdad", axis=1)#تم حذف الليبل الذي من المفترض ان يتنبئ به النموذج
    Y = data["NObeyesdad"] #الهدف الذي نريد التنبؤ به وهو مخزن في y
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
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.15, stratify=Y
    )
    

    #max_depth هنا وضعناها لأننا بدونها الشجرة ستتفرع بشكل مستمر وحينها قد تكون النتيجة غير دقيقة او
    #overfitting
    model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
    )
    
    model.fit(X_train, Y_train)
    
    return model, X_train, X_test, Y_train, Y_test, X.columns

def evaluate_model(model, X, Y, X_test, Y_test):
    
    #cross validation : الطريقة الي تعلمناها عشان نخلي توزيع البيانات متوازن
    #بهذه الطريقة، يتم تقسيم البيانات إلى 10 أقسام، ويتم تدريب النموذج على 9 أقسام واختباره على القسم العاشر
    #shuffle: معناه انه بكل مرة سيتم تغيير عينة الاختبار وعينات التدريب بشكل عشوائي
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    cv_scores = cross_val_score(model, X, Y, cv=cv, scoring='accuracy')
    
    Y_pred = model.predict(X_test)
    
    metrics = {
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),

        #precision : ببساطة بيظهر لك مدى دقة النموذج في التنبؤ بالنتائج الصحيحة
        #هنا بيتحقق كم نتيجة جابها صح مقارنة بالنتائج الخاطئة
        #هنا ممكن يصنف شيء صح وهو خطأ
        'precision': precision_score(Y_test, Y_pred, average='weighted'),

        #recall : بيظهر لك مدى قدرة النموذج على العثور على جميع النتائج الصحيحة ولكن الفرق بينه وبين السابق انه
        #يقارن بالنسبة لكم نتيجة صحيحة تم تصنيفها خاطئة والمفترض ان تكون صحيحة
        #هنا ممكن يصنف شيء خطأ بس هو صح
        'recall': recall_score(Y_test, Y_pred, average='weighted'),

        #بالنسبة لهذا النوع فهو خليط بين الإثنين السابقين، وهدفه هو مقارنة النتائج الصحيحة بالنتائج الخاطئة
        #يعني بالأول كان بيجيبلك كم نتيجة صح مقارنة بكم نتيجة خطا
        #والثاني كم نتيجة صح جابها مقارنة بكم نتيجة خطأ المفترض تكون صح
        #اما هذا مقارنة بين الاثنين الصح والخطأ
        'f1': f1_score(Y_test, Y_pred, average='weighted'),

        #الشرح بشكل عام، هسا النتيجة الي بالأول هي نسبة تحديده للصح من بين الأخطاء، يعني كل ما كانت اعلى يعني هو تنبئ بنسبة عالية صح مقارنة بالخطأ
        #الثاني كل ما كان اعلى يعني تنبئ بنسبة عالية صح مقارنة بالنتائج الصحيحة اللي مفترض تكون صح
        #الثالث هو مزيج بين الاثنين السابقين

        'test_accuracy': model.score(X_test, Y_test)

    }
    
    return metrics, Y_pred
def validate_input(value, min_val, max_val, name, required=False):
    if not value.strip() and not required:
        return None
    try:
        value = float(value)
        if not min_val <= value <= max_val:
            raise ValueError
        return value
    except ValueError:
        if required:
            raise ValueError(f"{name} must be between {min_val} and {max_val}")
        return None

def get_user_input(columns):
    default_values = {
        'gender': 0,
        'age': 25,
        'family_history_with_overweight': 0,
        'FAVC': 0,
        'FCVC': 2,
        'NCP': 3,
        'CAEC': 3,
        'SMOKE': 0,
        'CH2O': 2,
        'SCC': 0,
        'FAF': 1,
        'CALC': 3,
        'MTRANS': 4
    }
    
    input_prompts = {
        'height': ('Height (cm)', 50, 250, True),
        'weight': ('Weight (kg)', 10, 300, True),
        'gender': ('Gender (0:female, 1:male)', 0, 1, False),
        'age': ('Age', 0, 120, False),
        'family_history_with_overweight': ('Family history of obesity? (0:No, 1:Yes)', 0, 1, False),
        'FAVC': ('High-calorie food consumption? (0:No, 1:Yes)', 0, 1, False),
        'CAEC': ('Eating between meals (0:Always - 3:Never)', 0, 3, False),
        'SMOKE': ('Smoking? (0:No, 1:Yes)', 0, 1, False),
        'SCC': ('Monitor calorie intake? (0:No, 1:Yes)', 0, 1, False),
        'CALC': ('Alcohol consumption (0:Always - 3:Never)', 0, 3, False),
        'MTRANS': ('Transportation (0:Car - 4:Walking)', 0, 4, False)
    }
    
    user_data = {}
    for field, (prompt, min_val, max_val, required) in input_prompts.items():
        while True:
            value = input(f"Enter {prompt}" + (" (required): " if required else " (press Enter for default): "))
            validated_value = validate_input(value, min_val, max_val, field, required)
            
            if validated_value is None and not required:
                user_data[field] = default_values.get(field, 2)
                break
            elif validated_value is not None:
                user_data[field] = validated_value
                break
            elif required:
                print(f"Invalid input: {field} is required")
    
    # Add remaining default values
    for col in columns:
        if col not in user_data:
            user_data[col] = default_values.get(col, 2)
    
    return pd.DataFrame([user_data], columns=columns)

def predict_with_probabilities(model, input_data, categories):
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)[0]
    sorted_probs = sorted(zip(categories, probabilities), key=lambda x: x[1], reverse=True)
    return predictions[0], sorted_probs


def main():
    try:
        categories = [
            "Insufficient_Weight", "Normal_Weight", "Obesity_Type_I",
            "Obesity_Type_II", "Obesity_Type_III", "Overweight_Level_I",
            "Overweight_Level_II"
        ]
        
        print("\n=== Obesity Prediction System ===\n")
        print("Initializing model...")
        data = load_and_preprocess_data('Data/ObesityDataSet.csv')
        model, X_train, X_test, Y_train, Y_test, columns = train_model(data)
        metrics, Y_pred = evaluate_model(model, data.drop("NObeyesdad", axis=1), 
                                      data["NObeyesdad"], X_test, Y_test)
        
        print("\n=== Model Performance ===")
        print(f"Training Accuracy     : {model.score(X_train, Y_train):.2%}")
        print(f"Test Accuracy         : {metrics['test_accuracy']:.2%}")
        print(f"Cross-val Accuracy    : {metrics['cv_mean']:.2%} (±{metrics['cv_std']:.2%})")
        print(f"Precision            : {metrics['precision']:.2%}")
        print(f"Recall               : {metrics['recall']:.2%}")
        print(f"F1-Score             : {metrics['f1']:.2%}")
        
        print("\n=== Personal Information Entry ===")
        print("(Required: height and weight, optional: other fields)")
        user_input = get_user_input(columns)
        
        prediction, probabilities = predict_with_probabilities(model, user_input, categories)
        
        print("\n=== Prediction Results ===")
        print(f"Primary Prediction: {categories[prediction]}")
        print("\nProbability Distribution:")
        for category, prob in probabilities:
            bars = "█" * int(prob * 20)
            print(f"{category:<20}: {bars} {prob:.1%}")
        
        print(f"\nFinal Model Accuracy: {metrics['test_accuracy']:.2%}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()