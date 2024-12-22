# Obesity Prediction Project

## <span style="color: #3498db;">Description</span>
Building a predictive model to classify individuals into obesity categories.

## <span style="color: #2ecc71;">Requirements</span>
- Python 3.9+
- Libraries: 
  - numpy
  - pandas
  - scikit-learn
  - seaborn
  - matplotlib
  - imblearn

## <span style="color: #f39c12;">How to Run</span>
1. Clone the repository.
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
3. Later..

<span style="color: #e74c3c;">Column Encodings</span>

The dataset contains categorical variables, which are encoded as follows:

| Column                             | Text Value                | Numeric Code |
|------------------------------------|---------------------------|--------------|
| **Gender**                         | Female, Male              | 0, 1         |
| **family_history_with_overweight** | no, yes                   | 0, 1         |
| **FAVC (High-calorie food consumption)** | no, yes             | 0, 1         |
| **CAEC (Frequency of eating)**     | Always, Frequently, Sometimes, no | 0, 1, 2, 3 |
| **SMOKE**                          | no, yes                   | 0, 1         |
| **SCC (Symptoms of chronic conditions)** | no, yes              | 0, 1         |
| **CALC (Alcohol consumption)**     | Always, Frequently, Sometimes, no | 0, 1, 2, 3 |
| **MTRANS (Transportation)**        | Automobile, Bike, Motorbike, Public_Transportation, Walking | 0, 1, 2, 3, 4 |
| **NObeyesdad (Obesity category)**  | Insufficient_Weight, Normal_Weight, Obesity_Type_I, Obesity_Type_II, Obesity_Type_III, Overweight_Level_I, Overweight_Level_II | 0, 1, 2, 3, 4, 5, 6 |


<span style="color: #9b59b6;">Modeling</span>

The predictive model uses the encoded features listed above to determine obesity levels. Experiment with different machine learning algorithms for better performance.

