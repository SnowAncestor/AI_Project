# Obesity Prediction Model 🏥

A machine learning project that predicts obesity categories using lifestyle and physical condition data. The model classifies individuals into different obesity levels based on various behavioral and demographic features.

## 📊 Overview

This project implements a machine learning model to predict obesity levels based on eating habits, physical condition, and lifestyle choices. It uses a comprehensive dataset with both categorical and numerical features to make accurate predictions across seven different obesity categories.

## ✨ Features

- Multi-class obesity prediction
- Comprehensive data preprocessing
- Feature importance analysis
- Model performance evaluation
- Cross-validation implementation
- Handling of imbalanced classes
- Interactive visualizations

## 🔧 Requirements

- Python 3.9 or higher
- Required libraries:
  ```
  numpy>=1.21.0
  pandas>=1.3.0
  scikit-learn>=1.0.0
  seaborn>=0.11.0
  matplotlib>=3.4.0
  imblearn>=0.8.0
  ```

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SnowAncestor/AI_Project.git
   cd AI_Project
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

1. Later..

## 📝 Data Dictionary

### Categorical Features

| Feature | Description | Values | Encoding |
|---------|-------------|---------|----------|
| Gender | Gender of the individual | Female, Male | 0, 1 |
| family_history_with_overweight | Family history of overweight | no, yes | 0, 1 |
| FAVC | Consumption of high-calorie food | no, yes | 0, 1 |
| CAEC | Eating between meals | Always, Frequently, Sometimes, no | 0, 1, 2, 3 |
| SMOKE | Smoking status | no, yes | 0, 1 |
| SCC | Calories monitoring | no, yes | 0, 1 |
| CALC | Alcohol consumption | Always, Frequently, Sometimes, no | 0, 1, 2, 3 |
| MTRANS | Transportation used | Automobile, Bike, Motorbike, Public_Transportation, Walking | 0, 1, 2, 3, 4 |

### Target Variable

| Feature | Description | Categories | Encoding |
|---------|-------------|------------|----------|
| NObeyesdad | Obesity level | Insufficient_Weight, Normal_Weight, Obesity_Type_I, Obesity_Type_II, Obesity_Type_III, Overweight_Level_I, Overweight_Level_II | 0, 1, 2, 3, 4, 5, 6 |

### Numerical Features

- Age (years)
- Height (meters)
- Weight (kilograms)
- FCVC (Frequency of vegetable consumption)
- NCP (Number of main meals)
- CH2O (Water consumption)
- FAF (Physical activity frequency)
- TUE (Time using technology devices)

## 📈 Model Performance

The current model achieves the following metrics:
- Accuracy: [Add your accuracy]
- F1-Score: [Add your F1-score]
- ROC-AUC: [Add your ROC-AUC]

## 🔄 Model Pipeline

1. **Data Preprocessing**
   - Handling missing values
   - Feature encoding
   - Feature scaling
   - Class balancing using SMOTE

2. **Feature Engineering**
   - BMI calculation
   - Feature interactions
   - Polynomial features

3. **Model Training**
   - Cross-validation
   - Hyperparameter tuning
   - Model evaluation


## 🙏 Acknowledgments

- [Dataset Source](In The Data File)
