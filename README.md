# Random Forest Algorithm Implementation for Classification  

## 1. Overview  
This tutorial provides a beginner-friendly guide to implementing the **Random Forest Algorithm** in machine learning by solving a classification problem. The dataset used for this tutorial is from Kaggle, and we will cover:  

- Core concepts of **Random Forest**  
- **Loading and preprocessing** the dataset  
- **Visualization** based on various factors  
- **Training the model**  
- **Evaluating model performance**  

---

## 2. Tools and Libraries  

The following tools and libraries are used:  

- **Python** – Programming language  
- **Google Colab** – Coding environment  
- **Scikit-learn** – Machine learning library  
- **Pandas** – Data manipulation  
- **NumPy** – Handling numerical data  
- **Matplotlib/Seaborn** – Data visualization  

---

## 3. Dataset Selection  

The dataset is sourced from **Kaggle: Lung Cancer Risk in 25 Countries**  
🔗 [Dataset Link](https://www.kaggle.com/datasets/aizahzeeshan/lung-cancer-risk-in-25-countries)  

### Dataset Description  
- **Total Rows:** 220,632  
- **Total Columns:** 24  

### Columns in the dataset:  
`ID`, `Country`, `Population_Size`, `Age`, `Gender`, `Smoker`, `Years_of_Smoking`, `Cigarettes_per_Day`, `Passive_Smoker`, `Family_History`, `Lung_Cancer_Diagnosis`, `Cancer_Stage`, `Survival_Years`, `Adenocarcinoma_Type`, `Air_Pollution_Exposure`, `Occupational_Exposure`, `Indoor_Pollution`, `Healthcare_Access`, `Early_Detection`, `Treatment_Type`, `Developed_or_Developing`, `Annual_Lung_Cancer_Deaths`, `Lung_Cancer_Prevalence_Rate`, `Mortality_Rate`  

---

## 4. Data Pre-processing  

### 4.1 Checking Missing Values  
We use `isnull().sum()` to check for missing values in the dataset.  

### 4.2 Statistical Analysis  

A statistical summary of the dataset helps in understanding distributions, mean values, and identifying any potential data inconsistencies.  

### 4.3 Encoding Categorical Data  

#### **Label Encoding**  
Performed on categorical columns with only **two** categories:  
```python
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

cancer_df['Smoker'] = labelencoder.fit_transform(cancer_df['Smoker'])
cancer_df['Passive_Smoker'] = labelencoder.fit_transform(cancer_df['Passive_Smoker'])
cancer_df['Lung_Cancer_Diagnosis'] = labelencoder.fit_transform(cancer_df['Lung_Cancer_Diagnosis'])
