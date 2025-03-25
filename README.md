1.	Overview:

In this assignment creating a beginner friendly tutorial on Random Forest Algorithm implementation in machine learning by solving a classification problem. For this tutorial I used a dataset from Kaggle and explained the core concepts of Random Forest, loading of dataset, pre-processing on dataset and its visualization based on different factors and training the model and its evaluation to check model performance.

2.	Tools and Libraries:
●	Python: Programming language 
●	Google Colab: Coding Editor
●	Scikit-learn: Machine Learning Libraries
●	Pandas: For data manipulation.
●	Numpy: Handling and processing numerical data in python
●	Matplotlib/Seaborn: For Visualization, 


3.	Dataset Selection:
Dataset is selected from Kaggle Lung Cancer Risk in 25 Countries
https://www.kaggle.com/datasets/aizahzeeshan/lung-cancer-risk-in-25-countries

Dataset Description: Dataset has total 220632 rows and  24 columns

Dataset Columns:
'ID',  'Country',  'Population_Size',  'Age',  'Gender',  'Smoker',  'Years_of_Smoking', 'Cigarettes_per_Day', 'Passive_Smoker', 'Family_History', 'Lung_Cancer_Diagnosis', 'Cancer_Stage', 'Survival_Years', 'Adenocarcinoma_Type', 'Air_Pollution_Exposure', 'Occupational_Exposure', 'Indoor_Pollution', 'Healthcare_Access', 'Early_Detection', 'Treatment_Type', 'Developed_or_Developing', 'Annual_Lung_Cancer_Deaths', 'Lung_Cancer_Prevalence_Rate', 'Mortality_Rate'


4.	Data Pre-processing:
Checking missing values in the dataset using .isnull().sum() which shows a comprehensive summary of dataset if there’s any missing value in dataset.




4.1. Statistical Analysis:
 

4.2. Encoding of Categorical data:

Label encoding performed on there columns Smoker, Passive_Smoker, Lung_Cancer_Diagnosis because it has only two categories, like: yes/no, 1/0
cancer_df['Smoker'] = labelencoder.fit_transform(cancer_df['Smoker'])
cancer_df['Passive_Smoker'] = labelencoder.fit_transform(cancer_df['Passive_Smoker'])
cancer_df['Lung_Cancer_Diagnosis'] = labelencoder.fit_transform(cancer_df['Lung_Cancer_Diagnosis'])

●	One-hot encoding performed on these columns which has more than two categories
cat_columns = [
    'Gender',
    'Country',
    'Family_History',
    'Adenocarcinoma_Type',
    'Air_Pollution_Exposure',
    'Occupational_Exposure',
    'Indoor_Pollution',
    'Healthcare_Access',
    'Early_Detection',
    'Developed_or_Developing',
    'Cancer_Stage',
    'Treatment_Type'
]



encoded_new_df = pd.get_dummies(cancer_df, columns=cat_columns, drop_first=True)
encoded_new_df.head()

5.	Feature Selection:
Column → Lung_Cancer_Diagnosis selected for feature/ target of the dataset

# All columns of the dataset except column Lung_Cancer_Diagnosis.

X = encoded_new_df.drop(columns=['Lung_Cancer_Diagnosis'], axis=1)

# Label of the dataset.

y = encoded_new_df['Lung_Cancer_Diagnosis']


6.	Model Implementation

Random forest Classifier implemented by splitting the dataset in 20% for testing and 80% for training.

model = RandomForestClassifier(class_weight='balanced', random_state=42)

model.fit(X_train,y_train)


7.	Model Evaluation

By implementing a confusion matrix  and classification report evaluates the model and it gives high accuracy and precision.

7.1. Confusion Matrix:
True Negatives (TN) = 42325 --> Model correctly predicted that 42325 people don't have lung cancer.

False Positives (FP) = 0 --> Model not predict lung cancer, when the person didn't have it.

False Negatives (FN) = 0 --> Model not missed actual lung cancer.

True Positives (TP) = 1792 --> Model correctly predicted 1792 people had lung cancer.

True Negatives (TN) = 42325 --> Model correctly predicted that 42325 people don't have lung cancer.

False Positives (FP) = 0 --> Model not predict lung cancer, when the person didn't have it.
False Negatives (FN) = 0 --> Model not missed actual lung cancer.

True Positives (TP) = 1792 --> Model correctly predicted 1792 people had lung cancer.

7.2. Classification Report:

 


![image](https://github.com/user-attachments/assets/7f6abc08-4d14-451f-9952-9f4e24aff062)
