# Diabetes Prediction Using Logistic Regression

This project demonstrates how to build a machine learning model using Logistic Regression to predict whether a patient has Diabetes based on several health-related factors. The dataset used in this project is sourced from Kaggle and contains various attributes related to diabetes such as glucose levels, BMI, age, and insulin levels.

## Project Overview

The goal of this project is to develop a binary classification model to predict if a patient has Diabetes (1) or No Diabetes (0). The model is built using the Logistic Regression algorithm, which is well-suited for binary classification tasks.

## Dataset
Source: Kaggle - Diabetes Dataset
Number of Entries: 768
Features: 8 features and 1 target variable (Outcome).

## Features:
Pregnancies: Number of times the patient has been pregnant.
Glucose: Plasma glucose concentration after 2 hours in an oral glucose tolerance test.
BloodPressure: Diastolic blood pressure (mm Hg).
SkinThickness: Triceps skin fold thickness (mm).
Insulin: 2-hour serum insulin (mu U/ml).
BMI: Body mass index (weight in kg / height in m²).
DiabetesPedigreeFunction: A function that represents the likelihood of diabetes based on family history.
Age: Age of the patient.
Outcome: Target variable where 1 indicates the patient has diabetes and 0 indicates the patient does not.

## Technology Stack
Programming Language: Python
Libraries:
Pandas (for data manipulation)
Numpy (for numerical operations)
Matplotlib & Seaborn (for data visualization)
Scikit-learn (for building and evaluating the model)
IDE: Jupyter Notebook / VS Code / PyCharm

## Installation
To run this project locally, you need to have Python installed along with the following libraries:
pip install pandas numpy matplotlib seaborn scikit-learn

## How to Run
1. Clone the repository:
git clone https://github.com/ReshmaMocherla/Diabetes-Prediction.git

2. Navigate to the project directory:
cd Diabetes-Prediction

3. Run the Python script or Jupyter Notebook to train the model:
python diabetes_prediction.py

4. Alternatively, if using a Jupyter Notebook:
jupyter notebook

## Data Preprocessing
The dataset is first loaded and explored. The following preprocessing steps were applied:
Missing Data: No missing data was found in the dataset.
Feature Scaling: The features were scaled using StandardScaler to standardize them and make them comparable across different units.
Splitting Data: The dataset was split into a training set (80%) and a test set (20%) for model training and evaluation.


## Model Building and Evaluation
The logistic regression model was trained on the training set and evaluated on the test set using the following metrics:

### Evaluation Metrics:
Accuracy: 82%
Precision (Diabetes): 0.76
Recall (Diabetes): 0.62
F1-Score (Diabetes): 0.68

### Confusion Matrix:
[[98  9]
 [18 29]]
A heatmap visualization of the confusion matrix is included to analyze the performance of the model more clearly.

### Results:
Accuracy: 82% - The model correctly predicted 82% of the instances.
Precision: 0.76 - 76% of the patients predicted to have diabetes actually had diabetes.
Recall: 0.62 - The model correctly identified 62% of patients who actually have diabetes.
F1-Score: 0.68 - A balance between precision and recall.

## Future Work
Model Improvement: You can explore more advanced models like Random Forest, Support Vector Machine (SVM), or Gradient Boosting for potentially better performance.
Hyperparameter Tuning: Use GridSearchCV or RandomizedSearchCV to tune the hyperparameters of the Logistic Regression model and improve performance.
Cross-Validation: Implement cross-validation techniques to validate the model on different subsets of the data.
Deployment: The model can be deployed using Flask or FastAPI to serve real-time predictions via a web interface.

## Project Links
GitHub Repository: https://github.com/ReshmaMocherla/Diabetes-Prediction
Kaggle Dataset: Pima Indians Diabetes Dataset
