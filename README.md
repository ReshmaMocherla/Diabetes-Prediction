# Diabetes Prediction Using Machine Learning

## Project Overview

This project uses machine learning to predict whether a person may have diabetes based on diagnostic measurements. It explores data preprocessing, missing-value treatment, model comparison, hyperparameter tuning, and model evaluation.

The project was developed using Python and Jupyter Notebook as a practical application of machine learning classification techniques.

## Objectives

- Explore and understand the diabetes dataset.
- Clean and preprocess the data before model training.
- Handle potentially missing measurements represented by zero values.
- Train and compare multiple classification models.
- Improve model performance using `GridSearchCV` and cross-validation.
- Evaluate the selected model using classification metrics and a confusion matrix.
- Save and reload the trained model for predictions.

## Dataset

**Dataset:** Pima Indians Diabetes Dataset  
**File:** `diabetes.csv`

The dataset contains 768 records and 8 input features, along with a target column named `Outcome`.

The input features include:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

The target variable, `Outcome`, indicates the diabetes classification: `0` for non-diabetic and `1` for diabetic.

## Technologies Used

- Python
- Jupyter Notebook
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib

## Project Workflow

1. **Data loading:** Load the CSV dataset into a Pandas DataFrame.
2. **Exploratory data analysis:** Examine the dataset, feature distributions, and relationships between variables.
3. **Data preprocessing:** Investigate zero values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI, treating clinically implausible zeros as missing values where appropriate. Impute the affected values before training.
4. **Model training:** Train and compare seven classification models.
5. **Hyperparameter tuning:** Use `GridSearchCV` with 5-fold cross-validation to tune model parameters.
6. **Model evaluation:** Evaluate the selected model using accuracy, precision, recall, F1-score, and a confusion matrix.
7. **Model persistence:** Save the trained model using Joblib and verify that it can be loaded for predictions.

## Model Evaluation

The tuned Random Forest classifier achieved the following results.

### Cross-validation performance

| Metric | Score |
|---|---:|
| Accuracy | 76.55% |
| Precision | 63.64% |
| Recall | 77.10% |
| F1-score | 69.67% |

### Test-set performance

| Metric | Score |
|---|---:|
| Accuracy | 74.68% |
| Precision | 60.87% |
| Recall | 77.78% |
| F1-score | 68.29% |

### Confusion Matrix

| | Predicted: No Diabetes | Predicted: Diabetes |
|---|---:|---:|
| Actual: No Diabetes | 73 | 27 |
| Actual: Diabetes | 12 | 42 |

The model correctly classified 73 non-diabetic cases and 42 diabetic cases in the test set. It also produced 27 false positives and 12 false negatives.

These results illustrate the trade-offs between identifying positive cases and avoiding incorrect positive predictions.

## Saved Model

The trained model was saved as:

`diabetes_prediction_model.pkl`

Joblib was used to save and reload the model, and a sample prediction was tested successfully.

## How to Run the Project

1. Clone or download this repository.
2. Install the required libraries:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
   ```

3. Ensure `diabetes.csv` is available at the file path used in the notebook.
4. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open `Diabetes-Prediction.ipynb` and execute the cells in order.

## Key Learnings

- Data cleaning and preprocessing for real-world-style datasets.
- Exploratory data analysis and feature understanding.
- Classification model comparison.
- Hyperparameter optimization using GridSearchCV.
- Model evaluation using multiple performance metrics.
- Saving and reusing trained machine learning models.

## Limitations

This project is an educational machine learning exercise, not a clinically validated diagnostic system. Its predictions should not be used as a substitute for professional medical evaluation. Model performance may vary across different populations and datasets.

## Future Improvements

- Perform additional feature analysis and model comparison.
- Evaluate calibration and performance across different patient groups.
- Improve reproducibility by documenting the complete preprocessing and prediction pipeline.
- Develop an interactive application for demonstrating predictions, with appropriate medical disclaimers.

---

**Author:** Reshma Mocherla  
**Project:** Diabetes Prediction Using Machine Learning
