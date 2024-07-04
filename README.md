# Bank Customer Churn Prediction

This project aims to predict bank customer churn using various machine learning models and extensive feature engineering. The primary goal is to identify customers who are likely to close their bank accounts, enabling the bank to take proactive measures to retain them.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Exploration](#data-exploration)
- [Data Visualization](#data-visualization)
- [Feature Engineering](#feature-engineering)
- [Data Preprocessing](#data-preprocessing)
- [Model Training and Tuning](#model-training-and-tuning)
- [Evaluation](#evaluation)
- [Conclusion](#conclusion)
- [How to Use](#how-to-use)
- [Dependencies](#dependencies)

## Introduction

Customer churn prediction is crucial for banks to retain customers and reduce losses. By identifying customers who are likely to churn, banks can implement targeted retention strategies. This project uses machine learning techniques to predict customer churn based on various features.

## Dataset

The dataset contains information about bank customers, including features such as age, credit score, balance, estimated salary, and whether they are active members or have credit cards. There are no missing or duplicate values in the dataset.

## Data Exploration

During data exploration, we examined the distribution and relationships of features. This helped in understanding the underlying patterns and correlations in the data.

## Data Visualization

We visualized the data to identify skewness and patterns. The visualizations revealed that certain features were skewed, which was addressed in the data preprocessing step through data augmentation.

## Feature Engineering

We focused on creating new features from highly correlated ones, such as age, credit score, activity status, and the number of products. These engineered features enhanced the model's ability to predict churn.

## Data Preprocessing

We computed the correlation matrix and Predictive Power Score (PPS) to identify important features. High correlations and PPS scores for features like age, credit score, activity status, and the number of products guided our feature engineering efforts.

## Data Splitting

To prevent data leakage, we applied SMOTE (Synthetic Minority Over-sampling Technique) only to the training data. This technique balanced the classes, and we used a sampling strategy with a ratio of 0.5 to prevent any issues.

## Model Training and Tuning

### Base Neural Network Model

We started with a base neural network model to establish a benchmark. This model provided initial insights into the dataset's complexity and the importance of feature engineering.

### Model Tuning

The tuning process aimed to improve model performance by finding the best combination of parameters.

## Evaluation

The models were evaluated using metrics like accuracy and F1 score. Our best model achieved an accuracy of 86.1% and an F1 score of 0.64. The F1 score can be adjusted based on the classification threshold to balance precision and recall.

## Conclusion

In conclusion, our extensive preprocessing, feature engineering, and model tuning efforts resulted in a highly accurate model for predicting bank customer churn. The achieved accuracy of 86.1% and the F1 score of 0.64 demonstrate the model's effectiveness. Adjusting the classification threshold allows us to tailor the model to specific business needs.

## How to Use

1. Clone the repository.
2. Install the required dependencies.
3. Run the notebook or script to preprocess the data, train the models, and evaluate their performance.

## Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- keras-tuner

Make sure to install the dependencies using `pip`:

```sh
pip install pandas numpy scikit-learn seaborn matplotlib keras-tuner
