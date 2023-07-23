Credit Card Fraud Detection using PyCaret

Overview
The Credit Card Fraud Detection using PyCaret project aims to implement anomaly detection techniques to identify fraudulent transactions in credit card data. We will leverage both traditional Machine Learning (ML) techniques and AutoML using the PyCaret library in this notebook.


Dataset Description
The dataset contains credit card transactions made by European cardholders in September 2013. It comprises 284,807 transactions, with only 492 being fraudulent. The dataset is highly unbalanced, with fraudulent transactions accounting for only 0.172% of the total.
The features in the dataset are numerical and have been transformed using Principal Component Analysis (PCA), except for 'Time' and 'Amount'. The 'Time' feature represents the time elapsed between each transaction and the first transaction in the dataset. The 'Amount' feature denotes the transaction amount. The response variable 'Class' takes a value of 1 in case of fraud and 0 otherwise.
Due to confidentiality issues, we cannot provide the original features or additional background information about the data.


Steps Involved
1. Data Analysis
We will start by performing exploratory data analysis to gain insights into the dataset. This step includes visualizing data distributions, detecting outliers, and understanding feature correlations.
2. Feature Engineering
To enhance the predictive power of our models, we will perform feature engineering. This involves handling missing values, transforming data, and creating new relevant features.
3. Model Building and Prediction using ML Techniques
In this step, we will train various ML models, such as logistic regression, decision trees, random forests, and others, to detect fraudulent transactions. We will evaluate their performance using appropriate metrics, considering the class imbalance.
4. Model Building and Prediction using PyCaret (Auto ML)
PyCaret is an AutoML library that automates the process of training and tuning ML models. We will leverage PyCaret to build a more sophisticated ensemble of models, effectively handling the class imbalance and optimizing performance.


Evaluation Metrics
Due to the class imbalance, we will use the Area Under the Precision-Recall Curve (AUPRC) as the primary evaluation metric. The confusion matrix accuracy is not meaningful for unbalanced classification tasks.
