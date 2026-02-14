# Loan Approval Probability Prediction System


## Overview

The project is a machine learning based system that predicts if an individual will be approved for a loan based on financial and demographic data provided by the applicant. It does this by assigning probabilities to each loan type using the applicant's data. Instead of simply providing a Yes/No response, you can use the confidence score generated from the data to make data-driven decisions.



## Features

* Accept required applicant data
* Predict approval probability (%) as well as rejection probability (%)
* Binary classification (0 = approve, 1 = reject)
* Provides real-time decision support



## How It Works

1) The user enters the application details requested
2) Preprocess and encode data
3) Apply the trained ML model to the application
4) Output defined probabilities

Example Output:
  
```
---- Loan Approval Result ----
🎉 Congratulations! Your loan is likely to be APPROVED.
Approval probability: 99.43%
```

The final decision can be made based on a configurable threshold.


## Workflow

User Input > Data Preprocessing > Model Prediction > Probability Output 



## Technology Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Logistic Regression/Random Forest



## Model Details

Binary Classification 

Target : 
    0 = Approved,
    1 = Rejected

Will use the "predict_proba()" function to calculate confidence scores for two potential outcomes.



## Evaluation Metrics
Accuracy,
Precision,
F1 score,
ROC-AUC Score,
Confusion Matrix
