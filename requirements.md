# Requirements Document

## Introduction

This document specifies the requirements for a Customer Loan Approval Prediction System. The system uses machine learning to predict whether a customer's loan application should be approved or rejected based on financial and demographic features. The system processes loan applications containing 13 features and provides binary classification predictions (Approved/Rejected) using trained machine learning models.

## Glossary

- **System**: The Customer Loan Approval Prediction System
- **Loan_Application**: A data record containing customer information and loan details
- **Feature**: An input attribute used for prediction (e.g., income, CIBIL score)
- **Prediction**: The system's output indicating Approved or Rejected status
- **Model**: A trained machine learning classifier (Logistic Regression or Random Forest)
- **CIBIL_Score**: Credit score ranging from 300 to 900
- **Training_Dataset**: The 80% subset of data used to train models
- **Test_Dataset**: The 20% subset of data used to evaluate models
- **Preprocessor**: Component that transforms raw data into model-ready format
- **Scaler**: StandardScaler component that normalizes numerical features
- **Encoder**: Component that converts categorical values to numerical representations

## Requirements

### Requirement 1: Data Loading and Validation

**User Story:** As a data scientist, I want to load loan application data from CSV files, so that I can train and evaluate prediction models.

#### Acceptance Criteria

1. WHEN a CSV file path is provided, THE System SHALL load the data into a structured format
2. WHEN data is loaded, THE System SHALL validate that all 13 required features are present
3. WHEN data is loaded, THE System SHALL verify that no missing values exist in the dataset
4. WHEN data is loaded, THE System SHALL verify that no duplicate records exist
5. WHEN invalid data is encountered, THE System SHALL return a descriptive error message

### Requirement 2: Data Preprocessing

**User Story:** As a data scientist, I want to preprocess loan application data, so that it is suitable for machine learning model training.

#### Acceptance Criteria

1. WHEN categorical features are encountered, THE Encoder SHALL convert education values to numerical labels
2. WHEN categorical features are encountered, THE Encoder SHALL convert self_employed values to numerical labels
3. WHEN categorical features are encountered, THE Encoder SHALL convert loan_status values to numerical labels
4. WHEN numerical features are provided, THE Scaler SHALL apply standardization using StandardScaler
5. WHEN outliers are detected in residential_assets_value, THE Preprocessor SHALL handle them appropriately
6. WHEN outliers are detected in commercial_assets_value, THE Preprocessor SHALL handle them appropriately
7. WHEN outliers are detected in bank_asset_value, THE Preprocessor SHALL handle them appropriately
8. THE System SHALL preserve the loan_id feature without transformation

### Requirement 3: Dataset Splitting

**User Story:** As a data scientist, I want to split data into training and test sets, so that I can train models and evaluate their performance on unseen data.

#### Acceptance Criteria

1. WHEN splitting data, THE System SHALL allocate 80% of records to the Training_Dataset
2. WHEN splitting data, THE System SHALL allocate 20% of records to the Test_Dataset
3. WHEN splitting data, THE System SHALL maintain the distribution of Approved and Rejected classes
4. WHEN splitting data, THE System SHALL ensure no data leakage between training and test sets

### Requirement 4: Model Training

**User Story:** As a data scientist, I want to train machine learning models on loan application data, so that I can predict loan approval outcomes.

#### Acceptance Criteria

1. THE System SHALL support training a Logistic Regression classifier
2. THE System SHALL support training a Random Forest classifier
3. WHEN training a model, THE System SHALL use the Training_Dataset exclusively
4. WHEN training completes, THE System SHALL persist the trained Model for future predictions
5. WHEN training completes, THE System SHALL persist the Scaler for future data preprocessing

### Requirement 5: Model Prediction

**User Story:** As a loan officer, I want to predict loan approval status for new applications, so that I can make informed lending decisions.

#### Acceptance Criteria

1. WHEN a Loan_Application is provided, THE System SHALL preprocess it using the trained Scaler
2. WHEN a preprocessed Loan_Application is provided, THE Model SHALL generate a Prediction
3. WHEN generating predictions, THE System SHALL return either Approved or Rejected status
4. WHEN generating predictions on the Test_Dataset, THE System SHALL produce predictions for all records

### Requirement 6: Model Evaluation

**User Story:** As a data scientist, I want to evaluate model performance using multiple metrics, so that I can assess prediction quality and select the best model.

#### Acceptance Criteria

1. WHEN evaluating a Model, THE System SHALL calculate accuracy on the Test_Dataset
2. WHEN evaluating a Model, THE System SHALL calculate precision on the Test_Dataset
3. WHEN evaluating a Model, THE System SHALL calculate recall on the Test_Dataset
4. WHEN evaluating a Model, THE System SHALL calculate F1-score on the Test_Dataset
5. WHEN evaluating a Model, THE System SHALL calculate ROC-AUC score on the Test_Dataset
6. WHEN evaluating a Model, THE System SHALL generate a confusion matrix
7. WHEN evaluation completes, THE System SHALL return all metrics in a structured format

### Requirement 7: Feature Validation

**User Story:** As a system administrator, I want to validate input features, so that only valid loan applications are processed.

#### Acceptance Criteria

1. WHEN validating no_of_dependents, THE System SHALL accept values between 0 and 5 inclusive
2. WHEN validating education, THE System SHALL accept only Graduate or Not Graduate values
3. WHEN validating self_employed, THE System SHALL accept only Yes or No values
4. WHEN validating cibil_score, THE System SHALL accept values between 300 and 900 inclusive
5. WHEN validating loan_term, THE System SHALL accept values between 2 and 20 years inclusive
6. WHEN validating income_annum, THE System SHALL accept positive numerical values
7. WHEN validating loan_amount, THE System SHALL accept positive numerical values
8. WHEN validating asset values, THE System SHALL accept non-negative numerical values
9. WHEN invalid feature values are provided, THE System SHALL reject the Loan_Application with a descriptive error

### Requirement 8: Class Imbalance Handling

**User Story:** As a data scientist, I want the system to handle class imbalance, so that predictions are not biased toward the majority class.

#### Acceptance Criteria

1. WHEN training models, THE System SHALL account for the class distribution of 2,656 Approved vs 1,613 Rejected applications
2. WHERE class imbalance handling is enabled, THE System SHALL apply appropriate techniques during model training
3. WHEN evaluating models, THE System SHALL report metrics that reflect performance on both classes

### Requirement 9: Model Persistence and Loading

**User Story:** As a developer, I want to save and load trained models, so that I can reuse them without retraining.

#### Acceptance Criteria

1. WHEN a Model is trained, THE System SHALL serialize it to disk
2. WHEN a Scaler is fitted, THE System SHALL serialize it to disk
3. WHEN loading a saved Model, THE System SHALL restore it to its trained state
4. WHEN loading a saved Scaler, THE System SHALL restore it to its fitted state
5. WHEN serialization fails, THE System SHALL return a descriptive error message

### Requirement 10: Prediction Confidence

**User Story:** As a loan officer, I want to see prediction confidence scores, so that I can understand the model's certainty.

#### Acceptance Criteria

1. WHEN generating a Prediction, THE Model SHALL provide probability scores for both classes
2. WHEN providing probability scores, THE System SHALL ensure they sum to 1.0
3. WHEN providing probability scores, THE System SHALL return values between 0.0 and 1.0 inclusive
