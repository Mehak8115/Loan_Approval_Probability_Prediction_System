# Design Document: Customer Loan Approval Prediction System

## Overview

The Customer Loan Approval Prediction System is a machine learning application that predicts loan approval outcomes based on customer financial and demographic data. The system implements a complete ML pipeline including data loading, preprocessing, model training, evaluation, and prediction capabilities.

The system supports two classification algorithms:
- Logistic Regression (baseline linear model)
- Random Forest (ensemble tree-based model)

The architecture follows a modular design with clear separation between data processing, model training, and prediction components.

## Architecture

The system consists of five main components:

1. **Data Loader**: Handles CSV file reading and initial validation
2. **Data Preprocessor**: Transforms raw data into model-ready format
3. **Model Trainer**: Trains and persists ML models
4. **Predictor**: Generates predictions for new loan applications
5. **Evaluator**: Computes performance metrics on test data

### Component Interaction Flow

```
CSV File → Data Loader → Raw DataFrame
                              ↓
                        Data Preprocessor → Processed Features + Labels
                              ↓
                        Dataset Splitter → Training Set + Test Set
                              ↓
                        Model Trainer → Trained Model + Fitted Scaler
                              ↓
                        Predictor → Predictions
                              ↓
                        Evaluator → Performance Metrics
```

## Components and Interfaces

### 1. Data Loader

**Responsibility**: Load loan application data from CSV files and perform initial validation.

**Interface**:
```python
class DataLoader:
    def load_data(file_path: str) -> DataFrame:
        """
        Load loan application data from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with 13 features and loan_status
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If required columns are missing
        """
        
    def validate_schema(df: DataFrame) -> bool:
        """
        Validate that all required columns are present.
        
        Args:
            df: Input DataFrame
            
        Returns:
            True if schema is valid
            
        Raises:
            ValueError: If columns are missing or invalid
        """
        
    def check_data_quality(df: DataFrame) -> dict:
        """
        Check for missing values and duplicates.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with quality metrics
        """
```

**Implementation Notes**:
- Use pandas.read_csv() for file loading
- Required columns: loan_id, no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term, cibil_score, residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value, loan_status
- Validate no missing values (df.isnull().sum() == 0)
- Validate no duplicates (df.duplicated().sum() == 0)

### 2. Data Preprocessor

**Responsibility**: Transform raw data into model-ready format through encoding, scaling, and outlier handling.

**Interface**:
```python
class DataPreprocessor:
    def __init__(self):
        self.label_encoders: dict = {}
        self.scaler: StandardScaler = None
        
    def fit_transform(df: DataFrame) -> tuple[ndarray, ndarray]:
        """
        Fit preprocessor and transform data.
        
        Args:
            df: Raw DataFrame with all features
            
        Returns:
            Tuple of (X: feature matrix, y: target labels)
        """
        
    def transform(df: DataFrame) -> ndarray:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            df: Raw DataFrame with features (no target)
            
        Returns:
            Transformed feature matrix
            
        Raises:
            ValueError: If preprocessor not fitted
        """
        
    def encode_categorical(df: DataFrame) -> DataFrame:
        """
        Encode categorical features to numerical labels.
        
        Args:
            df: DataFrame with categorical columns
            
        Returns:
            DataFrame with encoded columns
        """
        
    def handle_outliers(df: DataFrame, columns: list) -> DataFrame:
        """
        Handle outliers in specified columns using IQR method.
        
        Args:
            df: Input DataFrame
            columns: List of column names to process
            
        Returns:
            DataFrame with outliers handled
        """
        
    def scale_features(X: ndarray) -> ndarray:
        """
        Apply StandardScaler to numerical features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Scaled feature matrix
        """
```

**Implementation Notes**:
- Categorical features to encode: education, self_employed, loan_status
- Use sklearn.preprocessing.LabelEncoder for each categorical feature
- Outlier columns: residential_assets_value, commercial_assets_value, bank_asset_value
- Outlier handling: IQR method (cap at Q1 - 1.5*IQR and Q3 + 1.5*IQR)
- Exclude loan_id from preprocessing
- Use sklearn.preprocessing.StandardScaler for numerical features
- Store fitted encoders and scaler for transform() method

### 3. Feature Validator

**Responsibility**: Validate input feature values against business rules.

**Interface**:
```python
class FeatureValidator:
    def validate_loan_application(data: dict) -> tuple[bool, str]:
        """
        Validate all features in a loan application.
        
        Args:
            data: Dictionary with feature names as keys
            
        Returns:
            Tuple of (is_valid: bool, error_message: str)
        """
        
    def validate_no_of_dependents(value: int) -> bool:
        """Validate dependents in range [0, 5]"""
        
    def validate_education(value: str) -> bool:
        """Validate education in {'Graduate', 'Not Graduate'}"""
        
    def validate_self_employed(value: str) -> bool:
        """Validate self_employed in {'Yes', 'No'}"""
        
    def validate_cibil_score(value: int) -> bool:
        """Validate CIBIL score in range [300, 900]"""
        
    def validate_loan_term(value: int) -> bool:
        """Validate loan term in range [2, 20]"""
        
    def validate_positive_amount(value: float) -> bool:
        """Validate positive numerical values"""
        
    def validate_non_negative_amount(value: float) -> bool:
        """Validate non-negative numerical values"""
```

**Implementation Notes**:
- Validate before preprocessing
- Return descriptive error messages for each validation failure
- Check data types and value ranges

### 4. Model Trainer

**Responsibility**: Train machine learning models and handle class imbalance.

**Interface**:
```python
class ModelTrainer:
    def __init__(self, model_type: str):
        """
        Initialize trainer with model type.
        
        Args:
            model_type: 'logistic_regression' or 'random_forest'
        """
        
    def train(X_train: ndarray, y_train: ndarray) -> object:
        """
        Train model on training data.
        
        Args:
            X_train: Training feature matrix
            y_train: Training labels
            
        Returns:
            Trained model object
        """
        
    def save_model(model: object, file_path: str) -> None:
        """
        Serialize model to disk.
        
        Args:
            model: Trained model
            file_path: Path to save model
        """
        
    def load_model(file_path: str) -> object:
        """
        Load serialized model from disk.
        
        Args:
            file_path: Path to saved model
            
        Returns:
            Loaded model object
        """
```

**Implementation Notes**:
- Logistic Regression: Use sklearn.linear_model.LogisticRegression with default parameters
- Random Forest: Use sklearn.ensemble.RandomForestClassifier with default parameters
- Class imbalance: Consider using class_weight='balanced' parameter
- Model persistence: Use joblib or pickle for serialization
- Save both model and scaler together

### 5. Predictor

**Responsibility**: Generate predictions and probability scores for loan applications.

**Interface**:
```python
class Predictor:
    def __init__(self, model: object, preprocessor: DataPreprocessor):
        """
        Initialize predictor with trained model and preprocessor.
        
        Args:
            model: Trained ML model
            preprocessor: Fitted DataPreprocessor
        """
        
    def predict(X: ndarray) -> ndarray:
        """
        Generate binary predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions (0 or 1)
        """
        
    def predict_proba(X: ndarray) -> ndarray:
        """
        Generate probability scores for both classes.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of shape (n_samples, 2) with probabilities
        """
        
    def predict_single(application: dict) -> tuple[str, float]:
        """
        Predict for a single loan application.
        
        Args:
            application: Dictionary with feature values
            
        Returns:
            Tuple of (prediction: 'Approved'/'Rejected', confidence: float)
        """
```

**Implementation Notes**:
- Use model.predict() for binary predictions
- Use model.predict_proba() for probability scores
- Ensure probabilities sum to 1.0
- Convert numerical predictions back to 'Approved'/'Rejected' labels

### 6. Evaluator

**Responsibility**: Compute performance metrics for model evaluation.

**Interface**:
```python
class Evaluator:
    def evaluate(y_true: ndarray, y_pred: ndarray, y_proba: ndarray) -> dict:
        """
        Compute all evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            
        Returns:
            Dictionary with all metrics
        """
        
    def compute_accuracy(y_true: ndarray, y_pred: ndarray) -> float:
        """Compute accuracy score"""
        
    def compute_precision(y_true: ndarray, y_pred: ndarray) -> float:
        """Compute precision score"""
        
    def compute_recall(y_true: ndarray, y_pred: ndarray) -> float:
        """Compute recall score"""
        
    def compute_f1_score(y_true: ndarray, y_pred: ndarray) -> float:
        """Compute F1 score"""
        
    def compute_roc_auc(y_true: ndarray, y_proba: ndarray) -> float:
        """Compute ROC-AUC score"""
        
    def compute_confusion_matrix(y_true: ndarray, y_pred: ndarray) -> ndarray:
        """Compute confusion matrix"""
```

**Implementation Notes**:
- Use sklearn.metrics for all metric calculations
- ROC-AUC requires probability scores for positive class
- Return metrics in structured dictionary format
- Include per-class metrics for precision, recall, F1

## Data Models

### Loan Application Record

```python
@dataclass
class LoanApplication:
    loan_id: str
    no_of_dependents: int  # Range: [0, 5]
    education: str  # Values: 'Graduate', 'Not Graduate'
    self_employed: str  # Values: 'Yes', 'No'
    income_annum: float  # Positive value
    loan_amount: float  # Positive value
    loan_term: int  # Range: [2, 20] years
    cibil_score: int  # Range: [300, 900]
    residential_assets_value: float  # Non-negative
    commercial_assets_value: float  # Non-negative
    luxury_assets_value: float  # Non-negative
    bank_asset_value: float  # Non-negative
```

### Prediction Result

```python
@dataclass
class PredictionResult:
    loan_id: str
    prediction: str  # 'Approved' or 'Rejected'
    confidence: float  # Range: [0.0, 1.0]
    probability_approved: float
    probability_rejected: float
```

### Evaluation Metrics

```python
@dataclass
class EvaluationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: ndarray  # Shape: (2, 2)
    class_distribution: dict  # {'Approved': count, 'Rejected': count}
```

### Dataset Split

```python
@dataclass
class DatasetSplit:
    X_train: ndarray
    X_test: ndarray
    y_train: ndarray
    y_test: ndarray
    train_size: int
    test_size: int
    train_ratio: float  # 0.8
    test_ratio: float  # 0.2
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*


### Property 1: Schema Validation Completeness
*For any* DataFrame, the schema validator should accept it if and only if all 13 required feature columns are present.
**Validates: Requirements 1.2**

### Property 2: Data Quality Detection
*For any* DataFrame, the quality checker should correctly identify whether missing values or duplicate records exist.
**Validates: Requirements 1.3, 1.4**

### Property 3: Categorical Encoding Consistency
*For any* DataFrame with categorical features (education, self_employed, loan_status), encoding then decoding should produce the original categorical values.
**Validates: Requirements 2.1, 2.2, 2.3**

### Property 4: Feature Scaling Properties
*For any* set of numerical features after StandardScaler transformation, the mean should be approximately 0 and standard deviation should be approximately 1.
**Validates: Requirements 2.4**

### Property 5: Outlier Handling Bounds
*For any* DataFrame with outliers in residential_assets_value, commercial_assets_value, or bank_asset_value, the outlier handler should cap values at Q1 - 1.5*IQR (lower bound) and Q3 + 1.5*IQR (upper bound).
**Validates: Requirements 2.5, 2.6, 2.7**

### Property 6: Loan ID Preservation
*For any* loan application, the loan_id value should remain unchanged after preprocessing.
**Validates: Requirements 2.8**

### Property 7: Dataset Split Ratio
*For any* dataset split operation, the training set should contain 80% of records and the test set should contain 20% of records (within rounding tolerance).
**Validates: Requirements 3.1, 3.2**

### Property 8: Stratified Split Preservation
*For any* dataset split operation, the proportion of Approved vs Rejected classes in the training set should be approximately equal to the proportion in the test set.
**Validates: Requirements 3.3**

### Property 9: No Data Leakage
*For any* dataset split, the intersection of training set record IDs and test set record IDs should be empty.
**Validates: Requirements 3.4**

### Property 10: Model Serialization Round-Trip
*For any* trained model and fitted scaler, serializing to disk then loading back should produce equivalent predictions and transformations on the same input data.
**Validates: Requirements 4.4, 4.5, 9.3, 9.4**

### Property 11: Prediction Output Validity
*For any* loan application input, the prediction output should be either 'Approved' or 'Rejected'.
**Validates: Requirements 5.3**

### Property 12: Prediction Completeness
*For any* batch of loan applications, the number of predictions generated should equal the number of input records.
**Validates: Requirements 5.4**

### Property 13: Evaluation Metrics Completeness
*For any* model evaluation, the result should contain all required metrics (accuracy, precision, recall, F1-score, ROC-AUC) and each metric should be in the valid range [0, 1].
**Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.7**

### Property 14: Confusion Matrix Structure
*For any* model evaluation, the confusion matrix should have shape (2, 2) and the sum of all elements should equal the total number of predictions.
**Validates: Requirements 6.6**

### Property 15: Dependents Range Validation
*For any* value of no_of_dependents, the validator should accept it if and only if it is an integer in the range [0, 5].
**Validates: Requirements 7.1**

### Property 16: Education Category Validation
*For any* education value, the validator should accept it if and only if it is exactly 'Graduate' or 'Not Graduate'.
**Validates: Requirements 7.2**

### Property 17: Self-Employed Category Validation
*For any* self_employed value, the validator should accept it if and only if it is exactly 'Yes' or 'No'.
**Validates: Requirements 7.3**

### Property 18: CIBIL Score Range Validation
*For any* cibil_score value, the validator should accept it if and only if it is an integer in the range [300, 900].
**Validates: Requirements 7.4**

### Property 19: Loan Term Range Validation
*For any* loan_term value, the validator should accept it if and only if it is an integer in the range [2, 20].
**Validates: Requirements 7.5**

### Property 20: Positive Amount Validation
*For any* income_annum or loan_amount value, the validator should accept it if and only if it is a positive number (> 0).
**Validates: Requirements 7.6, 7.7**

### Property 21: Non-Negative Asset Validation
*For any* asset value (residential, commercial, luxury, or bank), the validator should accept it if and only if it is non-negative (>= 0).
**Validates: Requirements 7.8**

### Property 22: Validation Error Descriptiveness
*For any* invalid loan application, the validator should return a descriptive error message indicating which feature failed validation and why.
**Validates: Requirements 7.9**

### Property 23: Per-Class Metrics Reporting
*For any* model evaluation, the result should include precision, recall, and F1-score computed separately for both the Approved and Rejected classes.
**Validates: Requirements 8.3**

### Property 24: Probability Score Validity
*For any* prediction with probability scores, the probabilities for both classes should be in the range [0.0, 1.0] and sum to 1.0 (within floating point tolerance of 1e-6).
**Validates: Requirements 10.1, 10.2, 10.3**

## Error Handling

The system implements comprehensive error handling across all components:

### Data Loading Errors
- **FileNotFoundError**: Raised when CSV file path does not exist
- **ValueError**: Raised when required columns are missing from CSV
- **ValueError**: Raised when data contains missing values or duplicates

### Validation Errors
- **ValidationError**: Custom exception for feature validation failures
  - Contains field name and specific validation rule that failed
  - Provides descriptive error messages for user feedback

### Preprocessing Errors
- **ValueError**: Raised when attempting to transform data with unfitted preprocessor
- **KeyError**: Raised when required features are missing from input data

### Model Errors
- **ValueError**: Raised when invalid model_type is specified
- **FileNotFoundError**: Raised when attempting to load non-existent model file
- **PickleError**: Raised when model deserialization fails

### Prediction Errors
- **ValueError**: Raised when input data shape doesn't match expected features
- **RuntimeError**: Raised when model hasn't been trained before prediction

### Error Handling Principles
1. Fail fast with descriptive error messages
2. Validate inputs before processing
3. Provide context in error messages (which field, what value, why invalid)
4. Log errors for debugging and monitoring
5. Never silently ignore errors

## Testing Strategy

The system employs a dual testing approach combining unit tests and property-based tests for comprehensive coverage.

### Unit Testing

Unit tests focus on specific examples, edge cases, and integration points:

**Data Loading Tests**:
- Test loading valid CSV file with all features
- Test error handling for missing file
- Test error handling for missing columns
- Test detection of missing values
- Test detection of duplicate records

**Preprocessing Tests**:
- Test encoding of specific categorical values
- Test scaling produces expected output for known input
- Test outlier handling with specific outlier values
- Test loan_id preservation
- Test error handling for unfitted preprocessor

**Validation Tests**:
- Test boundary values for each feature (min, max, just outside range)
- Test invalid categorical values
- Test error message content and format

**Model Training Tests**:
- Test Logistic Regression training completes successfully
- Test Random Forest training completes successfully
- Test model can be saved and loaded
- Test scaler can be saved and loaded

**Prediction Tests**:
- Test prediction on single application
- Test batch prediction
- Test probability scores are returned

**Evaluation Tests**:
- Test each metric is computed correctly for known predictions
- Test confusion matrix structure
- Test per-class metrics are included

### Property-Based Testing

Property-based tests verify universal properties across many generated inputs. The system uses **pytest with Hypothesis** (for Python) as the property-based testing library.

**Configuration**:
- Minimum 100 iterations per property test
- Each test tagged with feature name and property number
- Tag format: `# Feature: customer-loan-approval, Property {N}: {property_text}`

**Test Generators**:
- Generate random valid loan applications
- Generate random DataFrames with various shapes and values
- Generate edge cases (boundary values, empty data, extreme outliers)
- Generate invalid inputs for error testing

**Property Test Coverage**:

Each of the 24 correctness properties listed above must be implemented as a separate property-based test:

1. **Property 1**: Generate DataFrames with various column combinations, test schema validation
2. **Property 2**: Generate DataFrames with/without missing values and duplicates, test detection
3. **Property 3**: Generate random categorical values, test encoding round-trip
4. **Property 4**: Generate random numerical data, test scaling produces mean≈0, std≈1
5. **Property 5**: Generate data with outliers, test capping at IQR bounds
6. **Property 6**: Generate random loan_ids, test preservation through preprocessing
7. **Property 7**: Generate datasets of various sizes, test 80/20 split ratio
8. **Property 8**: Generate datasets with various class distributions, test stratification
9. **Property 9**: Generate random datasets, test train/test sets are disjoint
10. **Property 10**: Generate random models and scalers, test save/load round-trip
11. **Property 11**: Generate random inputs, test predictions are 'Approved' or 'Rejected'
12. **Property 12**: Generate batches of various sizes, test output count matches input count
13. **Property 13**: Generate random predictions, test all metrics present and in [0,1]
14. **Property 14**: Generate random predictions, test confusion matrix shape and sum
15. **Property 15**: Generate integers in and out of range [0,5], test validation
16. **Property 16**: Generate various strings, test only valid education values accepted
17. **Property 17**: Generate various strings, test only valid self_employed values accepted
18. **Property 18**: Generate integers in and out of range [300,900], test validation
19. **Property 19**: Generate integers in and out of range [2,20], test validation
20. **Property 20**: Generate positive, zero, and negative values, test validation
21. **Property 21**: Generate non-negative and negative values, test validation
22. **Property 22**: Generate invalid applications, test error messages are descriptive
23. **Property 23**: Generate random predictions, test per-class metrics included
24. **Property 24**: Generate random predictions, test probabilities sum to 1.0 and in [0,1]

**Example Property Test Structure**:
```python
# Feature: customer-loan-approval, Property 4: Feature Scaling Properties
@given(st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=10, max_size=1000))
@settings(max_examples=100)
def test_feature_scaling_properties(numerical_features):
    """For any set of numerical features after StandardScaler transformation,
    the mean should be approximately 0 and standard deviation should be approximately 1."""
    scaler = StandardScaler()
    X = np.array(numerical_features).reshape(-1, 1)
    X_scaled = scaler.fit_transform(X)
    
    assert abs(np.mean(X_scaled)) < 0.01  # Mean approximately 0
    assert abs(np.std(X_scaled) - 1.0) < 0.01  # Std approximately 1
```

### Testing Balance

- **Unit tests**: Focus on specific examples and integration between components
- **Property tests**: Handle comprehensive input coverage through randomization
- Both approaches are complementary and necessary for high confidence in correctness
- Property tests catch edge cases that might not be considered in unit tests
- Unit tests provide concrete examples that document expected behavior

### Continuous Integration

- Run all tests (unit + property) on every commit
- Fail build if any test fails
- Track test coverage (aim for >90% code coverage)
- Monitor property test execution time (should complete in <5 minutes)
