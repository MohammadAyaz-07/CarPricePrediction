from scipy.sparse import issparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
from pathlib import Path



# Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "used_cars_cleaned.csv"


# Load cleaned dataset
df = pd.read_csv(DATA_PATH)


# Display dataset information
print("Dataset Loaded Successfully!")
print("=" * 50)

print(f"Dataset Shape: {df.shape}")

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

# Feature selection analysis

print("\n" + "=" * 50)
print("FEATURE SELECTION ANALYSIS")
print("=" * 50)

print("\nUnique car names:")
print(df["name"].nunique())

print("\nMost common car names:")
print(df["name"].value_counts().head(10))

print("\nUnique values in categorical columns:")

categorical_columns = [
    "name",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]

for column in categorical_columns:
    print(f"{column}: {df[column].nunique()} unique values")

# Feature Engineering: Extract Car Brand

df["brand"] = df["name"].str.split().str[0]


print("\n" + "=" * 50)
print("BRAND EXTRACTION")
print("=" * 50)

print("\nFirst 10 extracted brands:")
print(df[["name", "brand"]].head(10))

print("\nNumber of unique brands:")
print(df["brand"].nunique())

print("\nBrand distribution:")
print(df["brand"].value_counts())

# Feature Engineering: Brand Normalization

brand_mapping = {
    "Land": "Land Rover",
    "Ashok": "Ashok Leyland"
}

df["brand"] = df["brand"].replace(brand_mapping)


# Verify brand normalization

print("\nBrand Normalization:")
print(df["brand"].value_counts())

# Define input features

numerical_features = [
    "year",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "torque",
    "seats"
]

categorical_features = [
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]


# Create feature matrix X

X = df[numerical_features + categorical_features]


# Create target variable y

y = df["selling_price"]


# Display results

print("\n" + "=" * 50)
print("FEATURES AND TARGET SEPARATION")
print("=" * 50)

print(f"\nX Shape: {X.shape}")
print(f"y Shape: {y.shape}")

print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)

print("\nTarget Variable:")
print(y.name)

print("\nFeature Matrix - First 5 Rows:")
print(X.head())

print("\nTarget Variable - First 5 Values:")
print(y.head())

# Train-Test Split
from sklearn.model_selection import train_test_split


# Split dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Display split information

print("\n" + "=" * 50)
print("TRAIN-TEST SPLIT")
print("=" * 50)

print(f"Training Features: {X_train.shape}")
print(f"Testing Features: {X_test.shape}")

print(f"\nTraining Target: {y_train.shape}")
print(f"Testing Target: {y_test.shape}")

print(f"\nTraining Data Percentage: {len(X_train) / len(X) * 100:.2f}%")
print(f"Testing Data Percentage: {len(X_test) / len(X) * 100:.2f}%")

# Numerical Preprocessing Pipeline

numerical_preprocessor = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# Fit on training data and transform

X_train_numerical = numerical_preprocessor.fit_transform(
    X_train[numerical_features]
)


# Transform testing data

X_test_numerical = numerical_preprocessor.transform(
    X_test[numerical_features]
)


# Validation

print("\n" + "=" * 50)
print("NUMERICAL PREPROCESSING")
print("=" * 50)

print("\nTraining Numerical Shape:")
print(X_train_numerical.shape)

print("\nTesting Numerical Shape:")
print(X_test_numerical.shape)

print("\nMissing Values After Preprocessing:")
print(pd.DataFrame(X_train_numerical).isnull().sum().sum())

print("\nNumerical Preprocessing Completed Successfully!")

# Categorical Preprocessing Pipeline

categorical_preprocessor = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# Fit on training data and transform
X_train_categorical = categorical_preprocessor.fit_transform(
    X_train[categorical_features]
)

# Transform testing data
X_test_categorical = categorical_preprocessor.transform(
    X_test[categorical_features]
)

# Validation
print("\n" + "=" * 50)
print("CATEGORICAL PREPROCESSING")
print("=" * 50)

print("\nTraining Categorical Shape:")
print(X_train_categorical.shape)

print("\nTesting Categorical Shape:")
print(X_test_categorical.shape)

print("\nCategorical Preprocessing Completed Successfully!")

# Combined Preprocessing Pipeline

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_preprocessor, numerical_features),
        ("categorical", categorical_preprocessor, categorical_features)
    ]
)

# Fit on training data and transform
X_train_processed = preprocessor.fit_transform(X_train)

# Transform testing data
X_test_processed = preprocessor.transform(X_test)

# Validation

print("\n" + "=" * 50)
print("COMBINED PREPROCESSING")
print("=" * 50)

print("\nProcessed Training Data Shape:")
print(X_train_processed.shape)

print("\nProcessed Testing Data Shape:")
print(X_test_processed.shape)

# Check missing values in training data
if issparse(X_train_processed):
    train_missing = pd.isnull(X_train_processed.data).sum()
else:
    train_missing = pd.isnull(X_train_processed).sum()

# Check missing values in testing data
if issparse(X_test_processed):
    test_missing = pd.isnull(X_test_processed.data).sum()
else:
    test_missing = pd.isnull(X_test_processed).sum()

print("\nTraining Data Missing Values:")
print(train_missing)

print("\nTesting Data Missing Values:")
print(test_missing)

print("\nCombined Preprocessing Completed Successfully!")

# Final Feature Engineering Validation

import numpy as np

print("\n" + "=" * 50)
print("FINAL FEATURE ENGINEERING VALIDATION")
print("=" * 50)

# 1. Dataset dimensions
assert X_train_processed.shape[0] == y_train.shape[0]
assert X_test_processed.shape[0] == y_test.shape[0]

print("\n1. Dataset dimensions: PASSED")

# 2. Feature alignment
assert X_train_processed.shape[1] == X_test_processed.shape[1]

print("2. Feature alignment: PASSED")

# 3. Missing values
assert train_missing == 0
assert test_missing == 0

print("3. Missing values: PASSED")

# 4. Numerical validity
if issparse(X_train_processed):
    train_values = X_train_processed.data
    test_values = X_test_processed.data
else:
    train_values = X_train_processed
    test_values = X_test_processed

assert np.isfinite(train_values).all()
assert np.isfinite(test_values).all()

print("4. Numerical validity: PASSED")

# 5. Target validation
assert y_train.isnull().sum() == 0
assert y_test.isnull().sum() == 0

print("5. Target validation: PASSED")

# 6. Train-test separation
assert len(X_train.index.intersection(X_test.index)) == 0

print("6. Train-test separation: PASSED")

# 7. Feature count validation
expected_features = (
    len(numerical_features)
    + X_train_categorical.shape[1]
)

assert X_train_processed.shape[1] == expected_features

print("7. Feature count validation: PASSED")

# Final result
print("\n" + "=" * 50)
print("ALL VALIDATIONS PASSED!")
print("FEATURE ENGINEERING PHASE COMPLETED!")
print("=" * 50)