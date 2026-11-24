import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor
from sklearn.base import clone

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import KFold
import joblib
# Load Dataset

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "used_cars_cleaned.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

# Extract and Normalize Brand

df["brand"] = df["name"].str.split().str[0]

brand_mapping = {
    "Land": "Land Rover",
    "Ashok": "Ashok Leyland"
}

df["brand"] = df["brand"].replace(brand_mapping)

# Separate Features and Target

X = df.drop(columns=["selling_price", "name"])
y = df["selling_price"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)


# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)

# Feature Groups

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

# Numerical Preprocessing

numerical_preprocessor = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# Categorical Preprocessing

categorical_preprocessor = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# Combined Preprocessor

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_preprocessor, numerical_features),
        ("categorical", categorical_preprocessor, categorical_features)
    ]
)

# Baseline Model

baseline_model = DummyRegressor(strategy="mean")

baseline_model.fit(X_train, y_train)

print("\nBaseline Model Trained Successfully!")

# Make Predictions

y_pred_baseline = baseline_model.predict(X_test)

print("\nFirst 10 Baseline Predictions:")
print(y_pred_baseline[:10])

# Model Evaluation

mae = mean_absolute_error(y_test, y_pred_baseline)

mse = mean_squared_error(y_test, y_pred_baseline)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred_baseline)

print("\n" + "=" * 50)
print("BASELINE MODEL EVALUATION")
print("=" * 50)

print(f"\nMAE:  ₹{mae:,.2f}")
print(f"MSE:  {mse:,.2f}")
print(f"RMSE: ₹{rmse:,.2f}")
print(f"R² Score: {r2:.4f}")

# Linear Regression Model Pipeline

linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

# Train the Model

linear_model.fit(X_train, y_train)

print("\nLinear Regression Model Trained Successfully!")

# Make Predictions

y_pred_linear = linear_model.predict(X_test)

print("\nFirst 10 Linear Regression Predictions:")
print(y_pred_linear[:10])

# Model Evaluation

mae_linear = mean_absolute_error(y_test, y_pred_linear)

mse_linear = mean_squared_error(y_test, y_pred_linear)

rmse_linear = np.sqrt(mse_linear)

r2_linear = r2_score(y_test, y_pred_linear)

print("\n" + "=" * 50)
print("LINEAR REGRESSION EVALUATION")
print("=" * 50)

print(f"\nMAE:  ₹{mae_linear:,.2f}")
print(f"MSE:  {mse_linear:,.2f}")
print(f"RMSE: ₹{rmse_linear:,.2f}")
print(f"R² Score: {r2_linear:.4f}")

# Decision Tree Regressor

decision_tree_model = Pipeline(
    steps=[
        ("preprocessor", clone(preprocessor)),
        ("model", DecisionTreeRegressor(random_state=42))
    ]
)

# Train the Model

decision_tree_model.fit(X_train, y_train)

print("\nDecision Tree Model Trained Successfully!")

# Make Predictions

y_pred_tree = decision_tree_model.predict(X_test)

print("\nFirst 10 Decision Tree Predictions:")
print(y_pred_tree[:10])

# Model Evaluation

mae_tree = mean_absolute_error(y_test, y_pred_tree)

mse_tree = mean_squared_error(y_test, y_pred_tree)

rmse_tree = np.sqrt(mse_tree)

r2_tree = r2_score(y_test, y_pred_tree)

print("\n" + "=" * 50)
print("DECISION TREE REGRESSOR EVALUATION")
print("=" * 50)

print(f"\nMAE:  ₹{mae_tree:,.2f}")
print(f"MSE:  {mse_tree:,.2f}")
print(f"RMSE: ₹{rmse_tree:,.2f}")
print(f"R² Score: {r2_tree:.4f}")

# Overfitting Analysis

y_train_pred_tree = decision_tree_model.predict(X_train)

train_r2_tree = r2_score(y_train, y_train_pred_tree)
test_r2_tree = r2_score(y_test, y_pred_tree)

print("\n" + "=" * 50)
print("DECISION TREE OVERFITTING ANALYSIS")
print("=" * 50)

print(f"\nTraining R² Score: {train_r2_tree:.4f}")
print(f"Testing R² Score:  {test_r2_tree:.4f}")

# Random Forest Regressor

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", clone(preprocessor)),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

# Train the Model

random_forest_model.fit(X_train, y_train)

print("\nRandom Forest Model Trained Successfully!")

# Make Predictions

y_pred_rf = random_forest_model.predict(X_test)

print("\nFirst 10 Random Forest Predictions:")
print(y_pred_rf[:10])

# Model Evaluation

mae_rf = mean_absolute_error(y_test, y_pred_rf)

mse_rf = mean_squared_error(y_test, y_pred_rf)

rmse_rf = np.sqrt(mse_rf)

r2_rf = r2_score(y_test, y_pred_rf)

print("\n" + "=" * 50)
print("RANDOM FOREST REGRESSOR EVALUATION")
print("=" * 50)

print(f"\nMAE:  ₹{mae_rf:,.2f}")
print(f"MSE:  {mse_rf:,.2f}")
print(f"RMSE: ₹{rmse_rf:,.2f}")
print(f"R² Score: {r2_rf:.4f}")

# Overfitting Analysis

y_train_pred_rf = random_forest_model.predict(X_train)

train_r2_rf = r2_score(y_train, y_train_pred_rf)
test_r2_rf = r2_score(y_test, y_pred_rf)

print("\n" + "=" * 50)
print("RANDOM FOREST OVERFITTING ANALYSIS")
print("=" * 50)

print(f"\nTraining R² Score: {train_r2_rf:.4f}")
print(f"Testing R² Score:  {test_r2_rf:.4f}")

# Gradient Boosting Regressor

gradient_boosting_model = Pipeline(
    steps=[
        ("preprocessor", clone(preprocessor)),
        ("model", GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ))
    ]
)

# Train the Model

gradient_boosting_model.fit(X_train, y_train)

print("\nGradient Boosting Model Trained Successfully!")

# Make Predictions

y_pred_gb = gradient_boosting_model.predict(X_test)

print("\nFirst 10 Gradient Boosting Predictions:")
print(y_pred_gb[:10])

# Model Evaluation

mae_gb = mean_absolute_error(y_test, y_pred_gb)

mse_gb = mean_squared_error(y_test, y_pred_gb)

rmse_gb = np.sqrt(mse_gb)

r2_gb = r2_score(y_test, y_pred_gb)

print("\n" + "=" * 50)
print("GRADIENT BOOSTING REGRESSOR EVALUATION")
print("=" * 50)

print(f"\nMAE:  ₹{mae_gb:,.2f}")
print(f"MSE:  {mse_gb:,.2f}")
print(f"RMSE: ₹{rmse_gb:,.2f}")
print(f"R² Score: {r2_gb:.4f}")

# Overfitting Analysis

y_train_pred_gb = gradient_boosting_model.predict(X_train)

train_r2_gb = r2_score(y_train, y_train_pred_gb)
test_r2_gb = r2_score(y_test, y_pred_gb)

print("\n" + "=" * 50)
print("GRADIENT BOOSTING OVERFITTING ANALYSIS")
print("=" * 50)

print(f"\nTraining R² Score: {train_r2_gb:.4f}")
print(f"Testing R² Score:  {test_r2_gb:.4f}")

# Model Comparison

model_results = pd.DataFrame({
    "Model": [
        "Baseline",
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting"
    ],

    "MAE": [
        mae,
        mae_linear,
        mae_tree,
        mae_rf,
        mae_gb
    ],

    "MSE": [
        mse,
        mse_linear,
        mse_tree,
        mse_rf,
        mse_gb
    ],

    "RMSE": [
        rmse,
        rmse_linear,
        rmse_tree,
        rmse_rf,
        rmse_gb
    ],

    "R2 Score": [
        r2,
        r2_linear,
        r2_tree,
        r2_rf,
        r2_gb
    ]
})

# Sort Models by RMSE
model_results = model_results.sort_values(
    by="RMSE",
    ascending=True
)

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(model_results.to_string(
    index=False,
    formatters={
        "MAE": "{:,.2f}".format,
        "MSE": "{:,.2f}".format,
        "RMSE": "{:,.2f}".format,
        "R2 Score": "{:.4f}".format
    }
))

# Best Model
best_model_name = model_results.iloc[0]["Model"]

print("\nBest Model Based on RMSE:", best_model_name)

# Cross-Validation Strategy

cv_strategy = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ==========================================
# RANDOM FOREST HYPERPARAMETER TUNING
# ==========================================

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", clone(preprocessor)),
        ("model", RandomForestRegressor(
            random_state=42,
            n_jobs=1
        ))
    ]
)

rf_param_grid = {
    "model__n_estimators": [100, 200, 300, 400],
    "model__max_depth": [None, 10, 15, 20, 25],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", 0.8, 1.0]
}

rf_search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=rf_param_grid,
    n_iter=12,
    cv=cv_strategy,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("\nStarting Random Forest Hyperparameter Tuning...")

rf_search.fit(X_train, y_train)

best_rf_model = rf_search.best_estimator_

print("\nRandom Forest Tuning Completed!")

print("\nBest Random Forest Parameters:")
print(rf_search.best_params_)

print("\nBest Cross-Validation RMSE:")
print(f"₹{-rf_search.best_score_:,.2f}")

# ==========================================
# GRADIENT BOOSTING HYPERPARAMETER TUNING
# ==========================================

gb_pipeline = Pipeline(
    steps=[
        ("preprocessor", clone(preprocessor)),
        ("model", GradientBoostingRegressor(
            random_state=42
        ))
    ]
)

gb_param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__max_depth": [2, 3, 4],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__subsample": [0.8, 1.0]
}

gb_search = RandomizedSearchCV(
    estimator=gb_pipeline,
    param_distributions=gb_param_grid,
    n_iter=12,
    cv=cv_strategy,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("\nStarting Gradient Boosting Hyperparameter Tuning...")

gb_search.fit(X_train, y_train)

best_gb_model = gb_search.best_estimator_

print("\nGradient Boosting Tuning Completed!")

print("\nBest Gradient Boosting Parameters:")
print(gb_search.best_params_)

print("\nBest Cross-Validation RMSE:")
print(f"₹{-gb_search.best_score_:,.2f}")

# ==========================================
# TUNED MODEL EVALUATION
# ==========================================

# Random Forest Predictions
y_pred_best_rf = best_rf_model.predict(X_test)

# Gradient Boosting Predictions
y_pred_best_gb = best_gb_model.predict(X_test)


# Random Forest Metrics
print("\n" + "=" * 50)
print("TUNED RANDOM FOREST EVALUATION")
print("=" * 50)

print(f"MAE:  ₹{mean_absolute_error(y_test, y_pred_best_rf):,.2f}")

print(f"RMSE: ₹{np.sqrt(mean_squared_error(y_test, y_pred_best_rf)):,.2f}")

print(f"R²:   {r2_score(y_test, y_pred_best_rf):.4f}")


# Gradient Boosting Metrics
print("\n" + "=" * 50)
print("TUNED GRADIENT BOOSTING EVALUATION")
print("=" * 50)

print(f"MAE:  ₹{mean_absolute_error(y_test, y_pred_best_gb):,.2f}")

print(f"RMSE: ₹{np.sqrt(mean_squared_error(y_test, y_pred_best_gb)):,.2f}")

print(f"R²:   {r2_score(y_test, y_pred_best_gb):.4f}")

# ==========================================
# SAVE FINAL MODEL
# ==========================================

# Create models directory
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Save the complete trained pipeline
MODEL_PATH = MODEL_DIR / "gradient_boosting_pipeline.joblib"

joblib.dump(best_gb_model, MODEL_PATH)

print("\n" + "=" * 50)
print("FINAL MODEL SAVED SUCCESSFULLY!")
print("=" * 50)

print(f"\nModel Location: {MODEL_PATH}")