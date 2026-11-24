import joblib
import pandas as pd
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model path
MODEL_PATH = BASE_DIR / "models" / "gradient_boosting_pipeline.joblib"

# Load saved model
loaded_model = joblib.load(MODEL_PATH)

print("Model Loaded Successfully!")
print("\nLoaded Model:")
print(loaded_model)

# Load cleaned dataset
DATA_PATH = BASE_DIR / "data" / "processed" / "used_cars_cleaned.csv"

df = pd.read_csv(DATA_PATH)

# Select one real vehicle
sample = df.iloc[[0]].copy()

# Extract brand
sample["brand"] = sample["name"].str.split().str[0]

# Normalize brand
brand_mapping = {
    "Land": "Land Rover",
    "Ashok": "Ashok Leyland"
}

sample["brand"] = sample["brand"].replace(brand_mapping)

# Prepare input features
sample_input = sample.drop(
    columns=["selling_price", "name"]
)

# Actual selling price
actual_price = sample["selling_price"].iloc[0]

# Predict selling price
predicted_price = loaded_model.predict(sample_input)[0]

# Display results
print("\n" + "=" * 45)
print("CAR PRICE PREDICTION TEST")
print("=" * 45)

print("\nActual Selling Price: ₹{:,.2f}".format(actual_price))

print("Predicted Selling Price: ₹{:,.2f}".format(predicted_price))

print(
    "Absolute Error: ₹{:,.2f}".format(
        abs(actual_price - predicted_price)
    )
)