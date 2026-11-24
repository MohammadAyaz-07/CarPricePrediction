
import joblib
import pandas as pd
import numpy as np

from pathlib import Path


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR / "models" / "gradient_boosting_pipeline.joblib"
)


# ==========================================
# FEATURE CONFIGURATION
# ==========================================

FEATURE_COLUMNS = [
    "year",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "torque",
    "seats",
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]


NUMERICAL_FEATURES = [
    "year",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "torque",
    "seats"
]


CATEGORICAL_FEATURES = [
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]


# ==========================================
# BRAND NORMALIZATION
# ==========================================

BRAND_MAPPING = {
    "Land": "Land Rover",
    "Ashok": "Ashok Leyland"
}


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# ==========================================
# EXTRACT VALID CATEGORIES
# ==========================================

categorical_encoder = (
    model.named_steps["preprocessor"]
    .named_transformers_["categorical"]
    .named_steps["encoder"]
)


VALID_CATEGORIES = {
    feature: set(categories)
    for feature, categories in zip(
        CATEGORICAL_FEATURES,
        categorical_encoder.categories_
    )
}


# ==========================================
# INPUT VALIDATION
# ==========================================

def validate_vehicle_data(vehicle_data):

    # Check required features
    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in vehicle_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing vehicle features: {missing_features}"
        )

    # Validate categorical features
    for feature in CATEGORICAL_FEATURES:

        value = vehicle_data[feature]

        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{feature} must be a non-empty string."
            )

        value = value.strip()

        if value not in VALID_CATEGORIES[feature]:
            raise ValueError(
                f"Invalid {feature}: '{value}'. "
                f"Valid options: "
                f"{sorted(VALID_CATEGORIES[feature])}"
            )

        vehicle_data[feature] = value

    # Validate numerical features
    numeric_values = {}

    for feature in NUMERICAL_FEATURES:

        value = vehicle_data[feature]

        if isinstance(value, (bool, np.bool_)):
            raise ValueError(
                f"{feature} must be a valid number."
            )

        try:
            value = float(value)

        except (TypeError, ValueError):
            raise ValueError(
                f"{feature} must be a valid number."
            )

        if not np.isfinite(value):
            raise ValueError(
                f"{feature} must be finite."
            )

        if value < 0:
            raise ValueError(
                f"{feature} cannot be negative."
            )

        numeric_values[feature] = value

    # Validate vehicle year
    if not numeric_values["year"].is_integer():
        raise ValueError(
            "Year must be a whole number."
        )

    if numeric_values["year"] == 0:
        raise ValueError(
            "Vehicle year must be greater than zero."
        )

    if numeric_values["year"] > pd.Timestamp.now().year:
        raise ValueError(
            "Vehicle year cannot be in the future."
        )

    # Validate seating capacity
    if not numeric_values["seats"].is_integer():
        raise ValueError(
            "Seats must be a whole number."
        )

    if numeric_values["seats"] == 0:
        raise ValueError(
            "Seats must be greater than zero."
        )

    # Store validated numerical values
    vehicle_data.update(numeric_values)

    return vehicle_data


# ==========================================
# CAR PRICE PREDICTION FUNCTION
# ==========================================

def predict_car_price(vehicle_data):

    # Check input type
    if not isinstance(vehicle_data, dict):
        raise TypeError(
            "Vehicle data must be provided as a dictionary."
        )

    # Copy input data
    vehicle_data = vehicle_data.copy()

    # Check required features before brand access
    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in vehicle_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing vehicle features: {missing_features}"
        )

    # Normalize brand
    vehicle_data["brand"] = BRAND_MAPPING.get(
        vehicle_data["brand"],
        vehicle_data["brand"]
    )

    # Validate vehicle information
    vehicle_data = validate_vehicle_data(vehicle_data)

    # Prepare input DataFrame
    input_df = pd.DataFrame(
        [vehicle_data],
        columns=FEATURE_COLUMNS
    )

    # Generate prediction
    prediction = model.predict(input_df)[0]

    return round(float(prediction), 2)
