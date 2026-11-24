import joblib
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model path
MODEL_PATH = BASE_DIR / "models" / "gradient_boosting_pipeline.joblib"

# Load saved model
loaded_model = joblib.load(MODEL_PATH)

print("Model Loaded Successfully!")

# Check model type
print("\nLoaded Model:")
print(loaded_model)

# Test prediction using existing test data