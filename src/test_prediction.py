
import pandas as pd
from pathlib import Path

from prediction import predict_car_price


# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load cleaned dataset
DATA_PATH = BASE_DIR / "data" / "processed" / "used_cars_cleaned.csv"

df = pd.read_csv(DATA_PATH)


# Select one real vehicle
sample = df.iloc[0].copy()

# Extract brand
brand = sample["name"].split()[0]

# Prepare vehicle details
vehicle_data = sample.drop(
    labels=["name", "selling_price"]
).to_dict()

vehicle_data["brand"] = brand


# Generate prediction
predicted_price = predict_car_price(vehicle_data)


# Actual price
actual_price = sample["selling_price"]


# Display results
print("\n" + "=" * 45)
print("PREDICTION SYSTEM TEST")
print("=" * 45)

print(f"\nVehicle Brand: {brand}")

print(f"Actual Selling Price: ₹{actual_price:,.2f}")

print(f"Predicted Selling Price: ₹{predicted_price:,.2f}")

print(
    f"Absolute Error: ₹{abs(actual_price - predicted_price):,.2f}"
)


# ==========================================
# INPUT VALIDATION TESTING
# ==========================================

print("\n" + "=" * 45)
print("INPUT VALIDATION TESTING")
print("=" * 45)


# Test 1: Negative kilometres driven
invalid_vehicle = vehicle_data.copy()
invalid_vehicle["km_driven"] = -500

try:
    predict_car_price(invalid_vehicle)
except ValueError as error:
    print("\nTest 1 Passed:", error)


# Test 2: Missing brand
invalid_vehicle = vehicle_data.copy()
del invalid_vehicle["brand"]

try:
    predict_car_price(invalid_vehicle)
except ValueError as error:
    print("\nTest 2 Passed:", error)


# Test 3: Invalid numerical input
invalid_vehicle = vehicle_data.copy()
invalid_vehicle["engine"] = "invalid"

try:
    predict_car_price(invalid_vehicle)
except ValueError as error:
    print("\nTest 3 Passed:", error)


# Test 4: Future vehicle year
invalid_vehicle = vehicle_data.copy()
invalid_vehicle["year"] = pd.Timestamp.now().year + 5

try:
    predict_car_price(invalid_vehicle)
except ValueError as error:
    print("\nTest 4 Passed:", error)

# ==========================================
# CATEGORICAL VALIDATION TESTING
# ==========================================

print("\n" + "=" * 45)
print("CATEGORICAL VALIDATION TESTING")
print("=" * 45)


# Test 1: Invalid fuel type
invalid_vehicle = vehicle_data.copy()
invalid_vehicle["fuel"] = "Nuclear"

try:
    predict_car_price(invalid_vehicle)
except ValueError as error:
    print("\nTest 1 Passed:", error)


# Test 2: Invalid transmission
invalid_vehicle = vehicle_data.copy()
invalid_vehicle["transmission"] = "Telepathic"

try:
    predict_car_price(invalid_vehicle)
except ValueError as error:
    print("\nTest 2 Passed:", error)


# Test 3: Invalid seller type
invalid_vehicle = vehicle_data.copy()
invalid_vehicle["seller_type"] = "Unknown Seller"

try:
    predict_car_price(invalid_vehicle)
except ValueError as error:
    print("\nTest 3 Passed:", error)