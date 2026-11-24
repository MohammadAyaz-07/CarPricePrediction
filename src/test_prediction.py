
import pandas as pd

from pathlib import Path
from prediction import predict_car_price


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data" / "processed" / "used_cars_cleaned.csv"
)


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(DATA_PATH)


# ==========================================
# PREPARE A REAL VEHICLE
# ==========================================

sample = df.iloc[0].copy()

# Extract brand
brand = sample["name"].split()[0]

# Prepare vehicle details
vehicle_data = sample.drop(
    labels=["name", "selling_price"]
).to_dict()

vehicle_data["brand"] = brand


# ==========================================
# PREDICTION SYSTEM TEST
# ==========================================

predicted_price = predict_car_price(vehicle_data)

actual_price = sample["selling_price"]

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
# VALIDATION TEST HELPER
# ==========================================

def run_validation_test(test_name, invalid_vehicle, expected_error):

    try:
        predict_car_price(invalid_vehicle)

    except ValueError as error:

        if expected_error in str(error):
            print(f"\n{test_name} Passed: {error}")
        else:
            print(
                f"\n{test_name} FAILED: Unexpected error: {error}"
            )

    except Exception as error:

        print(
            f"\n{test_name} FAILED: Unexpected exception: {error}"
        )

    else:

        print(
            f"\n{test_name} FAILED: Invalid input was accepted!"
        )


# ==========================================
# NUMERICAL INPUT VALIDATION TESTING
# ==========================================

print("\n" + "=" * 45)
print("INPUT VALIDATION TESTING")
print("=" * 45)


# Test 1: Negative kilometres driven

invalid_vehicle = vehicle_data.copy()
invalid_vehicle["km_driven"] = -500

run_validation_test(
    "Test 1",
    invalid_vehicle,
    "km_driven cannot be negative"
)


# Test 2: Missing brand

invalid_vehicle = vehicle_data.copy()
del invalid_vehicle["brand"]

run_validation_test(
    "Test 2",
    invalid_vehicle,
    "Missing vehicle features"
)


# Test 3: Invalid numerical input

invalid_vehicle = vehicle_data.copy()
invalid_vehicle["engine"] = "invalid"

run_validation_test(
    "Test 3",
    invalid_vehicle,
    "engine must be a valid number"
)


# Test 4: Future vehicle year

invalid_vehicle = vehicle_data.copy()
invalid_vehicle["year"] = pd.Timestamp.now().year + 5

run_validation_test(
    "Test 4",
    invalid_vehicle,
    "Vehicle year cannot be in the future"
)


# ==========================================
# CATEGORICAL VALIDATION TESTING
# ==========================================

print("\n" + "=" * 45)
print("CATEGORICAL VALIDATION TESTING")
print("=" * 45)


# Test 1: Invalid fuel type

invalid_vehicle = vehicle_data.copy()
invalid_vehicle["fuel"] = "Nuclear"

run_validation_test(
    "Test 1",
    invalid_vehicle,
    "Invalid fuel"
)


# Test 2: Invalid transmission

invalid_vehicle = vehicle_data.copy()
invalid_vehicle["transmission"] = "Telepathic"

run_validation_test(
    "Test 2",
    invalid_vehicle,
    "Invalid transmission"
)


# Test 3: Invalid seller type

invalid_vehicle = vehicle_data.copy()
invalid_vehicle["seller_type"] = "Unknown Seller"

run_validation_test(
    "Test 3",
    invalid_vehicle,
    "Invalid seller_type"
)


# ==========================================
# TESTING COMPLETED
# ==========================================

print("\n" + "=" * 45)
print("ALL VALIDATION TESTS EXECUTED")
print("=" * 45)
