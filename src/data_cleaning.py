from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "used_cars.csv"

PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "used_cars_cleaned.csv"
)


# ============================================================
# CONSTANTS
# ============================================================

EXPECTED_COLUMNS = [
    "name",
    "year",
    "selling_price",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "mileage",
    "engine",
    "max_power",
    "torque",
    "seats",
]

SPECIFICATION_COLUMNS = [
    "mileage",
    "engine",
    "max_power",
    "torque",
    "seats",
]

NUMERIC_COLUMNS = [
    "year",
    "selling_price",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "torque",
    "seats",
]


# ============================================================
# 1. LOAD DATASET
# ============================================================

def load_dataset():
    """
    Load the raw used-car dataset.
    """

    print("\n" + "=" * 60)
    print("LOADING RAW DATASET")
    print("=" * 60)

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Dataset shape: {df.shape}")
    print(f"Total rows: {df.shape[0]}")
    print(f"Total columns: {df.shape[1]}")

    return df


# ============================================================
# 2. DUPLICATE ANALYSIS
# ============================================================

def analyze_duplicates(df):
    """
    Analyze exact duplicate records.
    """

    print("\n" + "=" * 60)
    print("DUPLICATE ANALYSIS")
    print("=" * 60)

    duplicate_count = df.duplicated().sum()

    duplicate_rows = df[
        df.duplicated(keep=False)
    ]

    duplicate_percentage = (
        duplicate_count / len(df)
    ) * 100

    print(f"Exact duplicate records: {duplicate_count}")

    print(
        "Rows belonging to duplicate groups: "
        f"{len(duplicate_rows)}"
    )

    print(
        f"Duplicate percentage: "
        f"{duplicate_percentage:.2f}%"
    )


def investigate_duplicates(df):
    """
    Investigate duplicate records and repeated vehicle names.

    Repeated names are not automatically duplicates because
    vehicles with the same name may have different attributes.
    """

    print("\n" + "=" * 60)
    print("DUPLICATE INVESTIGATION")
    print("=" * 60)

    duplicate_rows = df[
        df.duplicated(keep=False)
    ]

    print(
        "Rows involved in duplicate groups: "
        f"{len(duplicate_rows)}"
    )

    duplicate_groups = (
        duplicate_rows
        .groupby(EXPECTED_COLUMNS, dropna=False)
        .size()
        .reset_index(name="count")
    )

    print(
        "Number of duplicate groups: "
        f"{len(duplicate_groups)}"
    )

    print("\nMost frequently duplicated records:")

    if not duplicate_groups.empty:
        print(
            duplicate_groups
            .sort_values("count", ascending=False)
            .head(10)
            .to_string(index=False)
        )

    print("\nRepeated vehicle names:")

    name_counts = df["name"].value_counts()

    repeated_names = name_counts[
        name_counts > 1
    ]

    print(
        "Vehicle names appearing more than once: "
        f"{len(repeated_names)}"
    )

    print(repeated_names.head(10))


def remove_duplicates(df):
    """
    Remove exact duplicate records only.
    """

    print("\n" + "=" * 60)
    print("REMOVING EXACT DUPLICATES")
    print("=" * 60)

    original_rows = len(df)

    df = df.drop_duplicates().copy()

    removed_rows = original_rows - len(df)

    print(f"Original rows: {original_rows}")
    print(f"Rows removed: {removed_rows}")
    print(f"Rows remaining: {len(df)}")

    return df


# ============================================================
# 3. MISSING VALUE ANALYSIS
# ============================================================

def analyze_missing_values(df):
    """
    Analyze missing values in the dataset.
    """

    print("\n" + "=" * 60)
    print("MISSING VALUE ANALYSIS")
    print("=" * 60)

    missing_count = df.isnull().sum()

    missing_percentage = (
        missing_count / len(df)
    ) * 100

    missing_summary = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing Percentage": missing_percentage,
    })

    missing_summary = missing_summary[
        missing_summary["Missing Count"] > 0
    ]

    if missing_summary.empty:
        print("No missing values found.")

    else:
        print(missing_summary)

    rows_with_missing = df.isnull().any(axis=1).sum()

    print(
        "\nRows containing missing values: "
        f"{rows_with_missing}"
    )


def investigate_missing_values(df):
    """
    Investigate combinations of missing vehicle specifications.
    """

    print("\n" + "=" * 60)
    print("MISSING VALUE INVESTIGATION")
    print("=" * 60)

    missing_combinations = (
        df[SPECIFICATION_COLUMNS]
        .isnull()
        .value_counts()
        .reset_index(name="count")
    )

    print("\nMissing specification combinations:")

    print(missing_combinations)

    all_specs_missing = (
        df[SPECIFICATION_COLUMNS]
        .isnull()
        .all(axis=1)
    )

    completely_missing_rows = df[
        all_specs_missing
    ]

    print("\nRows missing all specification fields:")

    print(
        f"Count: {len(completely_missing_rows)}"
    )

    if not completely_missing_rows.empty:

        print("\nFuel distribution:")

        print(
            completely_missing_rows["fuel"]
            .value_counts()
        )

        print("\nYear distribution:")

        print(
            completely_missing_rows["year"]
            .describe()
        )

        print("\nSelling price distribution:")

        print(
            completely_missing_rows["selling_price"]
            .describe()
        )


# ============================================================
# 4. NUMERIC EXTRACTION
# ============================================================

def extract_numeric_value(value):
    """
    Extract the first numerical value from a string.

    Examples:
        '22.37 kmpl' -> 22.37
        '2487 CC' -> 2487
        '214.56 bhp' -> 214.56
    """

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value == "":
        return np.nan

    match = re.search(
        r"-?\d+(?:,\d{3})*(?:\.\d+)?",
        value
    )

    if match is None:
        return np.nan

    numeric_string = match.group(0).replace(",", "")

    try:
        return float(numeric_string)

    except ValueError:
        return np.nan


# ============================================================
# 5. TORQUE CLEANING
# ============================================================

def clean_torque(value):
    """
    Convert torque values into Nm.

    Handles:
        190Nm@2000rpm
        22.4kgm@1750rpm
        12.7@2700(kgm@rpm)
        145@4100(kgm@rpm)
        380Nm(38.7kgm)@2500rpm

    Rules:
        1. Explicit Nm values take priority.
        2. Explicit kgm values are converted to Nm.
        3. Malformed kgm@rpm values:
           - <= 40: assumed kgm and converted.
           - > 40: assumed to already be Nm.
        4. Other malformed torque values are extracted
           as the first numeric value.
    """

    if pd.isna(value):
        return np.nan

    torque_string = str(value).strip().lower()

    if torque_string == "":
        return np.nan

    # --------------------------------------------------------
    # RULE 1: EXPLICIT NM
    # --------------------------------------------------------

    nm_match = re.search(
        r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*nm\b",
        torque_string,
        flags=re.IGNORECASE
    )

    if nm_match:

        torque_value = float(
            nm_match.group(1).replace(",", "")
        )

        return torque_value

    # --------------------------------------------------------
    # RULE 2: EXPLICIT KGM
    # --------------------------------------------------------

    kgm_match = re.search(
        r"(\d+(?:,\d{3})*(?:\.\d+)?)"
        r"\s*kg\s*[-.]?\s*m\b",
        torque_string,
        flags=re.IGNORECASE
    )

    if kgm_match:

        torque_value = float(
            kgm_match.group(1).replace(",", "")
        )

        return torque_value * 9.80665

    # --------------------------------------------------------
    # RULE 3: MALFORMED KGM@RPM FORMAT
    # --------------------------------------------------------

    malformed_kgm_match = re.search(
        r"(\d+(?:,\d{3})*(?:\.\d+)?)"
        r"\s*@.*?"
        r"\(\s*kg\s*[-.]?\s*m\s*@",
        torque_string,
        flags=re.IGNORECASE
    )

    if malformed_kgm_match:

        torque_value = float(
            malformed_kgm_match.group(1).replace(",", "")
        )

        if torque_value <= 40:

            return torque_value * 9.80665

        else:

            return torque_value

    # --------------------------------------------------------
    # RULE 4: GENERIC NUMERIC EXTRACTION
    # --------------------------------------------------------

    torque_value = extract_numeric_value(
        torque_string
    )

    return torque_value


# ============================================================
# 6. CLEAN VEHICLE SPECIFICATIONS
# ============================================================

def clean_specifications(df):
    """
    Clean and convert vehicle specification columns.
    """

    print("\n" + "=" * 60)
    print("CLEANING VEHICLE SPECIFICATIONS")
    print("=" * 60)

    df = df.copy()

    # --------------------------------------------------------
    # MILEAGE
    # --------------------------------------------------------

    print("Cleaning mileage...")

    df["mileage"] = df["mileage"].apply(
        extract_numeric_value
    )

    # --------------------------------------------------------
    # ENGINE
    # --------------------------------------------------------

    print("Cleaning engine...")

    df["engine"] = df["engine"].apply(
        extract_numeric_value
    )

    # --------------------------------------------------------
    # MAX POWER
    # --------------------------------------------------------

    print("Cleaning max_power...")

    df["max_power"] = df["max_power"].apply(
        extract_numeric_value
    )

    # --------------------------------------------------------
    # TORQUE
    # --------------------------------------------------------

    print("Cleaning torque...")

    df["torque"] = df["torque"].apply(
        clean_torque
    )

    # --------------------------------------------------------
    # SEATS
    # --------------------------------------------------------

    print("Cleaning seats...")

    df["seats"] = df["seats"].apply(
        extract_numeric_value
    )

    # --------------------------------------------------------
    # INVALID ZERO VALUES
    # --------------------------------------------------------

    print("\nHandling invalid zero values...")

    df.loc[
        df["mileage"] == 0,
        "mileage"
    ] = np.nan

    df.loc[
        df["max_power"] == 0,
        "max_power"
    ] = np.nan

    # --------------------------------------------------------
    # DATA TYPES
    # --------------------------------------------------------

    # Engine capacity is measured in whole CC values.
    df["engine"] = df["engine"].round().astype("Int64")

    # Seat count is an integer but can contain missing values.
    df["seats"] = df["seats"].round().astype("Int64")

    # Remaining specification values are continuous.
    df["mileage"] = df["mileage"].astype("Float64")

    df["max_power"] = df["max_power"].astype("Float64")

    df["torque"] = df["torque"].astype("Float64")

    print("Specification cleaning completed.")

    return df


# ============================================================
# 7. SUSPICIOUS VALUE INVESTIGATION
# ============================================================

def investigate_suspicious_values(df):
    """
    Investigate potentially suspicious numerical values.

    Values are investigated rather than automatically deleted.
    """

    print("\n" + "=" * 60)
    print("SUSPICIOUS VALUE INVESTIGATION")
    print("=" * 60)

    # --------------------------------------------------------
    # ZERO MILEAGE
    # --------------------------------------------------------

    zero_mileage = df[
        df["mileage"] == 0
    ]

    print("\nVehicles with zero mileage:")

    print(f"Count: {len(zero_mileage)}")

    # --------------------------------------------------------
    # ZERO MAX POWER
    # --------------------------------------------------------

    zero_power = df[
        df["max_power"] == 0
    ]

    print("\nVehicles with zero max_power:")

    print(f"Count: {len(zero_power)}")

    # --------------------------------------------------------
    # EXTREME TORQUE
    # --------------------------------------------------------

    extreme_torque = df[
        df["torque"] > 1000
    ]

    print(
        "\nVehicles with torque greater than 1000 Nm:"
    )

    print(f"Count: {len(extreme_torque)}")

    if not extreme_torque.empty:

        print(
            extreme_torque[
                [
                    "name",
                    "year",
                    "torque",
                    "selling_price"
                ]
            ]
        )

    # --------------------------------------------------------
    # HIGH SEAT COUNT
    # --------------------------------------------------------

    high_seat_count = df[
        df["seats"] > 10
    ]

    print("\nVehicles with more than 10 seats:")

    print(f"Count: {len(high_seat_count)}")

    if not high_seat_count.empty:

        print(
            high_seat_count[
                [
                    "name",
                    "year",
                    "seats",
                    "selling_price"
                ]
            ]
        )

    # --------------------------------------------------------
    # HIGH ENGINE CAPACITY
    # --------------------------------------------------------

    high_engine = df[
        df["engine"] > 3000
    ]

    print(
        "\nVehicles with engine capacity "
        "greater than 3000 CC:"
    )

    print(f"Count: {len(high_engine)}")

    if not high_engine.empty:

        print(
            high_engine[
                [
                    "name",
                    "year",
                    "engine",
                    "selling_price"
                ]
            ]
        )

    # --------------------------------------------------------
    # HIGH MAX POWER
    # --------------------------------------------------------

    high_power = df[
        df["max_power"] > 300
    ]

    print(
        "\nVehicles with max_power "
        "greater than 300 BHP:"
    )

    print(f"Count: {len(high_power)}")

    if not high_power.empty:

        print(
            high_power[
                [
                    "name",
                    "year",
                    "max_power",
                    "selling_price"
                ]
            ]
        )


# ============================================================
# 8. FINAL QUALITY AUDIT
# ============================================================

def final_quality_audit(df):
    """
    Investigate unusual values that may be legitimate
    but require manual inspection.

    This function does not automatically remove records.
    """

    print("\n" + "=" * 60)
    print("FINAL QUALITY AUDIT")
    print("=" * 60)

    # --------------------------------------------------------
    # EXTREMELY HIGH DISTANCE DRIVEN
    # --------------------------------------------------------

    print(
        "\nVehicles with km_driven above 1,000,000:"
    )

    high_km = df[
        df["km_driven"] > 1_000_000
    ]

    if high_km.empty:

        print("None found.")

    else:

        print(
            high_km[
                [
                    "name",
                    "year",
                    "km_driven",
                    "selling_price"
                ]
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # LOW TORQUE
    # --------------------------------------------------------

    print("\nVehicles with torque below 60 Nm:")

    low_torque = df[
        df["torque"] < 60
    ]

    if low_torque.empty:

        print("None found.")

    else:

        print(
            low_torque[
                [
                    "name",
                    "year",
                    "torque",
                    "selling_price"
                ]
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # HIGH TORQUE
    # --------------------------------------------------------

    print("\nVehicles with torque above 500 Nm:")

    high_torque = df[
        df["torque"] > 500
    ]

    if high_torque.empty:

        print("None found.")

    else:

        print(
            high_torque[
                [
                    "name",
                    "year",
                    "torque",
                    "selling_price"
                ]
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # UNUSUAL SEAT COUNTS
    # --------------------------------------------------------

    print("\nVehicles with unusual seat counts:")

    unusual_seats = df[
        (df["seats"] < 2) |
        (df["seats"] > 10)
    ]

    if unusual_seats.empty:

        print("None found.")

    else:

        print(
            unusual_seats[
                [
                    "name",
                    "year",
                    "seats",
                    "selling_price"
                ]
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # ZERO OR NEGATIVE VALUES
    # --------------------------------------------------------

    print("\nNegative numerical values:")

    negative_values = df[
        df[NUMERIC_COLUMNS]
        .select_dtypes(include="number")
        .lt(0)
        .any(axis=1)
    ]

    if negative_values.empty:

        print("None found.")

    else:

        print(
            negative_values[
                NUMERIC_COLUMNS
            ].to_string(index=False)
        )

    print("\nFinal quality audit completed.")


# ============================================================
# 9. CLEANED DATA SUMMARY
# ============================================================

def summarize_cleaned_data(df):
    """
    Display a summary of the cleaned dataset.
    """

    print("\n" + "=" * 60)
    print("CLEANED DATA SUMMARY")
    print("=" * 60)

    # --------------------------------------------------------
    # SHAPE
    # --------------------------------------------------------

    print("\nDataset shape:")

    print(df.shape)

    # --------------------------------------------------------
    # DATA TYPES
    # --------------------------------------------------------

    print("\nData types:")

    print(df.dtypes)

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    print("\nMissing values:")

    print(df.isnull().sum())

    # --------------------------------------------------------
    # NUMERICAL SUMMARY
    # --------------------------------------------------------

    print("\nNumerical summary:")

    print(
        df[NUMERIC_COLUMNS]
        .describe()
        .round(2)
    )


# ============================================================
# 10. DATA VALIDATION
# ============================================================

def validate_cleaned_data(df):
    """
    Validate the cleaned dataset.

    Raises an error if a critical validation fails.
    """

    print("\n" + "=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)

    # --------------------------------------------------------
    # COLUMN VALIDATION
    # --------------------------------------------------------

    missing_columns = set(
        EXPECTED_COLUMNS
    ) - set(df.columns)

    extra_columns = set(
        df.columns
    ) - set(EXPECTED_COLUMNS)

    if missing_columns:

        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if extra_columns:

        print(
            f"Warning: Extra columns detected: "
            f"{extra_columns}"
        )

    print("Column validation: PASSED")

    # --------------------------------------------------------
    # DUPLICATE VALIDATION
    # --------------------------------------------------------

    duplicate_count = df.duplicated().sum()

    if duplicate_count != 0:

        raise ValueError(
            f"Duplicate records remain: "
            f"{duplicate_count}"
        )

    print("Duplicate validation: PASSED")

    # --------------------------------------------------------
    # TARGET VALIDATION
    # --------------------------------------------------------

    if df["selling_price"].isnull().any():

        raise ValueError(
            "Missing values found in selling_price."
        )

    if (df["selling_price"] <= 0).any():

        raise ValueError(
            "Non-positive selling prices found."
        )

    print("Target validation: PASSED")

    # --------------------------------------------------------
    # NUMERIC DATATYPE VALIDATION
    # --------------------------------------------------------

    for column in NUMERIC_COLUMNS:

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):

            raise TypeError(
                f"{column} is not numeric."
            )

    print("Numerical datatype validation: PASSED")

    # --------------------------------------------------------
    # TORQUE VALIDATION
    # --------------------------------------------------------

    if (
        df["torque"].dropna() > 1000
    ).any():

        raise ValueError(
            "Torque values above 1000 Nm detected."
        )

    print("Torque validation: PASSED")

    # --------------------------------------------------------
    # NEGATIVE VALUE VALIDATION
    # --------------------------------------------------------

    non_negative_columns = [
        "selling_price",
        "km_driven",
        "mileage",
        "engine",
        "max_power",
        "torque",
        "seats",
    ]

    for column in non_negative_columns:

        negative_values = (
            df[column].dropna() < 0
        )

        if negative_values.any():

            raise ValueError(
                f"Negative values found in {column}."
            )

    print("Negative-value validation: PASSED")

    # --------------------------------------------------------
    # MISSING VALUE REPORT
    # --------------------------------------------------------

    print("\nMissing-value validation report:")

    print(df.isnull().sum())

    print("\nValidation completed successfully.")


# ============================================================
# 11. SAVE CLEANED DATASET
# ============================================================

def save_cleaned_dataset(df):
    """
    Save the cleaned dataset to the processed directory.
    """

    print("\n" + "=" * 60)
    print("SAVING CLEANED DATASET")
    print("=" * 60)

    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    print(
        "Dataset saved to:"
    )

    print(PROCESSED_DATA_PATH)

    print(
        f"Saved dataset shape: {df.shape}"
    )


# ============================================================
# 12. MAIN PIPELINE
# ============================================================

def main():
    """
    Execute the complete data cleaning pipeline.
    """

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # DUPLICATE ANALYSIS
    # --------------------------------------------------------

    analyze_duplicates(df)

    investigate_duplicates(df)

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    df = remove_duplicates(df)

    # --------------------------------------------------------
    # MISSING VALUE ANALYSIS
    # --------------------------------------------------------

    analyze_missing_values(df)

    investigate_missing_values(df)

    # --------------------------------------------------------
    # CLEAN SPECIFICATIONS
    # --------------------------------------------------------

    df = clean_specifications(df)

    # --------------------------------------------------------
    # SUSPICIOUS VALUE INVESTIGATION
    # --------------------------------------------------------

    investigate_suspicious_values(df)

    # --------------------------------------------------------
    # FINAL QUALITY AUDIT
    # --------------------------------------------------------

    final_quality_audit(df)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summarize_cleaned_data(df)

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validate_cleaned_data(df)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_cleaned_dataset(df)

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATA CLEANING PIPELINE COMPLETED")
    print("=" * 60)


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    main()