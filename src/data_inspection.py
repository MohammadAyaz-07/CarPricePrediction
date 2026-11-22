# ============================================
# CAR PRICE PREDICTION PROJECT
# MODULE 1: DATA INSPECTION
# ============================================

from pathlib import Path
import pandas as pd


# --------------------------------------------
# 1. PROJECT PATH CONFIGURATION
# --------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "used_cars.csv"


# --------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------

def load_dataset(path: Path) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}"
        )

    return pd.read_csv(path, encoding="latin1")


# --------------------------------------------
# 3. DATASET OVERVIEW
# --------------------------------------------

def display_overview(df: pd.DataFrame) -> None:

    print("\n" + "=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)

    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    print("\nColumn names:")
    print(df.columns.tolist())


# --------------------------------------------
# 4. DISPLAY SAMPLE DATA
# --------------------------------------------

def display_sample(df: pd.DataFrame) -> None:

    print("\n" + "=" * 50)
    print("FIRST 5 RECORDS")
    print("=" * 50)

    print(df.head())


# --------------------------------------------
# 5. DATA TYPES
# --------------------------------------------

def display_data_types(df: pd.DataFrame) -> None:

    print("\n" + "=" * 50)
    print("DATA TYPES")
    print("=" * 50)

    print(df.dtypes)


# --------------------------------------------
# 6. MISSING VALUES
# --------------------------------------------

def display_missing_values(df: pd.DataFrame) -> None:

    print("\n" + "=" * 50)
    print("MISSING VALUES")
    print("=" * 50)

    missing_values = df.isnull().sum()

    missing_percentage = (
        df.isnull().mean() * 100
    )

    missing_report = pd.DataFrame({
        "Missing Count": missing_values,
        "Missing Percentage": missing_percentage
    })

    print(
        missing_report.sort_values(
            by="Missing Count",
            ascending=False
        )
    )


# --------------------------------------------
# 7. DUPLICATE RECORDS
# --------------------------------------------

def display_duplicates(df: pd.DataFrame) -> None:

    print("\n" + "=" * 50)
    print("DUPLICATE RECORDS")
    print("=" * 50)

    duplicate_count = df.duplicated().sum()

    print(f"Duplicate rows: {duplicate_count}")


# --------------------------------------------
# 8. STATISTICAL SUMMARY
# --------------------------------------------

def display_statistics(df: pd.DataFrame) -> None:

    print("\n" + "=" * 50)
    print("NUMERICAL STATISTICS")
    print("=" * 50)

    print(df.describe())


# --------------------------------------------
# 9. UNIQUE VALUES
# --------------------------------------------

def display_unique_values(df: pd.DataFrame) -> None:

    print("\n" + "=" * 50)
    print("UNIQUE VALUES")
    print("=" * 50)

    for column in df.columns:

        print(
            f"{column}: "
            f"{df[column].nunique()} unique values"
        )


# --------------------------------------------
# 10. TARGET VARIABLE ANALYSIS
# --------------------------------------------

def display_target_statistics(df: pd.DataFrame) -> None:

    target_column = "selling_price"

    if target_column not in df.columns:
        print("\nTarget column not found!")
        return

    print("\n" + "=" * 50)
    print("SELLING PRICE STATISTICS")
    print("=" * 50)

    print(df[target_column].describe())


# --------------------------------------------
# 11. MAIN EXECUTION
# --------------------------------------------

def main():

    print("Loading dataset...")

    df = load_dataset(DATA_PATH)

    display_overview(df)
    display_sample(df)
    display_data_types(df)
    display_missing_values(df)
    display_duplicates(df)
    display_statistics(df)
    display_unique_values(df)
    display_target_statistics(df)

    print("\nData inspection completed successfully!")


if __name__ == "__main__":
    main()