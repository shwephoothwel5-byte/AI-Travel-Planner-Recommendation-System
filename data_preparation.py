# ============================================================
# Travel Planner Recommendation System
# Chapter 3 - Data Preparation
# ============================================================

import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 3.2 OVERVIEW OF DATASET
# ------------------------------------------------------------

print("=" * 70)
print("3.2 OVERVIEW OF DATASET")
print("=" * 70)

# Load CSV dataset
df = pd.read_csv("Travel_Planner_Dataset.csv", encoding="latin1")

print("\nDataset loaded successfully.")

# Number of rows and columns
print("\nDataset Shape:")
print(df.shape)

print("\nNumber of Records:")
print(len(df))

print("\nNumber of Attributes:")
print(len(df.columns))


# ------------------------------------------------------------
# 3.2.1 DATASET ATTRIBUTES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("3.2.1 DATASET ATTRIBUTES")
print("=" * 70)

print("\nColumn Names:")
for number, column in enumerate(df.columns, start=1):
    print(number, "-", column)

print("\nData Types:")
print(df.dtypes)


# ------------------------------------------------------------
# 3.2.2 SAMPLE DATASET
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("3.2.2 SAMPLE DATASET")
print("=" * 70)

print("\nFirst 10 Records:")
print(df.head(10).to_string())


# ------------------------------------------------------------
# 3.3 DATA CLEANING AND PREPROCESSING
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("3.3 DATA CLEANING AND PREPROCESSING")
print("=" * 70)

# ----- Check missing values -----

print("\nMissing Values BEFORE Cleaning:")
missing_before = df.isnull().sum()

print(missing_before)

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())


# ----- Check duplicate records -----

duplicates_before = df.duplicated().sum()

print("\nDuplicate Records BEFORE Cleaning:")
print(duplicates_before)


# ----- Remove duplicates -----

df_clean = df.drop_duplicates().copy()

print("\nDuplicate Records AFTER Cleaning:")
print(df_clean.duplicated().sum())


# ----- Remove unnecessary spaces -----

text_columns = df_clean.select_dtypes(include=["object"]).columns

for column in text_columns:
    df_clean[column] = df_clean[column].apply(
        lambda value: value.strip()
        if isinstance(value, str)
        else value
    )


# ----- Standardize missing text values -----

for column in text_columns:

    if df_clean[column].isnull().any():

        df_clean[column] = df_clean[column].fillna("Unknown")


# ----- Fill missing numerical values -----

numeric_columns = df_clean.select_dtypes(
    include=[np.number]
).columns

for column in numeric_columns:

    if df_clean[column].isnull().any():

        median_value = df_clean[column].median()

        df_clean[column] = df_clean[column].fillna(
            median_value
        )


print("\nMissing Values AFTER Cleaning:")
print(df_clean.isnull().sum())

print("\nTotal Missing Values AFTER Cleaning:")
print(df_clean.isnull().sum().sum())


# ------------------------------------------------------------
# CHECK DATA TYPES
# ------------------------------------------------------------

print("\nData Types After Cleaning:")
print(df_clean.dtypes)


# ------------------------------------------------------------
# REMOVE REDUNDANT DATA
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("REDUNDANT DATA CHECK")
print("-" * 70)

print("\nOriginal Number of Records:")
print(len(df))

print("\nRecords After Removing Duplicates:")
print(len(df_clean))

print("\nNumber of Removed Duplicate Records:")
print(len(df) - len(df_clean))


# ------------------------------------------------------------
# 3.3 CATEGORICAL DATA PREPARATION
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("CATEGORICAL DATA PREPARATION")
print("-" * 70)

print("\nCategorical Attributes:")

for column in text_columns:
    print("-", column)


# Example:
# Convert Destination_Type into machine-readable values
if "Destination_Type" in df_clean.columns:

    destination_encoded = pd.get_dummies(
        df_clean["Destination_Type"],
        prefix="Destination"
    )

    print("\nDestination Type Encoding Sample:")

    print(
        destination_encoded.head().to_string()
    )


# ------------------------------------------------------------
# NUMERICAL DATA PREPARATION
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("NUMERICAL DATA PREPARATION")
print("-" * 70)

print("\nNumerical Attributes:")

for column in numeric_columns:
    print("-", column)


# ------------------------------------------------------------
# 3.4 DATA CONSISTENCY AND INTEGRITY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("3.4 DATA CONSISTENCY AND INTEGRITY")
print("=" * 70)


# ----- Check unique destination names -----

if "Destination_Name" in df_clean.columns:

    print("\nNumber of Unique Destinations:")

    print(
        df_clean["Destination_Name"].nunique()
    )


# ----- Check negative numerical values -----

print("\nNegative Value Validation:")

for column in numeric_columns:

    negative_count = (
        df_clean[column] < 0
    ).sum()

    print(
        column,
        ":",
        negative_count,
        "negative values"
    )


# ----- Check missing values again -----

print("\nMissing Value Validation:")

print(
    "Total Missing Values:",
    df_clean.isnull().sum().sum()
)


# ----- Check duplicates again -----

print("\nDuplicate Validation:")

print(
    "Total Duplicate Records:",
    df_clean.duplicated().sum()
)


# ----- Dataset dimensions -----

print("\nFinal Dataset Dimensions:")

print(
    "Rows:",
    df_clean.shape[0]
)

print(
    "Columns:",
    df_clean.shape[1]
)


# ------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL DATA VALIDATION REPORT")
print("=" * 70)

print(
    "\nTotal Records:",
    len(df_clean)
)

print(
    "Total Attributes:",
    len(df_clean.columns)
)

print(
    "Missing Values:",
    df_clean.isnull().sum().sum()
)

print(
    "Duplicate Records:",
    df_clean.duplicated().sum()
)


# ------------------------------------------------------------
# SAVE CLEANED DATASET
# ------------------------------------------------------------

df_clean.to_csv(
    "Travel_Planner_Cleaned.csv",
    index=False
)

print(
    "\nCleaned dataset saved successfully as:"
)

print(
    "Travel_Planner_Cleaned.csv"
)

print("\n" + "=" * 70)
print("DATA PREPARATION COMPLETED SUCCESSFULLY")
print("=" * 70)