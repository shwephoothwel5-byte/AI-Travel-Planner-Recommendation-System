import pandas as pd

df = pd.read_csv(
    "Travel_Planner_Dataset.csv",
    encoding="latin1"
)

print("TRAVEL PLANNER DATA ANALYSIS")
print("----------------------------")

print("Dataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nSample Records:")
print(df.head())