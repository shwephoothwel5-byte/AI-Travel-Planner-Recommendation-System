
# ==============================================
# AI Travel Planner Recommendation System
# Chapter 4.3 - Model Training
# Content-Based Filtering + Cosine Similarity
# ==============================================

import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# STEP 1: Load cleaned dataset
BASE_DIR = Path(__file__).resolve().parent

dataset_path = BASE_DIR / "Travel_Planner_Cleaned.csv"

df = pd.read_csv(dataset_path)

print("\n===== STEP 1: DATASET LOADED =====")
print("Dataset Shape:", df.shape)

# STEP 2: Select recommendation features
categorical_features = [
    "Destination_Type",
    "Travel_Mode",
    "Best_Season"
]

numerical_features = [
    "Recommended_Stay_Days"
]

selected_features = categorical_features + numerical_features

X = df[selected_features].copy()

# Check for missing values
if X.isnull().any().any():
    raise ValueError(
        "Missing values detected. Run data_preparation.py first."
    )

print("\n===== STEP 2: SELECTED FEATURES =====")
print(X.head().to_string(index=False))

# STEP 3: Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features
        )
    ]
)

# STEP 4: Fit and transform destination features
destination_vectors = preprocessor.fit_transform(X)

print("\n===== STEP 3: FEATURE TRANSFORMATION =====")
print("Original Feature Shape:", X.shape)
print("Transformed Feature Shape:", destination_vectors.shape)

# STEP 5: Save trained preprocessing model
joblib.dump(
    preprocessor,
    BASE_DIR / "travel_preprocessor.pkl"
)

# Save destination feature matrix
joblib.dump(
    destination_vectors,
    BASE_DIR / "destination_vectors.pkl"
)

print("\n===== STEP 4: MODEL SAVED =====")
print("travel_preprocessor.pkl")
print("destination_vectors.pkl")

# STEP 6: Example user preferences
user_preferences = pd.DataFrame([{
    "Destination_Type": "Beach",
    "Travel_Mode": "Flight",
    "Best_Season": "November-March",
    "Recommended_Stay_Days": 7
}])

# Apply same preprocessing to user preferences
user_vector = preprocessor.transform(user_preferences)

# STEP 7: Calculate cosine similarity
similarity_scores = cosine_similarity(
    user_vector,
    destination_vectors
).flatten()

# STEP 8: Rank Top 5 destinations
top_indices = np.argsort(
    -similarity_scores,
    kind="stable"
)[:5]

recommendations = df.iloc[top_indices].copy()

recommendations["Similarity_Score"] = (
    similarity_scores[top_indices]
)

# STEP 9: Display recommendation results
print("\n===== STEP 5: TOP 5 RECOMMENDATIONS =====")

output_columns = [
    "Destination_Name",
    "Country",
    "Destination_Type",
    "Travel_Mode",
    "Best_Season",
    "Recommended_Stay_Days",
    "Similarity_Score"
]

print(
    recommendations[output_columns].to_string(index=False)
)

# STEP 10: Save results
recommendations[output_columns].to_csv(
    BASE_DIR / "model_training_results.csv",
    index=False
)

print("\nModel preparation completed successfully!")
