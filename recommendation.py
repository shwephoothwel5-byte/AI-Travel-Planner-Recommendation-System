
"""Content-based travel recommendations using the fitted Chapter 4 model."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / 'Travel_Planner_Cleaned.csv')
preprocessor = joblib.load(BASE_DIR / 'travel_preprocessor.pkl')
destination_vectors = joblib.load(BASE_DIR / 'destination_vectors.pkl')
if destination_vectors.shape[0] != len(df):
    raise ValueError('Dataset and saved vectors have different row counts. Rerun model_training.py.')

def recommend(destination_type, travel_mode, best_season, stay_days, top_n=5):
    user = pd.DataFrame([{
        'Destination_Type': destination_type,
        'Travel_Mode': travel_mode,
        'Best_Season': best_season,
        'Recommended_Stay_Days': stay_days,
    }])
    user_vector = preprocessor.transform(user)
    scores = cosine_similarity(user_vector, destination_vectors).ravel()
    indices = np.argsort(-scores, kind='stable')[:top_n]
    results = df.iloc[indices].copy()
    results['Similarity_Score'] = scores[indices]
    return results

if __name__ == '__main__':
    print(recommend('Beach', 'Flight', 'November-March', 7)[['Destination_Name','Country','Similarity_Score']].to_string(index=False))
