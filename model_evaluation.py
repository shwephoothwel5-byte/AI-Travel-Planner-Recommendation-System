
import pandas as pd
import time

from recommendation import recommend, df

test_cases = [
    ("Beach", "Flight", "November-March", 7),
    ("City", "Flight", "November-March", 5),
    ("Nature", "Bus", "June-August", 3)
]

evaluation_results = []

for destination, mode, season, days in test_cases:

    start = time.perf_counter()

    recommendations = recommend(
        destination,
        mode,
        season,
        days
    )

    runtime = time.perf_counter() - start

    # Relevant destinations in entire dataset
    relevant = (
        (df["Destination_Type"] == destination) &
        (df["Travel_Mode"] == mode)
    )

    total_relevant = relevant.sum()

    # Relevant destinations in Top 5
    relevant_top5 = (
        (recommendations["Destination_Type"] == destination) &
        (recommendations["Travel_Mode"] == mode)
    ).sum()

    precision = relevant_top5 / len(recommendations)

    recall = (
        relevant_top5 / total_relevant
        if total_relevant > 0 else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    evaluation_results.append({
        "Test Case": f"{destination} + {mode} + {days} Days",
        "Precision@5": round(precision, 4),
        "Recall@5": round(recall, 4),
        "F1-Score": round(f1, 4),
        "Runtime (s)": round(runtime, 6)
    })

evaluation_df = pd.DataFrame(evaluation_results)

print("\n===== MODEL EVALUATION RESULTS =====")
print(evaluation_df.to_string(index=False))

evaluation_df.to_csv(
    "travel_planner_comparison_results.csv",
    index=False
)

print("\nEvaluation results saved successfully!")
