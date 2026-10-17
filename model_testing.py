
from recommendation import recommend

test_cases = [
    ("Beach", "Flight", "November-March", 7),
    ("City", "Flight", "November-March", 5),
    ("Nature", "Bus", "June-August", 3)
]

for i, (destination, mode, season, days) in enumerate(test_cases, 1):

    print(f"\n========== TEST CASE {i} ==========")

    results = recommend(
        destination,
        mode,
        season,
        days
    )

    print(results.to_string(index=False))
