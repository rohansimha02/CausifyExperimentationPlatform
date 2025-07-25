# scripts/run_pipeline.py

"""
End-to-end pipeline runner for the Causify project.
This script executes the full experimental workflow:
1. Data preprocessing
2. Session feature aggregation
3. Merging user + session data
4. Hypothesis testing
5. CUPED adjustment
6. Propensity Score Matching
7. Uplift modeling
"""

import subprocess
import os

STAGES = [
    ("Preprocessing user data", "preprocess_data.py"),
    ("Generating session features", "session_features.py"),
    ("Merging user and session data", "merge_features.py"),
    ("Running hypothesis test (t-test)", "hypothesis_testing.py"),
    ("Running CUPED adjustment (using unique_actions)", "booking_cuped.py"),
    ("Running Propensity Score Matching (on booking_cuped)", "causal_inference.py"),
    ("Running uplift modeling", "uplift_modeling.py"),
    ("Generating dashboard data", "generate_dashboard_data.py")
]


def run_pipeline():
    print("\nStarting full Causify pipeline...\n")
    for desc, script in STAGES:
        print(f"{desc} — {script}")
        subprocess.run(["python", script], cwd=os.path.dirname(__file__))
        print("\n" + "-" * 60 + "\n")

    print("All stages complete. Uplift scores available at /data/uplift_scores.csv")


if __name__ == "__main__":
    run_pipeline()
