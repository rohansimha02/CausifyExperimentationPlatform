"""
This script merges the user-level session features with the cleaned user dataset.
The output is a single flat file containing all user attributes + behavioral features,
ready for causal inference and uplift modeling.
"""

import pandas as pd
import numpy as np
import os

# Input paths
CLEAN_USERS_PATH = "./data/clean_users.csv"
SESSION_FEATURES_PATH = "./data/user_session_features.csv"
OUTPUT_PATH = "./data/merged_users.csv"

def merge_user_features(user_path=CLEAN_USERS_PATH, session_path=SESSION_FEATURES_PATH, output_path=OUTPUT_PATH):
    print("Loading cleaned users and session features...")
    users = pd.read_csv(user_path)
    sessions = pd.read_csv(session_path)

    if "id" not in users.columns:
        raise ValueError("Column 'id' is missing from cleaned users file.")
    if "user_id" not in sessions.columns:
        raise ValueError("Column 'user_id' is missing from session features file.")

    print("Merging on user_id (inner join)...")
    merged = users.merge(sessions, how="inner", left_on="id", right_on="user_id")

    # Drop extra user_id column
    merged.drop(columns=["user_id"], inplace=True)

    # Inject treatment and synthetic booking outcome
    print("Injecting treatment assignment and simulated booking outcome...")
    np.random.seed(42)
    merged['treatment'] = np.random.binomial(1, 0.5, size=len(merged))

    merged['booking'] = np.where(
        merged['treatment'] == 1,
        np.random.binomial(1, 0.25, size=len(merged)),  # treated group uplift
        np.random.binomial(1, 0.15, size=len(merged))   # control group baseline
    )

    # Save final merged file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Merged dataset saved to {output_path}. Shape: {merged.shape}")
    print("Treatment + uplift simulation complete.")


if __name__ == "__main__":
    merge_user_features()
