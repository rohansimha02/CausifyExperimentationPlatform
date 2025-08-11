"""
Feature Merging

Merges user data with session features and simulates
treatment effects for causal inference analysis.
"""

import pandas as pd
import numpy as np
import os

# Configuration
CLEAN_USERS_PATH = "../data/clean_users.csv"
SESSION_FEATURES_PATH = "../data/user_session_features.csv"
OUTPUT_PATH = "../data/merged_users.csv"


def merge_user_features(user_path=CLEAN_USERS_PATH, session_path=SESSION_FEATURES_PATH, output_path=OUTPUT_PATH):
    """
    Merge user data with session features and simulate booking outcomes.
    
    Args:
        user_path (str): Path to cleaned user data
        session_path (str): Path to session features
        output_path (str): Path for merged output
    """
    print("Loading cleaned users and session features...")
    users = pd.read_csv(user_path)
    sessions = pd.read_csv(session_path)

    # Validate required columns
    if "id" not in users.columns:
        raise ValueError("Column 'id' is missing from cleaned users file.")
    if "user_id" not in sessions.columns:
        raise ValueError("Column 'user_id' is missing from session features file.")

    print("Merging datasets on user ID...")
    merged = users.merge(sessions, how="inner", left_on="id", right_on="user_id")
    merged.drop(columns=["user_id"], inplace=True)

    # Ensure treatment assignment exists
    if 'treatment' not in merged.columns:
        print("WARNING: Treatment assignment not found. Creating random assignment...")
        np.random.seed(42)
        merged['treatment'] = np.random.binomial(1, 0.5, size=len(merged))
    else:
        print("Treatment assignment found from preprocessing step.")

    # Simulate booking outcomes with treatment effect
    print("Simulating booking outcomes with treatment effect...")
    np.random.seed(42)
    merged['booking'] = np.where(
        merged['treatment'] == 1,
        np.random.binomial(1, 0.25, size=len(merged)),  # Treatment: 25% booking rate
        np.random.binomial(1, 0.15, size=len(merged))   # Control: 15% booking rate
    )

    # Save merged dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Merged dataset saved to {output_path}. Shape: {merged.shape}")
    print("Treatment effect simulation complete.")


if __name__ == "__main__":
    merge_user_features()
