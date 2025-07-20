# scripts/merge_features.py

"""
This script merges the user-level session features with the cleaned user dataset.
The output is a single flat file containing all user attributes + behavioral features,
ready for causal inference and uplift modeling.
"""

import pandas as pd
import os

# Input paths
CLEAN_USERS_PATH = "./data/clean_users.csv"
SESSION_FEATURES_PATH = "./data/user_session_features.csv"
OUTPUT_PATH = "./data/merged_users.csv"


def merge_user_features(user_path=CLEAN_USERS_PATH, session_path=SESSION_FEATURES_PATH, output_path=OUTPUT_PATH):
    print("Loading cleaned users and session features...")
    users = pd.read_csv(user_path)
    sessions = pd.read_csv(session_path)

    print("Merging on user_id (inner join)...")
    # user_id was dropped in preprocessing — we now re-load with it from original dataset
    original_users = pd.read_csv("./data/train_users_2.csv", usecols=["id"])
    original_users.reset_index(drop=True, inplace=True)
    users.reset_index(drop=True, inplace=True)
    users_with_id = pd.concat([original_users, users], axis=1)

    merged = users_with_id.merge(sessions, how="inner", left_on="id", right_on="user_id")

    # Drop extra user_id column
    merged.drop(columns=["user_id"], inplace=True)

    # Save final merged file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Merged dataset saved to {output_path}. Shape: {merged.shape}")


if __name__ == "__main__":
    merge_user_features()