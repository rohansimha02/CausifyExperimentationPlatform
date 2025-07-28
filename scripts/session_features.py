# scripts/session_features.py

"""
This script aggregates the Airbnb sessions dataset into per-user behavioral features.
These features can be joined with user-level data for uplift modeling, CUPED, and matching.
"""

import pandas as pd
import os

# Set paths
RAW_SESSIONS_PATH = "../data/sessions.csv"
OUTPUT_FEATURES_PATH = "../data/user_session_features.csv"


def generate_session_features(input_path=RAW_SESSIONS_PATH, output_path=OUTPUT_FEATURES_PATH):
    print("Loading sessions data...")
    df = pd.read_csv(input_path)

    print("Creating session-level aggregates per user...")
    # Aggregate session metrics by user_id
    agg_df = df.groupby("user_id").agg(
        total_actions=("action", "count"),
        unique_actions=("action", pd.Series.nunique),
        total_secs_elapsed=("secs_elapsed", "sum"),
        avg_secs_per_action=("secs_elapsed", "mean")
    ).reset_index()

    # Drop NaNs (some users may not have valid session durations)
    agg_df = agg_df.dropna()

    # Save features
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    agg_df.to_csv(output_path, index=False)
    print(f"Session features saved to {output_path}. Shape: {agg_df.shape}")


if __name__ == "__main__":
    generate_session_features()
