"""
Session Features Engineering

Aggregates session data into user-level behavioral features
for uplift modeling, CUPED adjustment, and propensity matching.
"""

import pandas as pd
import os

# Configuration
RAW_SESSIONS_PATH = "../data/sessions.csv"
OUTPUT_FEATURES_PATH = "../data/user_session_features.csv"


def generate_session_features(input_path=RAW_SESSIONS_PATH, output_path=OUTPUT_FEATURES_PATH):
    """
    Generate user-level behavioral features from session data.
    
    Args:
        input_path (str): Path to raw sessions data
        output_path (str): Path for output features file
    """
    print("Loading sessions data...")
    df = pd.read_csv(input_path)
    
    df['secs_elapsed'] = df['secs_elapsed'].fillna(0)

    print("Aggregating session metrics by user...")
    
    agg_ops = {
        "total_actions": ("action", "count"),
        "unique_actions": ("action", pd.Series.nunique),
        "total_secs_elapsed": ("secs_elapsed", "sum"),
        "avg_secs_per_action": ("secs_elapsed", "mean"),
        "median_secs_elapsed": ("secs_elapsed", "median"),
    }

    user_session_counts = df.groupby('user_id').size()
    
    agg_df = df.groupby("user_id").agg(**agg_ops).reset_index()
    
    agg_df = agg_df.merge(
        user_session_counts.reset_index().rename(columns={0: 'num_sessions'}),
        on='user_id', how='left'
    )

    agg_df = agg_df.fillna(0)
    
    agg_df['actions_per_session'] = agg_df['total_actions'] / agg_df['num_sessions'].clip(lower=1)
    agg_df['engagement_ratio'] = agg_df['unique_actions'] / agg_df['total_actions'].clip(lower=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    agg_df.to_csv(output_path, index=False)
    print(f"Session features saved to {output_path}. Shape: {agg_df.shape}")
    print(f"Features created: {list(agg_df.columns)}")


if __name__ == "__main__":
    generate_session_features()
