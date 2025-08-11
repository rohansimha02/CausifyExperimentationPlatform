"""
Data Preprocessing

Loads and cleans Airbnb user dataset, creates binary booking outcome,
and simulates randomized treatment assignment for experimentation.
"""

import pandas as pd
import numpy as np
import os

# Configuration
RAW_DATA_PATH = "../data/train_users_2.csv"
OUTPUT_DATA_PATH = "../data/clean_users.csv"


def preprocess_airbnb_data(input_path=RAW_DATA_PATH, output_path=OUTPUT_DATA_PATH, seed=42):
    """
    Clean and prepare Airbnb user data for experimentation analysis.
    
    Args:
        input_path (str): Path to raw data file
        output_path (str): Path for cleaned output file
        seed (int): Random seed for reproducible treatment assignment
    """
    print("Loading raw Airbnb data...")
    df = pd.read_csv(input_path)

    # Remove leakage columns while preserving user ID for downstream joins
    df.drop(columns=["date_first_booking"], inplace=True)

    # Create binary booking outcome (1 = booking made, 0 = no booking)
    df["booking"] = (df["country_destination"] != "NDF").astype(int)

    # Simulate randomized treatment assignment (50/50 split)
    np.random.seed(seed)
    df['treatment'] = np.random.binomial(1, 0.5, size=len(df))
    
    # Apply data quality filters
    df = df[(df["gender"] != "-unknown-") & (df["age"].between(15, 90))]

    # Select features for analysis
    keep_cols = [
        "id", "signup_method", "signup_flow", "gender", "age", "language",
        "affiliate_channel", "affiliate_provider", "signup_app", "first_device_type",
        "first_browser", "booking", "treatment"
    ]
    df_clean = df.loc[:, ~df.columns.duplicated()]
    df_clean = df_clean[keep_cols]

    # Engineer additional features
    df_clean["age_squared"] = df_clean["age"] ** 2
    df_clean["age_bucket"] = pd.cut(
        df_clean["age"], bins=[15, 25, 35, 45, 55, 65, 90], labels=False
    )

    # Encode categorical variables for ML compatibility
    cat_cols = [
        "signup_method", "language", "affiliate_channel", "affiliate_provider",
        "signup_app", "first_device_type", "first_browser", "gender"
    ]
    for col in cat_cols:
        df_clean[col] = pd.factorize(df_clean[col])[0]

    # Save cleaned dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}. Shape: {df_clean.shape}")


if __name__ == "__main__":
    preprocess_airbnb_data()
