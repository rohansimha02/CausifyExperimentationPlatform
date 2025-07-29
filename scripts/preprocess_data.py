"""
This script loads and cleans the Airbnb Kaggle user dataset,
creates a binary booking outcome, and simulates a randomized treatment group.
The output is a clean CSV file for downstream causal inference and experimentation analysis.
"""

import pandas as pd
import numpy as np
import os

# Set paths
RAW_DATA_PATH = "../data/train_users_2.csv"
OUTPUT_DATA_PATH = "../data/clean_users.csv"


def preprocess_airbnb_data(input_path=RAW_DATA_PATH, output_path=OUTPUT_DATA_PATH, seed=42):
    # Load raw dataset
    print("Loading data...")
    df = pd.read_csv(input_path)

    # Drop known leakage columns, but keep 'id'
    df.drop(columns=["date_first_booking"], inplace=True)

    # Create binary booking outcome (1 = made a booking, 0 = NDF = no booking)
    df["booking"] = (df["country_destination"] != "NDF").astype(int)

    # Simulate a randomized treatment assignment (50% of users)
    np.random.seed(42)
    df['treatment'] = np.random.binomial(1, 0.5, size=len(df))
    
    # Filter for valid entries (e.g., gender != unknown, reasonable age)
    df = df[(df["gender"] != "-unknown-") & (df["age"].between(15, 90))]

    # Select relevant columns (including 'id' for downstream merge)
    keep_cols = [
        "id",
        "signup_method",
        "signup_flow",
        "gender",
        "age",
        "language",
        "affiliate_channel",
        "affiliate_provider",
        "signup_app",
        "first_device_type",
        "first_browser",
        "booking",
        "treatment",
    ]
    df_clean = df.loc[:, ~df.columns.duplicated()]
    df_clean = df_clean[keep_cols]

    # Create engineered features
    df_clean["age_squared"] = df_clean["age"] ** 2
    df_clean["age_bucket"] = pd.cut(
        df_clean["age"], bins=[15, 25, 35, 45, 55, 65, 90], labels=False
    )

    # Encode categorical variables numerically for modeling convenience
    cat_cols = [
        "signup_method",
        "language",
        "affiliate_channel",
        "affiliate_provider",
        "signup_app",
        "first_device_type",
        "first_browser",
        "gender",
    ]
    for col in cat_cols:
        df_clean[col] = pd.factorize(df_clean[col])[0]

    # Save cleaned file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}. Shape: {df_clean.shape}")


if __name__ == "__main__":
    preprocess_airbnb_data()
