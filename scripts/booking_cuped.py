"""
CUPED Variance Reduction

Applies CUPED adjustment using behavioral features to reduce
outcome variance and improve statistical power.
"""

import pandas as pd
import numpy as np
import os

# Configuration
DATA_PATH = "../data/merged_users.csv"
OUTPUT_PATH = "../data/merged_users_cuped.csv"

# Covariates for variance reduction
CANDIDATE_COVARIATES = [
    "total_actions",
    "unique_actions", 
    "avg_secs_per_action",
    "total_secs_elapsed",
    "median_secs_elapsed",
]


def auto_cuped(data_path=DATA_PATH, output_path=OUTPUT_PATH, holdout_frac=0.5):
    """
    Apply multivariate CUPED using holdout split for theta estimation.
    
    Args:
        data_path (str): Path to input data
        output_path (str): Path for CUPED-adjusted output
        holdout_frac (float): Fraction of data for theta estimation
    """
    print("Loading merged user data...")
    df = pd.read_csv(data_path)

    # Clean data for CUPED analysis
    df = df.dropna(subset=CANDIDATE_COVARIATES)

    y = df["booking"].values
    X = df[CANDIDATE_COVARIATES].values

    # Create holdout split for unbiased theta estimation
    n_holdout = int(len(df) * holdout_frac)
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(df))
    idx_hold = perm[:n_holdout]
    idx_apply = perm[n_holdout:]

    # Center covariates and outcome on holdout set
    X_hold = X[idx_hold] - X[idx_hold].mean(axis=0)
    y_hold = y[idx_hold] - y[idx_hold].mean()

    # Estimate CUPED adjustment coefficients
    theta = np.linalg.pinv(X_hold.T @ X_hold) @ (X_hold.T @ y_hold)

    # Apply variance reduction to full dataset
    # CRITICAL: Use holdout means for centering to avoid data leakage
    X_centered = X - X[idx_hold].mean(axis=0)
    adjustment = (X_centered @ theta)
    df["booking_cuped"] = y - adjustment

    # Calculate adjusted treatment effects
    treatment_mean = df.loc[df["treatment"] == 1, "booking_cuped"].mean()
    control_mean = df.loc[df["treatment"] == 0, "booking_cuped"].mean()
    lift = treatment_mean - control_mean

    print("\nCUPED Results:")
    print(f"Adjustment coefficients (theta): {theta}")
    print(f"Adjusted Treatment Mean: {treatment_mean:.4f}")
    print(f"Adjusted Control Mean: {control_mean:.4f}")
    print(f"Estimated Lift: {lift:.4f}")

    # Save CUPED-adjusted dataset
    df.to_csv(output_path, index=False)
    print(f"\nCUPED-adjusted data saved to {output_path}")


if __name__ == "__main__":
    auto_cuped()
