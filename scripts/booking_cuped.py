"""
Auto-CUPED: Automatically selects the best covariate for CUPED adjustment
from session-level features to reduce outcome variance.
"""

import pandas as pd
import numpy as np
import os

DATA_PATH = "../data/merged_users.csv"
OUTPUT_PATH = "../data/merged_users_cuped.csv"

CANDIDATE_COVARIATES = [
    "total_actions",
    "unique_actions",
    "avg_secs_per_action",
    "total_secs_elapsed",
    "median_secs_elapsed",
]


def auto_cuped(data_path=DATA_PATH, output_path=OUTPUT_PATH, holdout_frac=0.5):
    """Apply multivariate CUPED using a hold-out split for theta estimation."""
    print("Loading data...")
    df = pd.read_csv(data_path)

    # Filter valid rows
    df = df.dropna(subset=CANDIDATE_COVARIATES)

    y = df["booking"].values
    X = df[CANDIDATE_COVARIATES].values

    # Split into holdout for theta estimation
    n_holdout = int(len(df) * holdout_frac)
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(df))
    idx_hold = perm[:n_holdout]
    idx_apply = perm[n_holdout:]

    X_hold = X[idx_hold] - X[idx_hold].mean(axis=0)
    y_hold = y[idx_hold] - y[idx_hold].mean()

    # Compute theta via multivariate regression
    theta = np.linalg.pinv(X_hold.T @ X_hold) @ (X_hold.T @ y_hold)

    # Apply adjustment to full data
    X_centered = X - X.mean(axis=0)
    adjustment = (X_centered @ theta)
    df["booking_cuped"] = y - adjustment

    treatment_mean = df.loc[df["treatment"] == 1, "booking_cuped"].mean()
    control_mean = df.loc[df["treatment"] == 0, "booking_cuped"].mean()
    lift = treatment_mean - control_mean

    print("\nCUPED with multiple covariates")
    print("Theta:", theta)
    print(f"Adjusted Treatment Mean: {treatment_mean:.4f}")
    print(f"Adjusted Control Mean:   {control_mean:.4f}")
    print(f"Estimated Lift:          {lift:.4f}")

    df.to_csv(output_path, index=False)
    print(f"\nSaved adjusted data to {output_path}")


if __name__ == "__main__":
    auto_cuped()
