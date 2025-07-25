"""
Auto-CUPED: Automatically selects the best covariate for CUPED adjustment
from session-level features to reduce outcome variance.
"""

import pandas as pd
import os

DATA_PATH = "./data/merged_users.csv"
OUTPUT_PATH = "./data/merged_users_cuped.csv"

CANDIDATE_COVARIATES = [
    "total_actions", "unique_actions", "avg_secs_per_action", "total_secs_elapsed"
]


def auto_cuped(data_path=DATA_PATH, output_path=OUTPUT_PATH):
    print("Loading data...")
    df = pd.read_csv(data_path)

    # Filter valid rows
    df = df.dropna(subset=CANDIDATE_COVARIATES)

    y = df["booking"]
    best_cov = None
    best_corr = 0
    best_theta = 0

    for cov in CANDIDATE_COVARIATES:
        x = df[cov]
        corr = abs(x.corr(y))
        if corr > best_corr:
            best_corr = corr
            best_cov = cov
            theta = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
            best_theta = theta

    print(f"\nBest covariate for CUPED: {best_cov} (corr = {best_corr:.4f})")
    df["booking_cuped"] = y - best_theta * (df[best_cov] - df[best_cov].mean())

    treatment_mean = df[df["treatment"] == 1]["booking_cuped"].mean()
    control_mean = df[df["treatment"] == 0]["booking_cuped"].mean()
    lift = treatment_mean - control_mean

    print(f"Theta: {best_theta:.4f}")
    print(f"Adjusted Treatment Mean: {treatment_mean:.4f}")
    print(f"Adjusted Control Mean:   {control_mean:.4f}")
    print(f"Estimated Lift:          {lift:.4f}")

    df.to_csv(output_path, index=False)
    print(f"\nSaved adjusted data to {output_path}")


if __name__ == "__main__":
    auto_cuped()
