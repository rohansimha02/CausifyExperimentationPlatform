# scripts/cuped_adjustment.py

"""
This script applies the CUPED method to reduce variance in the treatment effect
by adjusting for a pre-treatment covariate (e.g., total_secs_elapsed).
"""

import pandas as pd
import os

# Paths
MERGED_DATA_PATH = "../data/merged_users.csv"


def apply_cuped(data_path=MERGED_DATA_PATH):
    print("Loading merged user + session feature data...")
    df = pd.read_csv(data_path)

    # Define outcome and covariate
    y = df["booking"]
    x = df["total_secs_elapsed"]

    # Compute theta (coefficient from linear regression of y ~ x)
    theta = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()

    # Create adjusted outcome
    df["booking_cuped"] = df["booking"] - theta * (df["total_secs_elapsed"] - x.mean())

    # Compute CUPED-adjusted mean booking rates
    treatment_mean = df[df["treatment"] == 1]["booking_cuped"].mean()
    control_mean = df[df["treatment"] == 0]["booking_cuped"].mean()

    # Compute adjusted lift
    lift = treatment_mean - control_mean

    print("\n📊 CUPED Results:")
    print(f"Theta (CUPED adjustment coefficient): {theta:.4f}")
    print(f"Adjusted Treatment Mean: {treatment_mean:.4f}")
    print(f"Adjusted Control Mean:   {control_mean:.4f}")
    print(f"Estimated Lift (CUPED):   {lift:.4f}")


if __name__ == "__main__":
    apply_cuped()
