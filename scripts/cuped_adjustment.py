"""
This script applies the CUPED method to reduce variance in the treatment effect
by adjusting for a pre-treatment covariate (unique_actions).
"""

import pandas as pd
import os

# Paths
MERGED_DATA_PATH = "../data/merged_users.csv"
OUTPUT_PATH = "../data/merged_users_cuped.csv"

def apply_cuped(data_path=MERGED_DATA_PATH, output_path=OUTPUT_PATH):
    print("Loading merged user + session feature data...")
    df = pd.read_csv(data_path)

    # Drop rows with missing pre-treatment covariate
    df = df.dropna(subset=["unique_actions"])

    # Define outcome and covariate
    y = df["booking"]
    x = df["unique_actions"]

    # Compute theta (covariate adjustment factor)
    theta = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()

    # Apply CUPED adjustment
    df["booking_cuped"] = df["booking"] - theta * (df["unique_actions"] - x.mean())

    # Compute group means
    treatment_mean = df[df["treatment"] == 1]["booking_cuped"].mean()
    control_mean = df[df["treatment"] == 0]["booking_cuped"].mean()
    lift = treatment_mean - control_mean

    # Print results
    print("\n CUPED Results:")
    print(f"Theta (CUPED adjustment coefficient): {theta:.4f}")
    print(f"Adjusted Treatment Mean: {treatment_mean:.4f}")
    print(f"Adjusted Control Mean:   {control_mean:.4f}")
    print(f"Estimated Lift (CUPED):   {lift:.4f}")

    # Save file
    df.to_csv(output_path, index=False)
    print(f"\n CUPED-adjusted data saved to {output_path}")


if __name__ == "__main__":
    apply_cuped()
