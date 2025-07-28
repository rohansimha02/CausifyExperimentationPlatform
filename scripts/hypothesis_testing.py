# scripts/hypothesis_testing.py

"""
This script performs basic hypothesis testing for the causal question:
"Did the treatment increase the booking rate?"
It uses a two-sample t-test to compare treatment vs control groups.
"""

import pandas as pd
from scipy.stats import ttest_ind
import numpy as np

# Load cleaned user-level data
DATA_PATH = "../data/merged_users.csv"

def run_t_test(data_path=DATA_PATH):
    print("Loading data for hypothesis testing...")
    df = pd.read_csv(data_path)

    # Split into treatment and control groups
    treatment_group = df[df["treatment"] == 1]
    control_group = df[df["treatment"] == 0]

    # Compute mean booking rate for both groups
    treatment_rate = treatment_group["booking"].mean()
    control_rate = control_group["booking"].mean()

    print(f"Treatment booking rate: {treatment_rate:.4f}")
    print(f"Control booking rate:   {control_rate:.4f}")

    # Perform two-sample t-test on booking outcomes
    t_stat, p_val = ttest_ind(
        treatment_group["booking"],
        control_group["booking"],
        equal_var=False
    )

    print("T-Test Results:")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value:     {p_val:.4f}")

    if p_val < 0.05:
        print("Statistically significant difference (p < 0.05)")
    else:
        print("No statistically significant difference")


if __name__ == "__main__":
    run_t_test()
