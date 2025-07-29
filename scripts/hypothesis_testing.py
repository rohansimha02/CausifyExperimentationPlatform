# scripts/hypothesis_testing.py

"""
This script performs hypothesis testing for the causal question:
"Did the treatment increase the booking rate?"
It now uses a two-proportion z-test on the booking rate and reports
95% confidence intervals for the difference in proportions.
"""

import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
import numpy as np

# Load cleaned user-level data
DATA_PATH = "../data/merged_users.csv"

def run_z_test(data_path=DATA_PATH):
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

    # Perform two-proportion z-test
    count = np.array([treatment_group["booking"].sum(), control_group["booking"].sum()])
    nobs = np.array([len(treatment_group), len(control_group)])
    stat, p_val = proportions_ztest(count, nobs)

    diff = treatment_rate - control_rate
    stderr = np.sqrt(treatment_rate * (1 - treatment_rate) / nobs[0] +
                     control_rate * (1 - control_rate) / nobs[1])
    ci_low = diff - 1.96 * stderr
    ci_high = diff + 1.96 * stderr

    print("Z-Test Results:")
    print(f"Z-statistic: {stat:.4f}")
    print(f"P-value:     {p_val:.4f}")
    print(f"Difference in booking rate: {diff:.4f}")
    print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

    if p_val < 0.05:
        print("Statistically significant difference (p < 0.05)")
    else:
        print("No statistically significant difference")


if __name__ == "__main__":
    run_z_test()
