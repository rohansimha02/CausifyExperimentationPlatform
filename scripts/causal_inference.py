"""
This script estimates the Average Treatment Effect (ATE) using
Propensity Score Matching (PSM) with CUPED-adjusted outcomes.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import os

MERGED_DATA_PATH = "../data/merged_users_cuped.csv" 
OUTPUT_MATCHED_PATH = "../data/matched_users.csv"

def standardized_mean_diff(x_treat, x_ctrl):
    return (x_treat.mean() - x_ctrl.mean()) / np.sqrt(0.5 * (x_treat.var() + x_ctrl.var()))


def estimate_ate_with_psm(
    data_path=MERGED_DATA_PATH,
    output_path=OUTPUT_MATCHED_PATH,
    n_neighbors=5,
    caliper=0.05,
):
    print("Loading CUPED-adjusted data...")
    df = pd.read_csv(data_path)

    # Covariates (pre-treatment only)
    covariates = ["age", "unique_actions", "total_actions", "total_secs_elapsed"]
    df = df.dropna(subset=covariates + ["booking_cuped"])

    # Scale covariates
    scaler = StandardScaler()
    X = scaler.fit_transform(df[covariates])

    # Estimate propensity scores
    print("Estimating propensity scores...")
    model = LogisticRegression()
    model.fit(X, df["treatment"])
    df["propensity_score"] = model.predict_proba(X)[:, 1]

    # Split into treated/control
    treated = df[df["treatment"] == 1].copy()
    control = df[df["treatment"] == 0].copy()

    print("Matching treated users with nearest control neighbors...")
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(control[["propensity_score"]])
    distances, indices = nbrs.kneighbors(treated[["propensity_score"]])

    matched_rows = []
    for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
        mask = dist_row <= caliper
        if not mask.any():
            continue
        chosen = idx_row[mask][0]
        matched_rows.append((i, chosen))

    matched_treated = treated.iloc[[i for i, _ in matched_rows]].reset_index(drop=True)
    matched_control = control.iloc[[j for _, j in matched_rows]].reset_index(drop=True)

    # Estimate ATE on CUPED-adjusted outcome
    y_treat = matched_treated["booking_cuped"].values
    y_control = matched_control["booking_cuped"].values
    ate = (y_treat - y_control).mean()

    print("\nPSM Results (on CUPED-adjusted outcome):")
    print(f"Estimated ATE: {ate:.4f}")
    print(f"Matched sample size: {len(matched_treated)} pairs")

    # Optionally save matched dataset
    matched_treated["group"] = "treated"
    matched_control["group"] = "control"
    matched_df = pd.concat([matched_treated, matched_control], axis=0)
    matched_df.to_csv(output_path, index=False)
    print(f"Matched dataset saved to {output_path}")

    # Diagnostics: Standardized mean differences
    print("\nStandardized Mean Differences after matching:")
    for cov in covariates:
        smd_before = standardized_mean_diff(treated[cov], control[cov])
        smd_after = standardized_mean_diff(matched_treated[cov], matched_control[cov])
        print(f"{cov}: before={smd_before:.3f}, after={smd_after:.3f}")

if __name__ == "__main__":
    estimate_ate_with_psm()
