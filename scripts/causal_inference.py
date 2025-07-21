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

MERGED_DATA_PATH = "../data/merged_users_cuped.csv"  # Use CUPED-adjusted file
OUTPUT_MATCHED_PATH = "../data/matched_users.csv"

def estimate_ate_with_psm(data_path=MERGED_DATA_PATH, output_path=OUTPUT_MATCHED_PATH):
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
    nbrs = NearestNeighbors(n_neighbors=1).fit(control[["propensity_score"]])
    distances, indices = nbrs.kneighbors(treated[["propensity_score"]])

    matched_control = control.iloc[indices.flatten()].reset_index(drop=True)
    matched_treated = treated.reset_index(drop=True)

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

if __name__ == "__main__":
    estimate_ate_with_psm()
