"""
Causal Inference with Propensity Score Matching

Estimates Average Treatment Effect using propensity score matching
with CUPED-adjusted outcomes and balance diagnostics.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import os

# Configuration
MERGED_DATA_PATH = "../data/merged_users_cuped.csv" 
OUTPUT_MATCHED_PATH = "../data/matched_users.csv"


def standardized_mean_diff(x_treat, x_ctrl):
    """Calculate standardized mean difference for covariate balance assessment."""
    return (x_treat.mean() - x_ctrl.mean()) / np.sqrt(0.5 * (x_treat.var() + x_ctrl.var()))


def estimate_ate_with_psm(
    data_path=MERGED_DATA_PATH,
    output_path=OUTPUT_MATCHED_PATH,
    n_neighbors=5,
    caliper=0.05,
):
    """
    Estimate Average Treatment Effect using Propensity Score Matching.
    
    Args:
        data_path (str): Path to CUPED-adjusted data
        output_path (str): Path for matched dataset output
        n_neighbors (int): Number of nearest neighbors to consider
        caliper (float): Maximum propensity score distance for matching
    """
    print("Loading CUPED-adjusted data...")
    df = pd.read_csv(data_path)

    # Select pre-treatment covariates for matching
    covariates = ["age", "unique_actions", "total_actions", "total_secs_elapsed"]
    df = df.dropna(subset=covariates + ["booking_cuped"])

    # Standardize covariates for propensity score estimation
    scaler = StandardScaler()
    X = scaler.fit_transform(df[covariates])

    # Estimate propensity scores using logistic regression
    print("Estimating propensity scores...")
    model = LogisticRegression()
    model.fit(X, df["treatment"])
    df["propensity_score"] = model.predict_proba(X)[:, 1]

    # Separate treatment and control groups
    treated = df[df["treatment"] == 1].copy()
    control = df[df["treatment"] == 0].copy()

    print("Performing nearest neighbor matching...")
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(control[["propensity_score"]])
    distances, indices = nbrs.kneighbors(treated[["propensity_score"]])

    # Apply caliper constraint for match quality
    matched_rows = []
    for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
        # Find the best match within caliper
        valid_matches = dist_row <= caliper
        if not valid_matches.any():
            continue  # No valid matches within caliper
        
        # Select the closest match within caliper
        best_match_idx = np.where(valid_matches)[0][0]  # First (closest) valid match
        chosen = idx_row[best_match_idx]
        matched_rows.append((i, chosen))

    # Create matched samples
    matched_treated = treated.iloc[[i for i, _ in matched_rows]].reset_index(drop=True)
    matched_control = control.iloc[[j for _, j in matched_rows]].reset_index(drop=True)

    # Calculate ATE on CUPED-adjusted outcome
    y_treat = matched_treated["booking_cuped"].values
    y_control = matched_control["booking_cuped"].values
    ate = (y_treat - y_control).mean()

    print("\nPropensity Score Matching Results:")
    print(f"Estimated ATE: {ate:.4f}")
    print(f"Matched sample size: {len(matched_treated)} pairs")

    # Save matched dataset for further analysis
    matched_treated["group"] = "treated"
    matched_control["group"] = "control"
    matched_df = pd.concat([matched_treated, matched_control], axis=0)
    matched_df.to_csv(output_path, index=False)
    print(f"Matched dataset saved to {output_path}")

    # Balance diagnostics: check covariate balance improvement
    print("\nCovariate Balance (Standardized Mean Differences):")
    for cov in covariates:
        smd_before = standardized_mean_diff(treated[cov], control[cov])
        smd_after = standardized_mean_diff(matched_treated[cov], matched_control[cov])
        print(f"{cov}: before={smd_before:.3f}, after={smd_after:.3f}")


if __name__ == "__main__":
    estimate_ate_with_psm()
