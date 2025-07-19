# scripts/causal_inference.py

"""
This script estimates the Average Treatment Effect (ATE) using
Propensity Score Matching (PSM) to control for confounding variables.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

MERGED_DATA_PATH = "../data/merged_users.csv"


def estimate_ate_with_psm(data_path=MERGED_DATA_PATH):
    print("Loading data...")
    df = pd.read_csv(data_path)

    # Define covariates to balance treatment/control (pre-treatment only!)
    covariates = [
        "age", "total_actions", "unique_actions", "total_secs_elapsed"
    ]

    df = df.dropna(subset=covariates + ["booking"])

    # Scale features for matching
    scaler = StandardScaler()
    X = scaler.fit_transform(df[covariates])

    # Estimate propensity scores
    print("Estimating propensity scores...")
    model = LogisticRegression()
    model.fit(X, df["treatment"])
    df["propensity_score"] = model.predict_proba(X)[:, 1]

    # Split groups
    treated = df[df["treatment"] == 1].copy()
    control = df[df["treatment"] == 0].copy()

    # Nearest neighbor matching based on propensity score
    print("Matching treated and control units...")
    nbrs = NearestNeighbors(n_neighbors=1).fit(control[["propensity_score"]])
    distances, indices = nbrs.kneighbors(treated[["propensity_score"]])

    matched_control = control.iloc[indices.flatten()].reset_index(drop=True)
    matched_treated = treated.reset_index(drop=True)

    # Estimate ATE on matched pairs
    ate = (matched_treated["booking"].values - matched_control["booking"].values).mean()

    print("\n Propensity Score Matching Results:")
    print(f"Estimated ATE (matched pairs): {ate:.4f}")
    print(f"Matched sample size: {len(matched_treated)} pairs")


if __name__ == "__main__":
    estimate_ate_with_psm()