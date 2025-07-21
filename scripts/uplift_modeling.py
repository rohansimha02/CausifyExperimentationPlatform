# scripts/uplift_modeling.py

"""
This script performs uplift modeling using the X-Learner to estimate
heterogeneous treatment effects (HTE) — i.e., which users benefit most from treatment.
"""

import pandas as pd
from causalml.inference.meta import BaseXRegressor
from causalml.propensity import compute_propensity_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


MERGED_DATA_PATH = "../data/merged_users.csv"


def run_uplift_model(data_path=MERGED_DATA_PATH):
    print("Loading data...")
    df = pd.read_csv(data_path)

    # Drop rows with missing covariates or outcome
    covariates = [
        "age", "total_actions", "unique_actions", "total_secs_elapsed"
    ]
    df = df.dropna(subset=covariates + ["booking"])

    # Prepare inputs
    X = df[covariates].values
    treatment = df["treatment"].values
    y = df["booking"].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compute propensity scores (optional but helps for XLearner)
    print("Estimating propensity scores...")
    ps_model = LogisticRegression()
    ps_model.fit(X_scaled, treatment)
    p = ps_model.predict_proba(X_scaled)[:, 1]


    # Train X-Learner
    print("Training BaseXRegressor uplift model...")
    x_learner = BaseXRegressor(
    learner=RandomForestClassifier(n_estimators=100),
    control_name=0,
)

    # Patch required attribute manually to avoid internal error
    x_learner.propensity_model = {
        0: ps_model,
        1: ps_model
    }

    x_learner.fit(X_scaled, treatment, y, p)


    # Estimate treatment effect per user
    te = x_learner.predict(X_scaled)
    df["uplift_score"] = te

    # Save output for dashboard or analysis
    df_out = df[covariates + ["treatment", "booking", "uplift_score"]]
    df_out.to_csv("../data/uplift_scores.csv", index=False)
    print("Uplift scores saved to ../data/uplift_scores.csv")

    # Show top and bottom 5 users by uplift score
    print("\nTop 5 users most likely to benefit from treatment:")
    print(df_out.sort_values("uplift_score", ascending=False).head())

    print("\nTop 5 users least likely to benefit (or harmed) by treatment:")
    print(df_out.sort_values("uplift_score", ascending=True).head())


if __name__ == "__main__":
    run_uplift_model()