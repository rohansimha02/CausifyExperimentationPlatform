"""
Uplift Modeling with X-Learner

Estimates heterogeneous treatment effects using X-Learner
to identify users who benefit most from treatment.
"""

import pandas as pd
from causalml.inference.meta import BaseXRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Configuration
MERGED_DATA_PATH = "../data/merged_users.csv"


def run_uplift_model(data_path=MERGED_DATA_PATH, output_path=None):
    """
    Train X-Learner uplift model to estimate individual treatment effects.
    
    Args:
        data_path (str): Path to merged user data
        output_path (str): Path for uplift scores output (auto-generated if None)
    """
    print("Loading data for uplift modeling...")
    df = pd.read_csv(data_path)
    
    if output_path is None:
        from pathlib import Path
        data_dir = Path(data_path).parent
        output_path = data_dir / "uplift_scores.csv"

    covariates = ["age", "total_actions", "unique_actions", "total_secs_elapsed"]
    df = df.dropna(subset=covariates + ["booking"])

    X = df[covariates].values
    treatment = df["treatment"].values
    y = df["booking"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Stratified train/test split
    X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(
        X_scaled, treatment, y, test_size=0.2, random_state=42, stratify=treatment
    )

    # Estimate propensity scores for X-Learner
    print("Estimating propensity scores...")
    ps_model = LogisticRegression(random_state=42, max_iter=1000)
    ps_model.fit(X_train, t_train)
    p_train = ps_model.predict_proba(X_train)[:, 1]
    p_test = ps_model.predict_proba(X_test)[:, 1]
    
    # Validate propensity score overlap
    print(f"Propensity score overlap - Treated: [{p_train[t_train==1].min():.3f}, {p_train[t_train==1].max():.3f}]")
    print(f"Propensity score overlap - Control: [{p_train[t_train==0].min():.3f}, {p_train[t_train==0].max():.3f}]")

    # Train X-Learner uplift model
    print("Training X-Learner uplift model...")
    x_learner = BaseXRegressor(
        learner=RandomForestRegressor(
            n_estimators=100, max_depth=6, random_state=42, min_samples_leaf=10
        ),
        control_name=0
    )

    # Configure propensity model
    x_learner.propensity_model = {0: ps_model, 1: ps_model}

    # Fit uplift model
    x_learner.fit(X_train, t_train, y_train, p_train)

    # Generate treatment effect predictions
    te_train = x_learner.predict(X_train).flatten()
    te_test = x_learner.predict(X_test).flatten()

    # Model validation metrics
    # NOTE: For uplift models, we validate by comparing model ATE to observed ATE
    ate_model = np.mean(np.concatenate([te_train, te_test]))
    ate_observed = y[treatment==1].mean() - y[treatment==0].mean()
    
    # Calculate baseline conversion rate for validation
    control_baseline = y[treatment==0].mean()
    
    # For treated users in test set, predict their outcome
    treated_test_mask = t_test == 1
    if treated_test_mask.any():
        y_pred_treated = te_test[treated_test_mask] + control_baseline
        y_actual_treated = y_test[treated_test_mask]
        rmse_treated = np.sqrt(mean_squared_error(y_actual_treated, y_pred_treated))
        print(f"Test RMSE (treated outcome prediction): {rmse_treated:.4f}")
    
    print(f"Model ATE: {ate_model:.4f}")
    print(f"Observed ATE: {ate_observed:.4f}")
    print(f"ATE difference: {abs(ate_model - ate_observed):.4f}")

    # Create complete uplift score array
    te_full = np.zeros(len(df))
    n_train = len(te_train)
    n_test = len(te_test)
    
    te_full[:n_train] = te_train
    te_full[n_train:n_train+n_test] = te_test
    
    # Predict for any remaining observations
    if n_train + n_test < len(df):
        remaining_idx = np.arange(n_train + n_test, len(df))
        X_remaining = X_scaled[remaining_idx]
        te_remaining = x_learner.predict(X_remaining).flatten()
        te_full[remaining_idx] = te_remaining

    # Generate uplift scores with clipped version for robustness
    df["uplift_score"] = te_full
    df["uplift_score_clipped"] = df["uplift_score"].clip(-0.2, 0.2)

    # Save uplift scores for dashboard
    df_out = df[covariates + ["treatment", "booking", "uplift_score", "uplift_score_clipped"]]
    df_out.to_csv(output_path, index=False)
    print(f"Uplift scores saved to {output_path}")

    # Model performance summary
    print(f"\nModel Performance Summary:")
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Uplift score range: [{df['uplift_score'].min():.3f}, {df['uplift_score'].max():.3f}]")
    print(f"Uplift score std: {df['uplift_score'].std():.3f}")

    print("\nTop 5 users most likely to benefit from treatment:")
    print(df_out.sort_values("uplift_score", ascending=False).head())

    print("\nTop 5 users least likely to benefit from treatment:")
    print(df_out.sort_values("uplift_score", ascending=True).head())


if __name__ == "__main__":
    run_uplift_model()
