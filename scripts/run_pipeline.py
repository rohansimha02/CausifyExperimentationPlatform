# scripts/run_pipeline.py

"""
End-to-end pipeline runner for the Causify project.
This script executes the full experimental workflow:
1. Data preprocessing
2. Session feature aggregation
3. Merging user + session data
4. Hypothesis testing
5. CUPED adjustment
6. Propensity Score Matching
7. Uplift modeling
"""

import argparse
from pathlib import Path

from preprocess_data import preprocess_airbnb_data
from session_features import generate_session_features
from merge_features import merge_user_features
from hypothesis_testing import run_z_test
from booking_cuped import auto_cuped
from causal_inference import estimate_ate_with_psm
from uplift_modeling import run_uplift_model
from generate_dashboard_data import main as generate_dashboard

def run_pipeline(data_dir: str, n_neighbors: int = 5, caliper: float = 0.05):
    data_path = Path(data_dir)
    print("\nStarting full Causify pipeline...\n")

    preprocess_airbnb_data(
        input_path=data_path / "train_users_2.csv",
        output_path=data_path / "clean_users.csv",
    )

    generate_session_features(
        input_path=data_path / "sessions.csv",
        output_path=data_path / "user_session_features.csv",
    )

    merge_user_features(
        user_path=data_path / "clean_users.csv",
        session_path=data_path / "user_session_features.csv",
        output_path=data_path / "merged_users.csv",
    )

    run_z_test(data_path=data_path / "merged_users.csv")

    auto_cuped(
        data_path=data_path / "merged_users.csv",
        output_path=data_path / "merged_users_cuped.csv",
    )

    estimate_ate_with_psm(
        data_path=data_path / "merged_users_cuped.csv",
        output_path=data_path / "matched_users.csv",
        n_neighbors=n_neighbors,
        caliper=caliper,
    )

    run_uplift_model(data_path=data_path / "merged_users.csv")

    generate_dashboard()

    print("All stages complete. Uplift scores available at /data/uplift_scores.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full Causify pipeline")
    parser.add_argument("--data_dir", default="../data", help="Directory of input/output data")
    parser.add_argument("--n_neighbors", type=int, default=5, help="Number of neighbors for PSM")
    parser.add_argument("--caliper", type=float, default=0.05, help="Caliper for propensity score matching")
    args = parser.parse_args()

    run_pipeline(args.data_dir, args.n_neighbors, args.caliper)
