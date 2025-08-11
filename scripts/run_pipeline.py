"""
Causify Experimentation Pipeline

End-to-end workflow from raw data to dashboard-ready analytics.
Orchestrates preprocessing, feature engineering, statistical testing,
variance reduction, uplift modeling, and final data generation.
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
    """
    Execute the complete Causify analysis pipeline.
    
    Args:
        data_dir (str): Directory containing input data and for outputs
        n_neighbors (int): Number of neighbors for propensity score matching
        caliper (float): Maximum distance for PSM matching
    """
    data_path = Path(data_dir)
    print("\nStarting Causify pipeline execution...\n")
    
    # Validate required input files exist
    required_files = ["train_users_2.csv", "sessions.csv"]
    for file in required_files:
        if not (data_path / file).exists():
            raise FileNotFoundError(f"Required input file not found: {data_path / file}")
    
    print("Input validation passed")

    try:
        # Step 1: Data preprocessing
        print("\nStep 1: Preprocessing user data...")
        preprocess_airbnb_data(
            input_path=data_path / "train_users_2.csv",
            output_path=data_path / "clean_users.csv",
        )

        # Step 2: Session feature engineering
        print("\nStep 2: Generating session features...")
        generate_session_features(
            input_path=data_path / "sessions.csv",
            output_path=data_path / "user_session_features.csv",
        )

        # Step 3: Dataset merging
        print("\nStep 3: Merging user and session data...")
        merge_user_features(
            user_path=data_path / "clean_users.csv",
            session_path=data_path / "user_session_features.csv",
            output_path=data_path / "merged_users.csv",
        )

        # Step 4: Hypothesis testing
        print("\nStep 4: Running statistical tests...")
        test_results = run_z_test(data_path=data_path / "merged_users.csv")

        # Step 5: CUPED variance reduction
        print("\nStep 5: Applying CUPED adjustment...")
        auto_cuped(
            data_path=data_path / "merged_users.csv",
            output_path=data_path / "merged_users_cuped.csv",
        )

        # Step 6: Propensity score matching
        print("\nStep 6: Propensity score matching...")
        estimate_ate_with_psm(
            data_path=data_path / "merged_users_cuped.csv",
            output_path=data_path / "matched_users.csv",
            n_neighbors=n_neighbors,
            caliper=caliper,
        )

        # Step 7: Uplift modeling
        print("\nStep 7: Training uplift model...")
        run_uplift_model(
            data_path=data_path / "merged_users.csv",
            output_path=data_path / "uplift_scores.csv"
        )

        # Step 8: Dashboard data generation
        print("\nStep 8: Generating dashboard data...")
        generate_dashboard(data_dir)

        # Pipeline completion summary
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        print("Output files generated:")
        output_files = [
            "clean_users.csv", "user_session_features.csv", "merged_users.csv",
            "merged_users_cuped.csv", "matched_users.csv", "uplift_scores.csv",
            "final_dashboard_data.csv"
        ]
        for file in output_files:
            if (data_path / file).exists():
                print(f"  {file}")
            else:
                print(f"  {file} (MISSING)")
        
        print(f"\nKey Results:")
        if test_results:
            print(f"  Treatment effect: {test_results['effect_size']:.1%}")
            print(f"  Statistical significance: p = {test_results['p_value']:.4f}")
            print(f"  Effect size (Cohen's h): {test_results['cohens_h']:.3f}")
        
        print(f"\nNext Steps:")
        print(f"  Launch dashboard: cd dashboard && streamlit run app.py")
        print(f"  Review uplift scores: {data_path / 'uplift_scores.csv'}")
        
    except Exception as e:
        print(f"\nPIPELINE FAILED: {str(e)}")
        print("Check the error above and ensure all dependencies are installed.")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the complete Causify pipeline")
    parser.add_argument("--data_dir", default="../data", help="Directory for input/output data")
    parser.add_argument("--n_neighbors", type=int, default=5, help="Number of neighbors for PSM")
    parser.add_argument("--caliper", type=float, default=0.05, help="Caliper for propensity score matching")
    args = parser.parse_args()

    run_pipeline(args.data_dir, args.n_neighbors, args.caliper)
