# scripts/generate_final_data.py
import pandas as pd

def main():
    # Load input files
    merged = pd.read_csv("../data/merged_users_cuped.csv")
    uplift = pd.read_csv("../data/uplift_scores.csv")

    # Merge by shared features (no ID in uplift)
    merge_cols = ["age", "total_actions", "unique_actions", "total_secs_elapsed", "treatment", "booking"]

    # Include both raw and clipped uplift scores
    uplift_features = merge_cols + ["uplift_score", "uplift_score_clipped"]
    uplift_subset = uplift[uplift_features]

    final = pd.merge(merged, uplift_subset, on=merge_cols, how="left")

    # Output to processed folder
    final.to_csv("../data/final_dashboard_data.csv", index=False)
    print("final_dashboard_data.csv generated successfully.")

if __name__ == "__main__":
    main()
