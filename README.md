# Causify Experimentation Platform

Causify is an end-to-end sample platform for product experimentation. It showcases how causal inference and uplift modeling can reveal the true impact of feature launches on bookings. By quantifying incremental lift, growth teams can prioritize changes that move the needle and iterate faster. The repository spans data preparation, variance reduction with CUPED, matching-based estimation, uplift modeling and a Streamlit dashboard. Although the examples rely on public Airbnb data, the same approach can be applied to any product with user-level logs.

You can explore the dashboard at:

**[Dashboard URL goes here](#)**

## Repository Structure

- `scripts/` – Individual python modules that run each stage of the workflow
  - `preprocess_data.py` – cleans the raw Airbnb Kaggle data and simulates a treatment assignment
  - `session_features.py` – aggregates session logs into per-user features
  - `merge_features.py` – merges user attributes with aggregated session features and injects the treatment/outcome columns
  - `booking_cuped.py` – automatically chooses covariates and applies CUPED
  - `hypothesis_testing.py` – performs a simple z‑test on the booking rate
  - `causal_inference.py` – estimates the ATE via propensity score matching
  - `uplift_modeling.py` – trains an X‑Learner to score users by predicted uplift
  - `generate_dashboard_data.py` – joins model outputs for the dashboard
  - `run_pipeline.py` – convenience script to run every stage end‑to‑end
- `dashboard/` – Streamlit app that visualizes uplift scores and other metrics

## Getting Started

1. Obtain the Airbnb Kaggle datasets (`train_users_2.csv` and `sessions.csv`) and place them in a `data/` directory at the repository root.
2. Install dependencies (Python 3.10+ recommended):
   ```bash
   pip install pandas numpy scikit-learn causalml statsmodels plotly streamlit
   ```
3. Run the full pipeline:
   ```bash
   python scripts/run_pipeline.py --data_dir ./data
   ```
   This will generate processed CSV files in the `data/` folder, train an uplift model and prepare `final_dashboard_data.csv` for the dashboard.
4. Launch the dashboard:
   ```bash
   streamlit run dashboard/app.py
   ```

## Streamlit Dashboard

The Streamlit app visualizes uplift scores and other metrics produced by the pipeline. Run `streamlit run dashboard/app.py` to launch it locally. If you deploy the app externally, update the dashboard link near the top of this README.

## License

This project is provided for educational purposes only. It uses a synthetic workflow derived from the public Airbnb Kaggle dataset and carries no official affiliation with Airbnb.
