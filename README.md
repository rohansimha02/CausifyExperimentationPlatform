# Causify Experimentation Platform

An end-to-end platform for product experimentation demonstrating causal inference and uplift modeling on Airbnb booking data. Shows how growth teams can quantify incremental lift and make data-driven decisions about feature rollouts.

You can explore the interactive dashboard at: **[Dashboard URL goes here]**

## Features

- **Executive Dashboard** - Business-focused visualizations and insights
- **Causal Inference** - CUPED variance reduction and propensity score matching  
- **Uplift Modeling** - X-Learner to identify high-impact user segments
- **Targeting Strategy** - ROI optimization tools for smart rollouts

## Repository Structure

**`scripts/`** – Core analysis pipeline
- `preprocess_data.py` – data cleaning and treatment assignment
- `session_features.py` – behavioral feature engineering  
- `merge_features.py` – dataset integration with outcome simulation
- `booking_cuped.py` – variance reduction via CUPED
- `hypothesis_testing.py` – statistical significance testing
- `causal_inference.py` – propensity score matching for ATE estimation
- `uplift_modeling.py` – X-Learner for heterogeneous treatment effects
- `generate_dashboard_data.py` – final dataset preparation
- `run_pipeline.py` – end-to-end pipeline execution

**`dashboard/`** – Streamlit web application
- Executive summary with key metrics and statistical tests
- Uplift analysis showing individual user predictions
- Targeting simulator for rollout strategy optimization
- Model validation and diagnostic checks

## Getting Started

1. **Data Setup** - Download Airbnb datasets (`train_users_2.csv` and `sessions.csv`) from Kaggle and place in `data/` directory

2. **Installation**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Pipeline**:
   ```bash
   python scripts/run_pipeline.py --data_dir ./data
   ```

4. **Launch Dashboard**:
   ```bash
   streamlit run dashboard/app.py
   ```

## Dashboard

The Streamlit application provides an interactive interface for exploring experimental results:

- **Overview** - Statistical test results and business impact estimates
- **Uplift Analysis** - User-level treatment effect predictions and model calibration  
- **Targeting** - ROI calculators for optimized feature rollouts
- **Validation** - Randomization checks and model diagnostics

Supports both local data files and CSV uploads for flexible deployment.

## Technical Stack

Python pipeline using pandas, scikit-learn, CausalML, and statsmodels for statistical analysis. Streamlit dashboard with Plotly visualizations for interactive exploration.

## License

Educational use only. Uses public Airbnb dataset with no official affiliation.
