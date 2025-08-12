# Causify Experimentation Platform

An end-to-end platform for product experimentation demonstrating causal inference and uplift modeling on Airbnb booking data. This project shows how growth teams can quantify incremental lift and make data-driven decisions about feature rollouts using advanced statistical methods and machine learning.

The platform implements industry-standard experimental design principles including randomized controlled trials, variance reduction techniques, and heterogeneous treatment effect estimation. Built for data scientists and product managers who need to move beyond simple A/B testing to understand which users benefit most from new features.

**Explore the live dashboard:** [https://causifyexperimentationplatform.streamlit.app/](https://causifyexperimentationplatform.streamlit.app/)

## What You'll Find

The **live dashboard** demonstrates a complete experimental analysis workflow:

- **Overview** - Statistical test results, confidence intervals, and business impact estimates with dynamic filtering
- **Uplift Analysis** - User-level treatment effect predictions, model calibration diagnostics, and segment identification  
- **Targeting Strategy** - ROI calculators for optimized feature rollouts with scenario planning and sensitivity analysis
- **Model Validation** - Comprehensive randomization checks, balance diagnostics, and model performance metrics

All visualizations are interactive with real-time filtering by user characteristics. The analysis uses a simulated experiment comparing a new booking interface against the current one, with realistic user behavior patterns derived from actual Airbnb data.

Perfect for understanding modern experimentation techniques without needing to set up the infrastructure yourself.

## Technical Implementation

This platform demonstrates advanced experimentation techniques including:

**Statistical Methods**
- CUPED variance reduction using pre-treatment covariates
- Propensity score matching for treatment effect estimation
- Two-sample z-tests with confidence intervals and effect size calculation
- Randomization diagnostics and balance checks

**Machine Learning**
- X-Learner uplift modeling for heterogeneous treatment effects
- Random Forest base learners with cross-validation
- Model calibration and performance diagnostics
- Feature engineering from behavioral and demographic data

**Dashboard Technology**
- Built with Streamlit and Plotly for interactive exploration
- Real-time filtering and scenario planning capabilities
- Professional visualizations suitable for executive presentation
- Responsive design that works across devices

The complete pipeline processes user behavior data, simulates realistic experimental outcomes, and provides actionable insights for product teams.

## Repository Structure

This repository contains the complete pipeline for advanced experimentation analysis:

**`scripts/`** – End-to-end analysis pipeline
- Data preprocessing and feature engineering from user behavior logs
- Treatment assignment simulation with realistic randomization
- CUPED variance reduction and statistical testing
- Uplift modeling with X-Learner implementation
- Model validation and diagnostic checks

**`dashboard/`** – Interactive Streamlit application
- Production-ready interface with professional visualizations
- Real-time filtering and scenario planning capabilities
- Executive summary with key metrics and business recommendations
- Comprehensive model diagnostics and validation results

The pipeline demonstrates industry best practices for experimental design, causal inference, and machine learning in product experimentation.

## Using This Repository

**For Exploring Results:** Visit the [live dashboard](https://causifyexperimentationplatform.streamlit.app/) to interact with the analysis without any setup required.

**For Learning & Adaptation:** The complete source code is available for educational purposes and can be adapted for your own experimentation needs:

1. **Clone the repository** to explore the methodology and implementation details
2. **Review the scripts** to understand the statistical techniques and modeling approaches  
3. **Adapt the pipeline** for your own datasets by modifying the preprocessing and feature engineering steps
4. **Deploy your own version** using the included Streamlit application as a template

The methodology can be applied to any product experiment where you want to understand heterogeneous treatment effects and optimize rollout strategies.

## Running Locally (Optional)

If you want to run the pipeline yourself or modify the analysis:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline (requires Airbnb datasets)
python scripts/run_pipeline.py --data_dir ./data

# Launch the dashboard locally
streamlit run dashboard/app.py
```

The pipeline expects Airbnb datasets from Kaggle but can be adapted for other user behavior datasets.

## License

Educational and research use only. This project demonstrates advanced experimentation techniques using publicly available Airbnb data with no official affiliation or endorsement. 

The methodologies and code can be adapted for commercial use cases, but users should ensure compliance with their organization's data policies and ethical guidelines for experimentation.
