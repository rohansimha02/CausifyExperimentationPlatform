"""
Dashboard Data Generation

Combines all pipeline outputs into final dataset with statistical metrics,
confidence intervals, and model diagnostics for the dashboard.
"""

import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def calculate_statistical_tests(df):
    """Calculate comprehensive statistical test results for treatment vs control."""
    treatment_group = df[df["treatment"] == 1]
    control_group = df[df["treatment"] == 0]
    
    # Basic conversion rates
    treatment_rate = treatment_group["booking"].mean()
    control_rate = control_group["booking"].mean()
    
    # Two-proportion z-test for significance
    count = np.array([treatment_group["booking"].sum(), control_group["booking"].sum()])
    nobs = np.array([len(treatment_group), len(control_group)])
    stat, p_val = proportions_ztest(count, nobs)
    
    # Effect size calculations
    diff = treatment_rate - control_rate
    pooled_p = count.sum() / nobs.sum()
    stderr = np.sqrt(pooled_p * (1 - pooled_p) * (1/nobs[0] + 1/nobs[1]))
    ci_low = diff - 1.96 * stderr
    ci_high = diff + 1.96 * stderr
    
    # Cohen's h effect size for proportions
    h = 2 * (np.arcsin(np.sqrt(treatment_rate)) - np.arcsin(np.sqrt(control_rate)))
    
    return {
        'treatment_rate': treatment_rate,
        'control_rate': control_rate,
        'z_stat': stat,
        'p_value': p_val,
        'effect_size': diff,
        'ci_lower': ci_low,
        'ci_upper': ci_high,
        'cohens_h': h,
        'stderr': stderr
    }


def calculate_variance_reduction(df_raw, df_cuped):
    """Calculate CUPED variance reduction metrics."""
    original_var = df_raw['booking'].var()
    cuped_var = df_cuped['booking_cuped'].var()
    variance_reduction = ((original_var - cuped_var) / original_var) * 100
    
    return {
        'original_variance': original_var,
        'cuped_variance': cuped_var,
        'variance_reduction_pct': variance_reduction
    }


def add_propensity_scores(df):
    """Add propensity scores for randomization balance assessment."""
    covariates = ["age", "total_actions", "unique_actions", "total_secs_elapsed"]
    
    # Prepare clean data for modeling
    df_clean = df.dropna(subset=covariates + ["treatment"])
    X = df_clean[covariates].values
    treatment = df_clean["treatment"].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit propensity score model
    ps_model = LogisticRegression(random_state=42, max_iter=1000)
    ps_model.fit(X_scaled, treatment)
    
    # Predict propensity scores for all users
    propensity_scores = np.full(len(df), np.nan)
    clean_indices = df_clean.index
    propensity_scores[clean_indices] = ps_model.predict_proba(X_scaled)[:, 1]
    
    df['propensity_score'] = propensity_scores
    return df


def add_confidence_intervals(df, stats_results):
    """Add confidence intervals for booking rates by treatment group."""
    # Calculate standard errors by group
    treatment_se = np.sqrt(stats_results['treatment_rate'] * (1 - stats_results['treatment_rate']) / (df['treatment'] == 1).sum())
    control_se = np.sqrt(stats_results['control_rate'] * (1 - stats_results['control_rate']) / (df['treatment'] == 0).sum())
    
    # Apply confidence intervals based on group membership
    df['booking_rate_ci_lower'] = np.where(
        df['treatment'] == 1,
        stats_results['treatment_rate'] - 1.96 * treatment_se,
        stats_results['control_rate'] - 1.96 * control_se
    )
    
    df['booking_rate_ci_upper'] = np.where(
        df['treatment'] == 1,
        stats_results['treatment_rate'] + 1.96 * treatment_se,
        stats_results['control_rate'] + 1.96 * control_se
    )
    
    return df


def add_segment_statistics(df):
    """Add user segmentation for dashboard filtering and analysis."""
    # Age quintiles for demographic analysis
    df['age_quintile'] = pd.qcut(df['age'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    
    # Activity level segmentation
    df['activity_level'] = pd.cut(df['total_actions'], 
                                 bins=[0, 50, 150, 300, float('inf')], 
                                 labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Engagement quality segmentation
    df['engagement_level'] = pd.cut(df['engagement_ratio'], 
                                   bins=[0, 0.3, 0.7, 1.0], 
                                   labels=['Low', 'Medium', 'High'])
    
    return df


def add_model_diagnostics(df):
    """Add uplift model performance and diagnostic metrics."""
    # Uplift score deciles for validation analysis
    df['uplift_decile'] = pd.qcut(df['uplift_score'], 10, labels=False) + 1
    
    # High-value user indicator
    df['high_uplift'] = (df['uplift_score'] > df['uplift_score'].quantile(0.75)).astype(int)
    
    # Calculate uplift model performance metric
    uplift_performance = calculate_uplift_model_performance(df)
    df['uplift_model_performance'] = uplift_performance
    
    return df


def calculate_uplift_model_performance(df):
    """
    Calculate uplift model performance using ranking correlation.
    Measures how well uplift scores rank users by their actual treatment response.
    
    Returns:
        float: Performance score between 0.5 (random) and 1.0 (perfect)
    """
    # Split by treatment assignment
    treated = df[df['treatment'] == 1].copy()
    control = df[df['treatment'] == 0].copy()
    
    if len(treated) == 0 or len(control) == 0:
        return 0.5  # Random performance baseline
    
    # Calculate actual uplift by predicted uplift decile
    decile_performance = []
    
    for decile in range(1, 11):
        treated_decile = treated[treated['uplift_decile'] == decile]
        control_decile = control[control['uplift_decile'] == decile]
        
        if len(treated_decile) > 0 and len(control_decile) > 0:
            # Actual uplift in this decile
            treated_rate = treated_decile['booking'].mean()
            control_rate = control_decile['booking'].mean()
            actual_uplift = treated_rate - control_rate
            decile_performance.append((decile, actual_uplift))
    
    if len(decile_performance) < 3:  # Minimum for meaningful correlation
        return 0.5
    
    # Calculate Spearman correlation between rank and actual uplift
    deciles, actual_uplifts = zip(*decile_performance)
    
    try:
        from scipy.stats import spearmanr
        correlation, _ = spearmanr(deciles, actual_uplifts)
    except ImportError:
        # Fallback to Pearson correlation
        correlation = np.corrcoef(deciles, actual_uplifts)[0, 1]
    
    # Normalize correlation to 0.5-1.0 scale (0.5=random, 1.0=perfect)
    performance_score = 0.5 + (correlation * 0.5) if not np.isnan(correlation) else 0.5
    
    return max(0.0, min(1.0, performance_score))


def main(data_dir="../data"):
    """Generate comprehensive dashboard dataset with all analytics."""
    print("Generating enhanced dashboard data...")
    
    # Load all pipeline outputs
    print("Loading input datasets...")
    merged = pd.read_csv(f"{data_dir}/merged_users.csv")
    merged_cuped = pd.read_csv(f"{data_dir}/merged_users_cuped.csv")
    uplift = pd.read_csv(f"{data_dir}/uplift_scores.csv")
    
    # Merge uplift scores with CUPED data
    print("Merging uplift scores...")
    merge_cols = ["age", "total_actions", "unique_actions", "total_secs_elapsed", "treatment", "booking"]
    uplift_features = merge_cols + ["uplift_score", "uplift_score_clipped"]
    uplift_subset = uplift[uplift_features]
    
    final = pd.merge(merged_cuped, uplift_subset, on=merge_cols, how="left")
    
    # Calculate comprehensive analytics
    print("Calculating statistical test results...")
    stats_results = calculate_statistical_tests(final)
    
    print("Calculating CUPED variance reduction...")
    variance_results = calculate_variance_reduction(merged, merged_cuped)
    
    print("Adding propensity scores...")
    final = add_propensity_scores(final)
    
    print("Adding confidence intervals...")
    final = add_confidence_intervals(final, stats_results)
    
    print("Adding user segments...")
    final = add_segment_statistics(final)
    
    print("Adding model diagnostics...")
    final = add_model_diagnostics(final)
    
    # Add global metrics as metadata columns
    print("Adding global statistics...")
    for key, value in stats_results.items():
        final[f'global_{key}'] = value
    
    for key, value in variance_results.items():
        final[f'global_{key}'] = value
    
    # Save final dashboard dataset
    output_path = f"{data_dir}/final_dashboard_data.csv"
    final.to_csv(output_path, index=False)
    
    # Generate summary report
    print("\n" + "="*60)
    print("DASHBOARD DATA GENERATION COMPLETE")
    print("="*60)
    print(f"Output file: {output_path}")
    print(f"Total records: {len(final):,}")
    print(f"Treatment effect: {stats_results['effect_size']:.4f} ({stats_results['effect_size']:.1%})")
    print(f"P-value: {stats_results['p_value']:.2e}")
    print(f"Variance reduction: {variance_results['variance_reduction_pct']:.1f}%")
    print(f"Total features: {final.shape[1]} columns")
    
    print("\nNew Features Added:")
    new_features = [
        'propensity_score', 'booking_rate_ci_lower', 'booking_rate_ci_upper',
        'age_quintile', 'activity_level', 'engagement_level', 
        'uplift_decile', 'high_uplift', 'uplift_model_performance'
    ]
    for feature in new_features:
        if feature in final.columns:
            print(f"  {feature}")
    
    print("\nDashboard dataset ready for deployment!")


if __name__ == "__main__":
    main()
