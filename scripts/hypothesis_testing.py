"""
Statistical Hypothesis Testing

Performs two-proportion z-test to determine treatment significance
with confidence intervals and effect size calculations.
"""

import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
import numpy as np

# Configuration
DATA_PATH = "../data/merged_users.csv"


def run_z_test(data_path=DATA_PATH):
    """
    Perform two-proportion z-test to compare treatment and control booking rates.
    
    Args:
        data_path (str): Path to merged user data
        
    Returns:
        dict: Statistical test results including z-stat, p-value, effect size, CI
    """
    print("Loading data for hypothesis testing...")
    df = pd.read_csv(data_path)

    # Split by treatment assignment
    treatment_group = df[df["treatment"] == 1]
    control_group = df[df["treatment"] == 0]

    # Calculate booking rates
    treatment_rate = treatment_group["booking"].mean()
    control_rate = control_group["booking"].mean()
    
    print(f"Sample sizes - Treatment: {len(treatment_group):,}, Control: {len(control_group):,}")
    print(f"Treatment booking rate: {treatment_rate:.4f}")
    print(f"Control booking rate: {control_rate:.4f}")

    # Two-proportion z-test
    count = np.array([treatment_group["booking"].sum(), control_group["booking"].sum()])
    nobs = np.array([len(treatment_group), len(control_group)])
    stat, p_val = proportions_ztest(count, nobs)

    # Effect size and confidence interval calculations
    diff = treatment_rate - control_rate
    pooled_p = count.sum() / nobs.sum()
    
    # Use pooled SE for z-test statistic (already done by proportions_ztest)
    pooled_stderr = np.sqrt(pooled_p * (1 - pooled_p) * (1/nobs[0] + 1/nobs[1]))
    
    # Use unpooled SE for confidence interval (more appropriate)
    unpooled_stderr = np.sqrt((treatment_rate * (1 - treatment_rate) / nobs[0]) + 
                              (control_rate * (1 - control_rate) / nobs[1]))
    ci_low = diff - 1.96 * unpooled_stderr
    ci_high = diff + 1.96 * unpooled_stderr
    
    # Cohen's h effect size for proportions
    h = 2 * (np.arcsin(np.sqrt(treatment_rate)) - np.arcsin(np.sqrt(control_rate)))

    # Results summary
    print("\n" + "="*50)
    print("Statistical Test Results:")
    print("="*50)
    print(f"Z-statistic: {stat:.4f}")
    print(f"P-value: {p_val:.6f}")
    print(f"Difference in booking rate: {diff:.4f} ({diff:.1%})")
    print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Effect size (Cohen's h): {h:.4f}")
    print(f"Statistical power: {'High' if abs(stat) > 2.8 else 'Moderate' if abs(stat) > 1.96 else 'Low'}")

    # Significance interpretation
    if p_val < 0.001:
        print("*** Highly statistically significant difference (p < 0.001) ***")
    elif p_val < 0.01:
        print("** Statistically significant difference (p < 0.01) **")
    elif p_val < 0.05:
        print("* Statistically significant difference (p < 0.05) *")
    else:
        print("No statistically significant difference (p >= 0.05)")
    
    # Treatment balance check
    print(f"\nTreatment assignment balance: {len(treatment_group)/len(df):.1%} treated")
    
    return {
        'z_stat': stat,
        'p_value': p_val,
        'effect_size': diff,
        'ci_lower': ci_low,
        'ci_upper': ci_high,
        'cohens_h': h
    }


if __name__ == "__main__":
    run_z_test()
