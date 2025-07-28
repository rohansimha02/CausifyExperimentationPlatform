# ==========================================
# Causify Experimentation Platform
# A causal inference and uplift modeling system for A/B testing workflows.
# ==========================================
# This Streamlit app visualizes treatment effects from a simulated A/B test on Airbnb session data.
# It uses CUPED (for variance reduction), hypothesis testing, and uplift modeling (X-Learner) to estimate
# how different user segments respond to a new booking experience.
# The dataset includes session-level behavioral aggregates, treatment assignment, outcomes, and causal scores.
# ==========================================


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
   df = pd.read_csv("./data/final_dashboard_data.csv")
   df["Treatment Label"] = df["treatment"].map({0: "Control", 1: "Treated"})
   return df


df = load_data()


def calculate_relative_impact(df, deployment_pct):
   """Calculate relative impact without assuming dollar values"""
  
   # Sort users by uplift score (descending)
   df_sorted = df.sort_values('uplift_score', ascending=False)
  
   # Select top X% of users
   n_deploy = int(len(df_sorted) * (deployment_pct / 100))
   selected_users = df_sorted.head(n_deploy)
  
   # Calculate expected additional bookings
   additional_bookings = selected_users['uplift_score'].sum()
  
   # Baseline bookings from these users
   baseline_bookings = selected_users['booking'].sum()
  
   # Relative improvement
   if baseline_bookings > 0:
       relative_improvement = (additional_bookings / baseline_bookings) * 100
   else:
       relative_improvement = 0
  
   return {
       'additional_bookings': additional_bookings,
       'baseline_bookings': baseline_bookings,
       'relative_improvement': relative_improvement,
       'users_targeted': n_deploy,
       'avg_uplift': selected_users['uplift_score'].mean()
   }
# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(page_title="Causify Experimentation Platform", layout="wide")


# ----------------------------
# Sidebar: Global Filters
# ----------------------------
with st.sidebar:
   st.title("🌐 Dashboard Filters")
   st.markdown("Use these global filters to narrow down the cohort being analyzed.")
   treatment_group = st.radio("Filter by Treatment Group", ["All", "Treated", "Control"])
   segment_feature = st.selectbox("Segment Feature (for HTE)", ["age", "total_actions", "unique_actions", "total_secs_elapsed"])
   age_range = st.slider("Filter by Age", int(df["age"].min()), int(df["age"].max()), (18, 80))
   action_range = st.slider("Filter by Total Actions", int(df["total_actions"].min()), int(df["total_actions"].max()), (0, 500))
   st.markdown("---")
   st.caption("Powered by Streamlit · Plotly · CausalML")


# ----------------------------
# Apply Filters
# ----------------------------
if treatment_group == "Treated":
   df = df[df["treatment"] == 1]
elif treatment_group == "Control":
   df = df[df["treatment"] == 0]


df = df[
   df["age"].between(age_range[0], age_range[1]) &
   df["total_actions"].between(action_range[0], action_range[1])
]


# ----------------------------
# Dashboard Intro
# ----------------------------
st.title("📈 Causify Experimentation Platform")
st.markdown("##### Welcome to **Causify**, a simulated dashboard for analyzing treatment effects from an Airbnb A/B test.")
st.markdown("##### This app showcases **how causal inference and uplift modeling can help product teams make smarter, targeted rollout decisions**.")


st.markdown("""


### 🧪 Experiment Setup
- **Treatment**: New booking UI tested on 50% of users.
- **Outcome**: Whether the user booked a stay (binary outcome).
- **Features**: Session-level aggregates derived from Airbnb logs (e.g., time spent, number of actions).
- **Techniques Used**:
 - **CUPED**: Variance reduction technique to improve experiment precision
 - **Hypothesis Testing**: Classical t-tests to confirm significance
 - **Uplift Modeling**: X-Learner estimates the **Individual Treatment Effect (ITE)** for each user
- **Pipeline**: Data preprocessing → CUPED → t-tests → X-Learner


This simulation mirrors how data science teams at product-oriented companies use experimentation to guide feature rollout.
""")


# ----------------------------
# Tabs Setup
# ----------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
   "Overview", "Uplift Distribution", "Segment Analysis", "User Rankings",
   "Final Summary", "CUPED Adjustment", "Glossary"
])


# ----------------------------
# Tab 1: Overview
# ----------------------------
with tab1:
   st.subheader("📋 Key Experiment Metrics")


   col1, col2, col3 = st.columns(3)
   col1.metric("Total Users", f"{len(df):,}", "Filtered sample size")
   col2.metric("Treated Users", f"{(df['treatment'] == 1).sum():,}")
   col3.metric("Control Users", f"{(df['treatment'] == 0).sum():,}")


   col4, col5, col6 = st.columns(3)
   col4.metric("Booking Rate (Treated)", f"{df[df['treatment'] == 1]['booking'].mean():.2%}")
   col5.metric("Booking Rate (Control)", f"{df[df['treatment'] == 0]['booking'].mean():.2%}")
   col6.metric("Avg Uplift Score", f"{df['uplift_score'].mean():.4f}")


   st.success("Estimated uplift: ~10% increase in bookings. Modeled and validated via X-Learner and statistical tests.")


   st.markdown("This provides the first high-level indication that the treatment is effective.")


   st.divider()
   st.subheader("📊 Booking Rate by Group")
   st.markdown("Shows raw outcome differences between groups (before CUPED adjustment).")


   fig = px.bar(
       df.groupby("Treatment Label")["booking"].mean().reset_index(),
       x="Treatment Label", y="booking",
       labels={"booking": "Booking Rate"},
       color="Treatment Label",
       color_discrete_sequence=["#A7C7E7", "#5DAF83"]
   )
   st.plotly_chart(fig, use_container_width=True)


# ----------------------------
# Tab 2: Uplift Distribution
# ----------------------------
with tab2:
   st.subheader("🔁 Distribution of Uplift Scores")
   st.markdown("""
Each user's uplift score is the model's **estimated effect of receiving the treatment**.
In other words: “If we show this user the new UI, how much more likely are they to book?”


This is calculated using an **X-Learner uplift model**, which separately models outcomes under treatment and control, and estimates the difference.
""")


   dist_fig = px.histogram(df, x="uplift_score_clipped", nbins=50, marginal="rug", color_discrete_sequence=["#F92E2E"])
   dist_fig.update_layout(title="Distribution of Estimated Treatment Effects")
   st.plotly_chart(dist_fig, use_container_width=True)


   st.info("Scores near 0 = neutral. Positive = likely to benefit. Negative = likely harmed.")


   st.subheader("📊 Heatmap: Uplift by Age × Engagement")
   st.markdown("Explore heterogeneity by age and activity level (total actions).")


   df_heatmap = df.copy()
   df_heatmap["age_bin"] = pd.cut(df_heatmap["age"], bins=5)
   df_heatmap["action_bin"] = pd.cut(df_heatmap["total_actions"], bins=5)


   pivot = df_heatmap.pivot_table(index="age_bin", columns="action_bin", values="uplift_score_clipped", aggfunc="mean")


   heatmap = go.Figure(data=go.Heatmap(
       z=pivot.values,
       x=[str(c) for c in pivot.columns],
       y=[str(i) for i in pivot.index],
       colorscale="RdBu",
       colorbar_title="Avg Uplift"
   ))
   heatmap.update_layout(title="Uplift by Age × Total Actions")
   st.plotly_chart(heatmap, use_container_width=True)


# ----------------------------
# Tab 3: Segment Analysis
# ----------------------------
with tab3:
   st.subheader(f"📈 Uplift by {segment_feature.title()}")
   st.markdown("""
This scatterplot shows how predicted uplift varies along a chosen feature (e.g. age or activity).
We apply **LOWESS smoothing** to show local trends in heterogeneity of treatment effect (HTE).
""")


   seg_fig = px.scatter(
       df.sort_values(segment_feature),
       x=segment_feature,
       y="uplift_score",
       trendline="lowess"
   )
   seg_fig.update_layout(title=f"Uplift by {segment_feature.title()}")
   st.plotly_chart(seg_fig, use_container_width=True)


   st.subheader("📋 Segment-Level Summary")
   st.markdown("Summarized view of average uplift across quantile buckets of the selected segment.")


   binned = pd.cut(df[segment_feature], bins=5)
   grouped = df.groupby(binned)["uplift_score"].mean().reset_index()
   grouped.columns = [segment_feature.title(), "Avg Uplift"]


   st.dataframe(grouped.style.background_gradient(cmap="RdYlGn", subset=["Avg Uplift"]), use_container_width=True)


# ----------------------------
# Tab 4: User Rankings
# ----------------------------
with tab4:
   st.subheader("📓 Top Individual Predictions")


   st.markdown("""
This table highlights users who would benefit most — or least — from receiving the treatment.


This is a key tool in **personalized interventions** — targeting users with high predicted benefit and avoiding harm.
""")


   col1, col2 = st.columns(2)
   with col1:
       st.markdown("🔼 **Top 5 Helped by Treatment**")
       st.dataframe(df.sort_values("uplift_score", ascending=False).head(5), use_container_width=True)
   with col2:
       st.markdown("🔽 **Top 5 Harmed by Treatment**")
       st.dataframe(df.sort_values("uplift_score").head(5), use_container_width=True)


   st.download_button("📥 Download All Uplift Scores", df.to_csv(index=False), file_name="uplift_scores.csv")




# ----------------------------
# Tab 5: Final Summary
# ----------------------------
with tab5:
   st.subheader("🎯 Strategic Takeaways & Business Impact")


   col1, col2, col3 = st.columns(3)
   col1.metric("Average Treatment Effect", "~10.9%", "via X-Learner")
   col2.metric("Statistical Significance", "p < 0.05", "T-test validated")
   col3.metric("Top Segment Uplift", "11.8%", "Age 25–35, >100 actions")


   st.markdown("""
   ### 📊 Experiment Results Summary
   This A/B test shows a **statistically significant** and **practically meaningful** uplift in booking rates.
   - **Treated users** booked at a rate of **25.4%** vs **14.6%** for control — an **absolute gain of 10.8%**
   - **X-Learner** estimated an **ATE of 10.9%**, aligning with observed differences
   - **High-uplift segment** (age 25–35 with >100 actions) saw **11.8% uplift**, slightly outperforming the 10.8% average elsewhere


   ### 📌 Deployment Strategy
   Adopt a **phased rollout**:
   - Start with younger, high-engagement users to maximize ROI early
   - Expand to full population given broad uplift observed across segments


   Leverage **uplift scores** for smarter targeting in future campaigns, and continue using **CUPED** to reduce variance and required sample sizes.


   **Business Impact**: With minimal implementation cost, an 11% conversion lift could yield major revenue gains — especially when prioritized toward high-impact cohorts.
   """)


   st.divider()
   st.subheader("📈 Deployment Impact Calculator")
   st.markdown("Interactive tool to evaluate different targeting strategies and expected business outcomes.")


   deployment_pct = st.slider(
       "Target Top X% of Users by Predicted Uplift",
       min_value=10,
       max_value=100,
       value=50,
       help="Select what percentage of users to target, ranked by individual uplift scores"
   )


   impact = calculate_relative_impact(df, deployment_pct)
  
   col1, col2, col3 = st.columns(3)
   col1.metric("Users Targeted", f"{impact['users_targeted']:,}")
   col2.metric("Expected Additional Bookings", f"{impact['additional_bookings']:.1f}")
   col3.metric("Relative Booking Increase", f"{impact['relative_improvement']:.1f}%")
  
   st.info(f"**Strategy insight**: Targeting the top {deployment_pct}% of users by uplift score would generate {impact['additional_bookings']:.1f} additional bookings from {impact['users_targeted']:,} users, representing a {impact['relative_improvement']:.1f}% increase over their baseline booking behavior.")
  
   if deployment_pct <= 25:
       st.success("🎯 **Conservative targeting**: Highest ROI, focusing on users most likely to benefit")
   elif deployment_pct <= 75:
       st.info("⚖️ **Balanced approach**: Good coverage while maintaining strong average uplift")
   else:
       st.warning("🚀 **Broad deployment**: Maximum reach, includes some lower-uplift users")




# ----------------------------
# Tab 6: CUPED Adjustment
# ----------------------------
with tab6:
   st.subheader("📉 CUPED vs Raw Booking Rates")
   st.markdown("""
**CUPED (Controlled Pre-Experiment Data)** uses a pre-treatment covariate highly correlated with the outcome
to reduce variance in the estimated effect.
""")              


   st.info("**NOTE:** In this case, the adjusted booking rate is **nearly identical** to the raw outcome due to lower correlation.")


   agg = df.groupby("Treatment Label").agg(
       raw_booking=("booking", "mean"),
       cuped_booking=("booking_cuped", "mean")
   ).reset_index()


   cuped_fig = go.Figure()
   cuped_fig.add_trace(go.Bar(name="Raw Booking Rate", x=agg["Treatment Label"], y=agg["raw_booking"]))
   cuped_fig.add_trace(go.Bar(name="CUPED Adjusted", x=agg["Treatment Label"], y=agg["cuped_booking"]))
   cuped_fig.update_layout(barmode="group", title="Comparison of Raw vs CUPED-Adjusted Booking Rates")
   st.plotly_chart(cuped_fig, use_container_width=True)


# ----------------------------
# Tab 7: Glossary
# ----------------------------
with tab7:
   st.subheader("📚 Glossary")
   st.markdown("""
- **Uplift Modeling**: Predicts the difference in outcome between treatment and control for each user. 
- **CUPED**: Uses pre-experiment data to reduce variance and increase precision. 
- **ATE**: Average Treatment Effect — mean uplift across all users. 
- **ITE**: Individual Treatment Effect — uplift for a specific user or segment. 
- **HTE**: Heterogeneous Treatment Effects — variation in uplift across user segments. 
- **X-Learner**: A two-stage model for estimating ITEs in observational or experimental data. 
- **Propensity Score Matching**: Balances treatment and control groups using covariates. 
- **T-Test**: Statistical test to compare means between treated and control groups.
""")
# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("© 2025 Causify Experimentation Platform | Built with Streamlit, Plotly, and CausalML | Simulated Airbnb A/B Experiment")

