# ==========================================
# 📊 Causify: A/B Testing + Uplift Modeling Dashboard
# ==========================================
# This Streamlit app visualizes treatment effects from a simulated A/B test.
# It combines classic statistical inference (t-tests), CUPED adjustment for variance reduction,
# and modern uplift modeling to understand how different users respond to treatment.
# Users can interactively explore uplift by segment, behavior, or individually,
# helping simulate targeted rollout decisions in a product setting.
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------
# Load Data with Caching
# ----------------------------
@st.cache_data
def load_data():
    """
    Load final merged and enriched dataset with:
    - demographic and behavioral features
    - treatment assignment
    - simulated booking outcomes
    - estimated uplift scores
    """
    df = pd.read_csv("./data/final_dashboard_data.csv")
    df["Treatment Label"] = df["treatment"].map({0: "Control", 1: "Treated"})
    return df

df = load_data()

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(page_title="Causify Dashboard", layout="wide")

# ----------------------------
# Sidebar Filters (Global)
# ----------------------------
with st.sidebar:
    st.title("🌐 Dashboard Filters")

    # Global treatment group filter
    treatment_group = st.radio("Filter by Treatment Group", ["All", "Treated", "Control"])

    # Segment feature selection for analysis
    segment_feature = st.selectbox(
        "Segment Feature (for HTE)",
        ["age", "total_actions", "unique_actions", "total_secs_elapsed"]
    )

    # Age slider to filter visualized cohort
    age_range = st.slider("Filter by Age", int(df["age"].min()), int(df["age"].max()), (18, 80))

    # Interaction range filter
    action_range = st.slider("Filter by Total Actions", int(df["total_actions"].min()), int(df["total_actions"].max()), (0, 500))

    st.markdown("---")
    st.caption("Built with Streamlit, Plotly, and CausalML")

# ----------------------------
# Apply Filter Logic
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
# Dashboard Introduction
# ----------------------------
st.title("📈 Causify: A/B Test + Causal Inference Dashboard")

st.markdown("""
Welcome to **Causify**, an interactive dashboard for analyzing treatment effects from a simulated Airbnb A/B test.

### 🔬 Experimental Context
- **Treatment**: A new booking experience interface shown to a randomized group of users.
- **Simulation Setup**: Treated users have a 25% booking probability, while control users have 15%.
- **Objective**: Identify which users benefit or are harmed using **causal inference** and **uplift modeling**.

This dashboard showcases:
- 📉 CUPED (Controlled Pre-experiment Data) adjustment
- 🧪 Hypothesis testing
- 🧠 Uplift modeling for individual-level targeting
- 📊 Segmentation analysis
""")

# ----------------------------
# Tab Navigation
# ----------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview", "Distributions", "Segments", "Users",
    "CUPED Comparison", "Feature Importance", "Summary"
])

# ----------------------------
# Tab 1: Overview & Key Metrics
# ----------------------------
with tab1:
    st.subheader("📋 Key Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Users", f"{len(df):,}")
    col2.metric("Treated Users", f"{(df['treatment'] == 1).sum():,}")
    col3.metric("Control Users", f"{(df['treatment'] == 0).sum():,}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Booking Rate (Treated)", f"{df[df['treatment'] == 1]['booking'].mean():.2%}")
    col5.metric("Booking Rate (Control)", f"{df[df['treatment'] == 0]['booking'].mean():.2%}")
    col6.metric("Avg Uplift Score", f"{df['uplift_score'].mean():.4f}")

    st.success("Statistically significant uplift detected. Treatment increased booking probability by ~10%.")

    st.divider()

    st.subheader("📊 Booking Rate by Group")
    fig = px.bar(
        df.groupby("Treatment Label")["booking"].mean().reset_index(),
        x="Treatment Label", y="booking",
        labels={"booking": "Booking Rate"},
        color="Treatment Label",
        color_discrete_sequence=["#A7C7E7", "#5DAF83"]
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Tab 2: Uplift Distribution & Heatmap
# ----------------------------
with tab2:
    st.subheader("🔁 Uplift Score Distribution")

    dist_fig = px.histogram(df, x="uplift_score", nbins=50, marginal="rug", color_discrete_sequence=["#8E44AD"])
    dist_fig.update_layout(title="Distribution of Estimated Treatment Effects (Uplift Scores)")
    st.plotly_chart(dist_fig, use_container_width=True)

    st.info("Uplift scores near 0 show neutral users. The tails show who is most helped (positive) or harmed (negative).")

    st.subheader("📊 Heatmap: Uplift by Age × Total Actions")

    # Bin numeric features into groups
    df_heatmap = df.copy()
    df_heatmap["age_bin"] = pd.cut(df_heatmap["age"], bins=5)
    df_heatmap["action_bin"] = pd.cut(df_heatmap["total_actions"], bins=5)

    # Create pivot table for heatmap
    pivot = df_heatmap.pivot_table(index="age_bin", columns="action_bin", values="uplift_score", aggfunc="mean")

    heatmap = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=[str(i) for i in pivot.index],
        colorscale="RdBu",
        colorbar_title="Avg Uplift"
    ))
    heatmap.update_layout(title="Uplift Heatmap by Segment (Age × Engagement)")
    st.plotly_chart(heatmap, use_container_width=True)

# ----------------------------
# Tab 3: Segment-Level Uplift
# ----------------------------
with tab3:
    st.subheader(f"📈 Uplift by {segment_feature.title()}")

    seg_fig = px.scatter(
        df.sort_values(segment_feature),
        x=segment_feature,
        y="uplift_score",
        trendline="lowess"
    )
    seg_fig.update_layout(title=f"Uplift by {segment_feature}")
    st.plotly_chart(seg_fig, use_container_width=True)

    st.subheader("📋 Segment Summary Table")

    # Group by selected feature and calculate average uplift
    segment_groups = pd.cut(df[segment_feature], bins=5)
    segment_summary = df.groupby(segment_groups)["uplift_score"].mean().reset_index()
    segment_summary.columns = [segment_feature.title(), "Avg Uplift"]

    styled_df = segment_summary.style.background_gradient(cmap="RdYlGn", subset=["Avg Uplift"])
    st.dataframe(styled_df, use_container_width=True)

# ----------------------------
# Tab 4: User-Level Rankings
# ----------------------------
with tab4:
    st.subheader("📓 Individual User Impact")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("🔼 **Top 5 Users Most Helped by Treatment**")
        st.dataframe(df.sort_values("uplift_score", ascending=False).head(5), use_container_width=True)

    with col2:
        st.markdown("🔽 **Top 5 Users Most Likely Harmed by Treatment**")
        st.dataframe(df.sort_values("uplift_score").head(5), use_container_width=True)

    st.download_button(
        "📂 Download Uplift Data",
        data=df.to_csv(index=False),
        file_name="uplift_scores.csv"
    )

# ----------------------------
# Tab 5: CUPED Comparison
# ----------------------------
with tab5:
    st.subheader("📉 CUPED vs Raw Booking Rates")

    # Group booking and cuped_booking rates by treatment group
    agg = df.groupby("Treatment Label").agg(
        raw_booking=("booking", "mean"),
        cuped_booking=("booking_cuped", "mean")
    ).reset_index()

    # CUPED vs raw comparison chart
    cuped_fig = go.Figure()
    cuped_fig.add_trace(go.Bar(name="Raw Booking Rate", x=agg["Treatment Label"], y=agg["raw_booking"]))
    cuped_fig.add_trace(go.Bar(name="CUPED Adjusted", x=agg["Treatment Label"], y=agg["cuped_booking"]))
    cuped_fig.update_layout(barmode="group", title="Comparison of Raw vs CUPED-Adjusted Booking Rates")
    st.plotly_chart(cuped_fig, use_container_width=True)

    st.info("CUPED adjusts for baseline differences using pre-treatment covariates, reducing variance and increasing sensitivity.")


# ----------------------------
# Tab 7: Final Summary + Takeaways
# ----------------------------
with tab7:
    st.subheader("🎯 Summary & Recommendations")

    st.markdown("""
- **✅ High-Uplift Segments**: Young users (25–35) with high engagement (100+ actions)
- **⚠️ Avoid**: Older, low-engagement users who may be negatively impacted
- **📈 Strategy**: Roll out treatment to high-uplift segments for max ROI
- **🧪 Result**: T-test and CUPED confirm 10% uplift (p < 0.001)
- **🧮 ATE Estimate**: Matching-based causal inference confirms uplift (~10.6%)

This simulation illustrates how experimentation + uplift modeling supports **data-driven, personalized product rollouts**.
""")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("© 2025 Causify | Built with Streamlit, Plotly, and CausalML | Simulated Airbnb A/B Experiment")
