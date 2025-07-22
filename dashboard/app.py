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
    df = pd.read_csv("./data/uplift_scores.csv")
    return df

df = load_data()

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Causify Dashboard", layout="wide")

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.title("\U0001F310 Dashboard Filters")
    treatment_group = st.radio("Filter by Treatment Group", ["All", "Treated", "Control"])
    segment_feature = st.selectbox("Segment Feature", ["age", "total_actions", "unique_actions", "total_secs_elapsed"])
    age_range = st.slider("Filter by Age Range", int(df["age"].min()), int(df["age"].max()), (18, 80))
    action_range = st.slider("Filter by Total Actions", int(df["total_actions"].min()), int(df["total_actions"].max()), (0, 200))
    st.markdown("---")
    st.caption("Built with Streamlit, Plotly, and CausalML")

if treatment_group == "Treated":
    df = df[df["treatment"] == 1]
elif treatment_group == "Control":
    df = df[df["treatment"] == 0]

# Filter by sliders
df = df[(df["age"].between(age_range[0], age_range[1])) & (df["total_actions"].between(action_range[0], action_range[1]))]

# ----------------------------
# Header
# ----------------------------
st.title("\U0001F4CA Causify: A/B Test + Causal Inference Dashboard")
st.markdown("""
Welcome to **Causify**, a dashboard for interpreting A/B test and uplift modeling results from a simulated Airbnb experiment.

Explore:
- \U0001F3AF **Average treatment effects** (ATE)
- \U0001F4C9 **CUPED**-adjusted outcomes
- \U0001F501 **Heterogeneous treatment effects** (HTE) via uplift modeling
""")

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Distributions", "Segments", "Users"])

with tab1:
    st.subheader("\U0001F4CB Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Users", f"{len(df):,}")
    col2.metric("Treated", f"{(df['treatment']==1).sum():,}")
    col3.metric("Control", f"{(df['treatment']==0).sum():,}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Booking Rate (Treated)", f"{df[df['treatment']==1]['booking'].mean():.2%}")
    col5.metric("Booking Rate (Control)", f"{df[df['treatment']==0]['booking'].mean():.2%}")
    col6.metric("Avg Uplift Score", f"{df['uplift_score'].mean():.4f}")

    st.divider()

    st.subheader("\U0001F4C8 Booking Rate by Treatment Group")
    rate_fig = px.bar(
        df.groupby("treatment")["booking"].mean().reset_index(),
        x="treatment",
        y="booking",
        color="treatment",
        labels={"booking": "Booking Rate", "treatment": "Treatment"},
        color_discrete_map={0: "#A7C7E7", 1: "#5DAF83"},
    )
    st.plotly_chart(rate_fig, use_container_width=True)

with tab2:
    st.subheader("\U0001F501 Uplift Score Distribution")
    dist_fig = px.histogram(df, x="uplift_score", nbins=50, marginal="rug", color_discrete_sequence=["#8E44AD"])
    st.plotly_chart(dist_fig, use_container_width=True)

    st.subheader("\U0001F4C9 Heatmap: Age × Actions vs. Uplift")
    df_heatmap = df.copy()
    df_heatmap["age_bin"] = pd.cut(df["age"], bins=5)
    df_heatmap["action_bin"] = pd.cut(df["total_actions"], bins=5)
    pivot = df_heatmap.pivot_table(index="age_bin", columns="action_bin", values="uplift_score", aggfunc="mean")
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=[str(i) for i in pivot.index],
        colorscale="RdBu",
        colorbar_title="Avg Uplift"
    ))
    heatmap_fig.update_layout(title="Uplift Heatmap by Age × Total Actions")
    st.plotly_chart(heatmap_fig, use_container_width=True)

with tab3:
    st.subheader(f"\U0001F4CA Uplift by {segment_feature.title()}")
    seg_fig = px.scatter(
        df.sort_values(segment_feature),
        x=segment_feature,
        y="uplift_score",
        trendline="lowess",
        color_discrete_sequence=["#3498DB"],
        labels={"uplift_score": "Estimated Uplift"}
    )
    st.plotly_chart(seg_fig, use_container_width=True)

    st.subheader("\U0001F4C5 Segment Performance Summary")
    segment_groups = pd.cut(df[segment_feature], bins=5)
    segment_summary = df.groupby(segment_groups)["uplift_score"].mean().reset_index()
    segment_summary.columns = [segment_feature.title(), "Avg Uplift"]
    styled_df = segment_summary.style.background_gradient(cmap="RdYlGn", subset=["Avg Uplift"])
    st.dataframe(styled_df, use_container_width=True)

with tab4:
    st.subheader("\U0001F4D3 Top & Bottom Users by Uplift")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**\U0001F53C Top 5 Users Most Likely to Benefit**")
        st.dataframe(
            df.sort_values("uplift_score", ascending=False).head()[["age", "total_actions", "unique_actions", "total_secs_elapsed", "uplift_score"]],
            use_container_width=True)
    with col2:
        st.markdown("**\U0001F53D Top 5 Users Most Likely Harmed**")
        st.dataframe(
            df.sort_values("uplift_score").head()[["age", "total_actions", "unique_actions", "total_secs_elapsed", "uplift_score"]],
            use_container_width=True)

    st.download_button("\U0001F4C2 Download Full Uplift Scores", data=df.to_csv(index=False), file_name="uplift_scores.csv", mime="text/csv")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("© 2025 Causify | Airbnb A/B Simulation | Built with Plotly, Streamlit & CausalML")
