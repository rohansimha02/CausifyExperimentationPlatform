import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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
sns.set_style("whitegrid")

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.title("🌐 Dashboard Filters")
    treatment_group = st.radio("Filter by Treatment Group", ["All", "Treated", "Control"])
    segment_feature = st.selectbox("Segment Feature", ["age", "total_actions", "unique_actions", "total_secs_elapsed"])
    st.markdown("---")
    st.caption("Built with Streamlit, CausalML, and Matplotlib")

if treatment_group == "Treated":
    df = df[df["treatment"] == 1]
elif treatment_group == "Control":
    df = df[df["treatment"] == 0]

# ----------------------------
# Title
# ----------------------------
st.title("📊 Causify: A/B Test + Causal Inference Dashboard")
st.markdown("""
Welcome to **Causify**, a dashboard for interpreting A/B test and uplift modeling results from a simulated Airbnb experiment.

We explore:
- 🎯 **Average treatment effects** (ATE)
- 📉 **CUPED**-adjusted outcomes
- 🔄 **Heterogeneous treatment effects** (HTE) via uplift modeling
""")

# ----------------------------
# Metrics Summary
# ----------------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🔢 Total Users", f"{len(df):,}")
with col2:
    st.metric("🎯 Treated", f"{(df['treatment']==1).sum():,}")
with col3:
    st.metric("📉 Control", f"{(df['treatment']==0).sum():,}")

col4, col5, col6 = st.columns(3)
with col4:
    st.metric("🌐 Booking Rate (Treated)", f"{df[df['treatment']==1]['booking'].mean():.2%}")
with col5:
    st.metric("📈 Booking Rate (Control)", f"{df[df['treatment']==0]['booking'].mean():.2%}")
with col6:
    st.metric("🔄 Avg Uplift Score", f"{df['uplift_score'].mean():.4f}")

st.divider()

# ----------------------------
# Booking Rate by Group
# ----------------------------
st.subheader("📊 Booking Rate by Treatment Group")
fig, ax = plt.subplots()
rates = df.groupby("treatment")["booking"].mean()
labels = ["Control", "Treated"]
ax.bar(labels, rates, color=["#A7C7E7", "#5DAF83"])
ax.set_ylabel("Booking Rate")
ax.set_title("Conversion Rate: Treated vs Control")
st.pyplot(fig)
st.markdown("🧠 **Insight**: Compare the raw effectiveness of the treatment group versus control in terms of booking conversion.")

# ----------------------------
# Uplift Distribution
# ----------------------------
st.subheader("🔄 Uplift Score Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(df["uplift_score"], bins=50, kde=True, ax=ax2, color="#8E44AD")
ax2.set_title("Distribution of Estimated Individual Treatment Effects (Uplift Scores)")
ax2.set_xlabel("Estimated Uplift")
st.pyplot(fig2)
st.markdown("🧠 **Insight**: A wide distribution indicates strong heterogeneity in user response. High variance = opportunity for targeting.")

# ----------------------------
# Uplift by Segment
# ----------------------------
st.subheader(f"📈 Uplift by {segment_feature.title()}")
fig3, ax3 = plt.subplots()
df_sorted = df.sort_values(segment_feature)
sns.lineplot(data=df_sorted, x=segment_feature, y="uplift_score", ax=ax3, color="#3498DB")
ax3.set_ylabel("Estimated Uplift Score")
ax3.set_xlabel(segment_feature.replace("_", " ").title())
ax3.set_title(f"HTE by {segment_feature.replace('_', ' ').title()}")
st.pyplot(fig3)
st.markdown("🧠 **Insight**: Spot user segments with positive or negative lift — helpful for campaign targeting or exclusion.")

# ----------------------------
# Segment Table: Top/Bottom
# ----------------------------
st.subheader("📅 Segment Performance Summary")
segment_groups = pd.cut(df[segment_feature], bins=5)
segment_summary = df.groupby(segment_groups)["uplift_score"].mean().reset_index()
segment_summary.columns = [segment_feature.title(), "Avg Uplift"]
st.dataframe(segment_summary.sort_values("Avg Uplift", ascending=False), use_container_width=True)

# ----------------------------
# Top & Bottom Users
# ----------------------------
st.subheader("📃 Top/Bottom Users by Uplift Score")
col7, col8 = st.columns(2)
with col7:
    st.markdown("**🔼 Top 5 Users Most Likely to Benefit**")
    st.dataframe(df.sort_values("uplift_score", ascending=False).head()[["age", "total_actions", "unique_actions", "total_secs_elapsed", "uplift_score"]], use_container_width=True)
with col8:
    st.markdown("**🔽 Top 5 Users Most Likely Harmed**")
    st.dataframe(df.sort_values("uplift_score").head()[["age", "total_actions", "unique_actions", "total_secs_elapsed", "uplift_score"]], use_container_width=True)

# ----------------------------
# Export
# ----------------------------
st.download_button("📂 Download Uplift Scores CSV", data=df.to_csv(index=False), file_name="uplift_scores.csv", mime="text/csv")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("© 2025 Causify | Built using Streamlit & CausalML | Airbnb dataset simulation")
