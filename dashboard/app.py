# dashboard/app.py

"""
Causify Streamlit Dashboard:
Visualizes A/B test results, CUPED-adjusted lift, and uplift modeling outputs from Airbnb experiment data.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
@st.cache_data

def load_data():
    df = pd.read_csv("../data/uplift_scores.csv")
    return df

df = load_data()

# Page config
st.set_page_config(page_title="Causify Dashboard", layout="wide")

# Title
st.title("Causify: A/B Testing and Causal Inference Dashboard")
st.markdown("""
This dashboard visualizes the results of a simulated A/B test on Airbnb data.
It shows the impact of a new homepage feature on booking conversion using:
- Hypothesis testing
- CUPED variance reduction
- Uplift modeling (X-Learner)
""")

# Metrics summary
col1, col2 = st.columns(2)

with col1:
    booking_treated = df[df["treatment"] == 1]["booking"].mean()
    st.metric("Booking Rate (Treated)", f"{booking_treated:.2%}")

with col2:
    booking_control = df[df["treatment"] == 0]["booking"].mean()
    st.metric("Booking Rate (Control)", f"{booking_control:.2%}")

# Booking rate bar chart
fig, ax = plt.subplots()
rates = df.groupby("treatment")["booking"].mean()
labels = ["Control", "Treated"]
ax.bar(labels, rates, color=["gray", "green"])
ax.set_ylabel("Booking Rate")
ax.set_title("Booking Rate by Treatment Group")
st.pyplot(fig)

st.divider()

# Uplift distribution
st.subheader("Uplift Score Distribution (X-Learner)")
fig2, ax2 = plt.subplots()
sns.histplot(df["uplift_score"], bins=50, kde=True, ax=ax2, color="purple")
ax2.set_title("Distribution of Individual Treatment Effects")
ax2.set_xlabel("Estimated Uplift")
st.pyplot(fig2)

# Segment filter
st.subheader("Explore Uplift by Segment")
segment = st.selectbox("Select Feature to Segment By:", ["age", "total_actions", "unique_actions", "total_secs_elapsed"])

fig3, ax3 = plt.subplots()
df_sorted = df.sort_values(segment)
sns.lineplot(data=df_sorted, x=segment, y="uplift_score", ax=ax3)
ax3.set_title(f"Average Uplift by {segment}")
ax3.set_ylabel("Estimated Uplift")
ax3.set_xlabel(segment)
st.pyplot(fig3)

# Footer
st.markdown("---")
st.caption("Built with using Streamlit, CausalML, and Airbnb Kaggle data (public dataset). Not affiliated with Airbnb.")
