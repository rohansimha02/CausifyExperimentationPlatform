# ==========================================
# Causify Experimentation Platform
# ==========================================
# A/B testing dashboard with uplift modeling,
# statistical validation, and ROI-focused targeting.
# ==========================================

from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Page configuration and color scheme
st.set_page_config(
    page_title="Causify Experimentation Platform",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Brand color palette
COLORS = {
    "primary": "#6c5ce7",
    "accent":  "#e84393",
    "teal":    "#00b894",
    "grid":    "rgba(108,92,231,0.18)",
    "softbg":  "rgba(108,92,231,0.06)"
}

st.markdown(f"""
<style>
  #MainMenu, header {{visibility: hidden;}}
  footer {{visibility: hidden;}}

  .stTabs [data-baseweb="tab-list"] {{ gap: 12px; justify-content: center; }}
  .stTabs [data-baseweb="tab"] {{
    padding: 8px 16px; border-radius: 8px; border: 1px solid rgba(0,0,0,0.06);
  }}

  .kpi-card {{
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 14px;
    padding: 14px 16px;
    background: linear-gradient(135deg, rgba(108,92,231,0.08), rgba(232,67,147,0.08));
  }}
  .kpi-title {{ font-size: 0.9rem; color: #4b5563; margin-bottom: 2px; }}
  .kpi-value {{ font-size: 1.8rem; font-weight: 700; }}
  .kpi-sub   {{ font-size: 0.9rem; color: #6b7280; }}

  .notice {{
    border: 1px solid rgba(16,185,129,0.25);
    background: rgba(16,185,129,0.08);
    padding: 12px 14px; border-radius: 10px; margin: 6px 0 14px 0;
  }}
  .warn {{
    border: 1px solid rgba(251,191,36,0.35);
    background: rgba(251,191,36,0.08);
    padding: 12px 14px; border-radius: 10px; margin: 6px 0 14px 0;
  }}
</style>
""", unsafe_allow_html=True)

# Data source configuration
DEFAULT_LOCAL = Path(__file__).resolve().parent / "final_dashboard_data.csv"
DEFAULT_PARENT = Path(__file__).resolve().parent.parent / "data" / "final_dashboard_data.csv"

st.sidebar.header("Data")
local_exists = DEFAULT_LOCAL.exists() or DEFAULT_PARENT.exists()
source = st.sidebar.selectbox(
    "Source",
    options=(["Local file"] if local_exists else []) + ["Upload"],
    index=0 if local_exists else 0
)
uploaded = None
if source == "Upload":
    uploaded = st.sidebar.file_uploader("Upload final_dashboard_data.csv", type=["csv"])

@st.cache_data(ttl=3600)
def _load_data_from_source(path_candidates: list[str], raw_bytes: bytes | None):
    if raw_bytes is not None:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    else:
        path = next((p for p in path_candidates if p and Path(p).exists()), None)
        if path is None:
            raise ValueError("No data source provided. Upload a CSV or place final_dashboard_data.csv next to app.py or in ../data/.")
        df = pd.read_csv(path)

    required = {
        "treatment", "booking", "age",
        "uplift_score", "uplift_score_clipped", "propensity_score",
        "global_treatment_rate", "global_control_rate", "global_effect_size",
        "global_z_stat", "global_ci_lower", "global_ci_upper",
        "global_variance_reduction_pct"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    df["treatment"] = df["treatment"].astype(int)
    df["booking"] = df["booking"].astype(int)
    if "activity_level" in df.columns:
        df["activity_level"] = df["activity_level"].astype("category")
    if "engagement_level" in df.columns:
        df["engagement_level"] = df["engagement_level"].astype("category")

    df["group_label"] = df["treatment"].map({0: "Control", 1: "Treatment"})
    return df

try:
    df = _load_data_from_source(
        [str(DEFAULT_LOCAL), str(DEFAULT_PARENT)] if source == "Local file" else [],
        uploaded.getvalue() if uploaded is not None else None
    )
except Exception as e:
    st.error(str(e))
    st.stop()

# Sidebar controls and filters
st.sidebar.header("Controls")

treatment_group = st.sidebar.radio(
    "Treatment Group",
    ["All Groups", "Treatment Only", "Control Only"],
    help="Filter by assignment"
)

use_clipped = st.sidebar.toggle("Use clipped uplift", value=True, help="Compare raw vs clipped uplift for sensitivity.")
uplift_col = "uplift_score_clipped" if use_clipped else "uplift_score"

booking_value = st.sidebar.number_input("Value per booking ($)", min_value=0, value=150, step=10)
impact_scale = st.sidebar.number_input("Users scale for impact", min_value=10_000, value=100_000, step=10_000)

with st.sidebar.expander("Filters", expanded=False):
    if "activity_level" in df.columns:
        act_opts = sorted(df["activity_level"].dropna().unique().tolist())
        sel_act = st.multiselect("Activity level", act_opts, default=act_opts)
    else:
        sel_act = None

    if "engagement_level" in df.columns:
        eng_opts = sorted(df["engagement_level"].dropna().unique().tolist())
        sel_eng = st.multiselect("Engagement level", eng_opts, default=eng_opts)
    else:
        sel_eng = None

    min_age, max_age = int(np.nanmin(df["age"])), int(np.nanmax(df["age"]))
    age_rng = st.slider("Age", min_age, max_age, (min_age, max_age))

# Assignment filter
df_view = df
if treatment_group == "Treatment Only":
    df_view = df_view[df_view["treatment"] == 1]
elif treatment_group == "Control Only":
    df_view = df_view[df_view["treatment"] == 0]

# Apply filters
mask = (
    df_view["age"].between(age_rng[0], age_rng[1])
    & (df_view["activity_level"].isin(sel_act) if sel_act else True)
    & (df_view["engagement_level"].isin(sel_eng) if sel_eng else True)
)
df_f = df_view.loc[mask].copy()

# Utility functions
def pct(x, d=1): return f"{x:.{d}%}"

def tc_rates(frame: pd.DataFrame):
    g = frame.groupby("group_label")["booking"].mean()
    return float(g.get("Treatment", np.nan)), float(g.get("Control", np.nan))

def ci_includes_zero(lo, hi): return lo <= 0 <= hi

def calculate_lift_stats(df_filtered):
    """Calculate lift statistics for filtered data"""
    tr, cr = tc_rates(df_filtered)
    lift = tr - cr
    
    n_tr = (df_filtered["group_label"] == "Treatment").sum()
    n_cr = (df_filtered["group_label"] == "Control").sum()
    
    se_tr = np.sqrt(tr * (1 - tr) / max(n_tr, 1))
    se_cr = np.sqrt(cr * (1 - cr) / max(n_cr, 1))
    se_diff = np.sqrt(se_tr**2 + se_cr**2)
    
    z_stat = lift / se_diff if se_diff > 0 else 0
    margin = 1.96 * se_diff
    ci_lower = lift - margin
    ci_upper = lift + margin
    
    return lift, z_stat, ci_lower, ci_upper

def compute_deciles(frame: pd.DataFrame, score_col: str, q=10):
    valid_frame = frame.dropna(subset=[score_col])
    r = valid_frame[score_col].rank(method="first", pct=True)
    dec = np.ceil(r * q).clip(1, q).astype(int)
    result = frame.copy()
    result['decile'] = np.nan
    result.loc[valid_frame.index, 'decile'] = dec
    result['decile'] = result['decile'].astype('Int64')
    return result, q

def decile_calibration(frame: pd.DataFrame, dec_col="decile"):
    valid_frame = frame.dropna(subset=[dec_col])
    agg = (
        valid_frame.groupby([dec_col, "treatment"])["booking"]
        .agg(["mean", "count"])
        .reset_index()
        .pivot(index=dec_col, columns="treatment", values=["mean", "count"])
        .sort_index()
    )
    tr = agg[("mean", 1)].fillna(0.0)
    cr = agg[("mean", 0)].fillna(0.0)
    n_tr = agg[("count", 1)].fillna(0)
    n_cr = agg[("count", 0)].fillna(0)
    lift = (tr - cr).fillna(0.0)
    return pd.DataFrame({
        "bucket": agg.index,
        "treat_rate": tr.values,
        "ctrl_rate": cr.values,
        "lift": lift.values,
        "n_treat": n_tr.values,
        "n_ctrl": n_cr.values
    })

def build_gain_curve(frame: pd.DataFrame, score_col: str, step=0.1):
    tmp = frame[["booking", "treatment", score_col]].dropna().sort_values(score_col, ascending=False).reset_index(drop=True)
    tmp["incremental"] = tmp[score_col]
    x, y = [], []
    n = len(tmp)
    for frac in np.arange(step, 1.0 + 1e-9, step):
        k = int(n * frac)
        x.append(frac)
        y.append(tmp.iloc[:k]["incremental"].sum())
    return x, y

# Main dashboard content
st.title("Causify Experimentation Platform")

st.markdown(
    "**What this is:** A randomized experiment on Airbnb booking data."
    " Compares a *new booking interface* (treatment) to the current (control),"
    " summarizes results, shows who benefits most, and outlines an ROI-first rollout."
)

st.markdown(
    "Tabs: **(1)** experiment results, **(2)** uplift (who benefits), **(3)** targeting strategy, **(4)** validation."
)

# Tab navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Uplift Analysis", 
    "Targeting Strategy",
    "Model Summary",
    "Summary & Recommendations"
])

# Tab 1: Overview
with tab1:
    st.subheader("Executive Summary")
    st.caption("Does the new booking interface improve conversion rates? By how much, and can we trust the results?")

    # Get current filtered statistics
    tr, cr = tc_rates(df_f)
    lift_val, z_stat, ci_lower, ci_upper = calculate_lift_stats(df_f)

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Total users (current view)</div>'
                    f'<div class="kpi-value">{len(df_f):,}</div>'
                    f'<div class="kpi-sub">Filters on the left</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Treatment conversion</div>'
                    f'<div class="kpi-value">{pct(tr,2)}</div>'
                    f'<div class="kpi-sub">New interface</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Control conversion</div>'
                    f'<div class="kpi-value">{pct(cr,2)}</div>'
                    f'<div class="kpi-sub">Current interface</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Lift (treatment − control)</div>'
                    f'<div class="kpi-value">{pct(lift_val,1)}</div>'
                    f'<div class="kpi-sub">95% CI {pct(ci_lower,1)} to {pct(ci_upper,1)} · z={z_stat:.2f}</div></div>',
                    unsafe_allow_html=True)

    if ci_includes_zero(ci_lower, ci_upper):
        st.markdown(
            f'<div class="warn"><strong>Result:</strong> Inconclusive at 95%. '
            f'Observed lift {pct(lift_val,1)} (CI {pct(ci_lower,1)} to {pct(ci_upper,1)}). '
            f'Consider more sample or stronger variance reduction.</div>',
            unsafe_allow_html=True
        )
    else:
        added = lift_val * impact_scale
        st.markdown(
            f'<div class="notice"><strong>Result:</strong> Statistically significant improvement. '
            f'At {impact_scale:,} users → ≈ {added:,.0f} extra bookings '
            f'(≈ ${added*booking_value:,.0f}).</div>',
            unsafe_allow_html=True
        )

    # Comparison chart
    n_tr = (df_f["group_label"] == "Treatment").sum()
    n_cr = (df_f["group_label"] == "Control").sum()
    se_tr = np.sqrt(tr * (1 - tr) / max(n_tr, 1))
    se_cr = np.sqrt(cr * (1 - cr) / max(n_cr, 1))

    fig = go.Figure()
    # Control bar
    fig.add_trace(go.Bar(
        x=["Current interface (Control)"],
        y=[cr],
        error_y=dict(type="data", array=[1.96*se_cr], visible=True, color=COLORS["accent"], thickness=2),
        marker=dict(color=COLORS["primary"]),
        text=[f'{cr:.2%}'],
        textposition='outside',
        hovertemplate="<b>%{x}</b><br>Conversion: %{y:.2%}<extra></extra>",
        showlegend=False
    ))
    # Treatment bar
    fig.add_trace(go.Bar(
        x=["New interface (Treatment)"],
        y=[tr],
        error_y=dict(type="data", array=[1.96*se_tr], visible=True, color="#8e44ad", thickness=2),
        marker=dict(color=COLORS["accent"]),
        text=[f'{tr:.2%}'],
        textposition='outside',
        hovertemplate="<b>%{x}</b><br>Conversion: %{y:.2%}<extra></extra>",
        showlegend=False
    ))
    fig.add_annotation(
        x=1, y=tr*(1.12), text=f"<b>+{lift_val:.1%} lift</b>",
        showarrow=True, arrowhead=2, arrowcolor=COLORS["teal"], arrowwidth=2,
        font=dict(size=14, color=COLORS["teal"]),
        bgcolor="rgba(0,184,148,0.12)", bordercolor=COLORS["teal"], borderwidth=1
    )
    fig.add_shape(type="rect", x0=-0.5, y0=0, x1=1.5, y1=max(cr, tr)*1.3, fillcolor=COLORS["softbg"], line=dict(width=0))
    fig.update_layout(
        title=dict(
            text="New vs Current Interface (Two-Sample Z-Test with 95% CI)", 
            x=0.5, 
            xanchor='center'
        ),
        yaxis_title="Booking conversion rate",
        xaxis_title="Experiment group",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(tickformat='.1%', gridcolor=COLORS["grid"]),
        height=420, margin=dict(t=50,b=40,l=40,r=40), showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Uplift analysis
with tab2:
    st.subheader("Who Benefits from the New Interface?")

    mean_uplift = df_f[uplift_col].mean()
    st.caption(f"Uplift = predicted increase in booking chance if treated. Dashed line = average predicted improvement ({mean_uplift:.2%}).")

    # Individual uplift distribution
    dist = px.histogram(
        df_f, x=uplift_col, nbins=40, marginal="rug",
        color_discrete_sequence=[COLORS["accent"]],
        labels={uplift_col: "Predicted improvement in booking chance"}
    )
    dist.add_vline(
        x=mean_uplift,
        line_dash="dash",
        line_color=COLORS["primary"]
    )
    dist.update_layout(
        title=dict(text="Distribution of Individual Uplift Predictions", x=0.5, xanchor='center'),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(gridcolor=COLORS["grid"], title="Users"),
        xaxis=dict(gridcolor=COLORS["grid"]),
        height=400
    )

    st.plotly_chart(dist, use_container_width=True)

    pos_share = (df_f[uplift_col] > 0).mean()
    hi_share = (df_f[uplift_col] > np.percentile(df_f[uplift_col], 80)).mean()
    max_gain = df_f[uplift_col].max()
    c1, c2, c3 = st.columns(3)
    for title, val, sub in [
        ("Users who benefit", pct(pos_share,0), "Share with > 0% predicted improvement"),
        ("High-impact users", pct(hi_share,0), "Top 20% by predicted improvement"),
        ("Maximum predicted gain", pct(max_gain,1), "Largest expected single-user lift")
    ]:
        with (c1 if title=="Users who benefit" else c2 if title=="High-impact users" else c3):
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-title">{title}</div>'
                f'<div class="kpi-value">{val}</div>'
                f'<div class="kpi-sub">{sub}</div></div>',
                unsafe_allow_html=True
            )

    st.subheader("Model Validation: Do Higher Uplift Scores Really Lift More?")
    st.caption("Users bucketed into 10 groups by predicted improvement, showing observed lift per bucket.")

    df_dec, _ = compute_deciles(df_f, uplift_col, q=10)
    calib = decile_calibration(df_dec, dec_col="decile").sort_values("bucket")

    line = go.Figure()
    line.add_trace(
        go.Scatter(
            x=calib["bucket"],
            y=calib["lift"],
            mode="lines+markers",
            name="Observed lift",
            line=dict(color=COLORS["primary"], width=3)
        )
    )
    line.update_layout(
        title=dict(text="Model Calibration: Observed Lift by Predicted Impact Group", x=0.5, xanchor='center'),
        xaxis_title="Impact Group (1 = Lowest, 10 = Highest)",
        yaxis_title="Observed Lift (New – Current)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(tickformat=".2%", gridcolor=COLORS["grid"]),
        height=500,
        margin=dict(l=40, r=40, t=60, b=100)
    )
    st.plotly_chart(line, use_container_width=True)

# Tab 3: Targeting
with tab3:
    st.subheader("How to Roll Out for Maximum Impact")
    st.caption("Target the highest predicted improvers first to maximize ROI.")

    pct_slider = st.slider("Choose rollout size (top % by predicted improvement)", 5, 100, 30, step=5)
    n_target = int(len(df_f) * (pct_slider / 100))
    df_rank = df_f.sort_values(uplift_col, ascending=False)

    inc_bookings = df_rank.iloc[:n_target][uplift_col].sum()
    base_total_inc = df_f[uplift_col].sum()

    avg_uplift_targeted = inc_bookings / max(n_target, 1)
    avg_uplift_full = base_total_inc / max(len(df_f), 1)
    roi_boost = avg_uplift_targeted / max(avg_uplift_full, 1e-9)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">Users targeted</div>'
            f'<div class="kpi-value">{n_target:,}</div>'
            f'<div class="kpi-sub">{pct_slider}% of current view</div></div>',
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">Extra bookings</div>'
            f'<div class="kpi-value">{inc_bookings:,.0f}</div>'
            f'<div class="kpi-sub">Sum of predicted uplift in targeted cohort</div></div>',
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">ROI boost (per user)</div>'
            f'<div class="kpi-value">{pct(roi_boost - 1,1) if roi_boost>=0 else pct(roi_boost,1)}</div>'
            f'<div class="kpi-sub">Avg uplift/user vs full rollout</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("&nbsp;", unsafe_allow_html=True)

    if pct_slider <= 25:
        st.success(
            f"Conservative rollout: focus on top {pct_slider}% to maximize ROI; validate with holdouts."
        )
    elif pct_slider <= 50:
        st.info(
            f"Balanced approach: target {pct_slider}% for coverage and efficiency; expand if realized lift holds."
        )
    else:
        st.warning(
            f"Aggressive rollout: {pct_slider}% coverage. Expect diminishing returns; monitor ROI closely."
        )
    st.subheader("Cumulative Impact When Targeting Highest-Scoring Users")
    st.caption("More users targeted → more incremental bookings. Early slope = efficiency (Qini methodology).")

    xs, ys = build_gain_curve(df_f, uplift_col, step=0.1)
    qini = go.Figure()
    qini.add_trace(
        go.Scatter(
            x=[x * 100 for x in xs],
            y=ys,
            mode="lines+markers",
            line=dict(color=COLORS["accent"], width=3),
            marker=dict(size=7, color=COLORS["primary"])
        )
    )
    qini.update_layout(
        title=dict(text="ROI Analysis: Cumulative Impact (Qini Curve)", x=0.5, xanchor='center'),
        xaxis_title="% of users targeted (highest first)",
        yaxis_title="Cumulative incremental bookings",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(gridcolor=COLORS["grid"]),
        xaxis=dict(gridcolor=COLORS["grid"]),
        height=420,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(qini, use_container_width=True)

# Tab 4: Validation
with tab4:
    st.subheader("Experiment Validation: Was the Test Fair and Are Predictions Credible?")
    st.caption("Quality checks for randomization and model trust.")

    st.subheader("Balance & Randomization Diagnostics")
    
    # SMD calculation
    def calculate_smds(df, features):
        smds = []
        for feature in features:
            if feature in df.columns and df[feature].dtype in ['int64', 'float64']:
                treat = df[df['treatment'] == 1][feature].dropna()
                ctrl = df[df['treatment'] == 0][feature].dropna()
                
                if len(treat) > 0 and len(ctrl) > 0:
                    mean_diff = treat.mean() - ctrl.mean()
                    pooled_std = np.sqrt(((treat.var() + ctrl.var()) / 2))
                    if pooled_std > 0:
                        smd = mean_diff / pooled_std
                        smds.append({'feature': feature, 'smd': abs(smd)})
        return pd.DataFrame(smds)
    
    baseline_features = ['age', 'total_actions', 'unique_actions', 'total_secs_elapsed', 'num_sessions', 'actions_per_session']
    available_features = [f for f in baseline_features if f in df_f.columns]
    smd_df = calculate_smds(df_f, available_features)
    
    col_smd, col_prop = st.columns(2)
    
    with col_smd:
        st.markdown("**Baseline Balance (SMDs)**")
        st.caption("Near zero = well-matched groups pre-experiment.")
        if not smd_df.empty:
            fig_smd = go.Figure()
            colors = ["#ff69b4" if smd < 0.1 else "#ff1493" if smd < 0.2 else "#dc143c" for smd in smd_df['smd']]
            fig_smd.add_trace(go.Bar(
                x=smd_df['smd'],
                y=smd_df['feature'],
                orientation='h',
                marker=dict(color=colors, opacity=0.8),
                hovertemplate="<b>%{y}</b><br>SMD: %{x:.3f}<extra></extra>"
            ))
            fig_smd.add_vline(x=0.1, line=dict(color=COLORS["teal"], width=2, dash="dash"))
            fig_smd.update_layout(
                title=dict(text="Covariate Balance", x=0.5, xanchor='center', font=dict(size=14)),
                xaxis=dict(title="Standardized Mean Difference", gridcolor=COLORS["grid"], tickformat=".2f"),
                yaxis=dict(title="Features"),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=320,
                showlegend=False,
                margin=dict(l=10, r=10, t=40, b=40)
            )
            st.plotly_chart(fig_smd, use_container_width=True)
            good_balance = (smd_df['smd'] < 0.1).sum()
            total_features = len(smd_df)
            st.caption(f"✅ {good_balance}/{total_features} features well-balanced (SMD < 0.1)")
        else:
            st.info("SMDs require numeric baseline features")
    
    with col_prop:
        st.markdown("**Randomization Quality (Propensity)**")
        st.caption("Peak near 0.5 suggests true random assignment.")
        if "propensity_score" in df_f.columns:
            fig_prop = go.Figure()
            ps_mean = df_f["propensity_score"].mean()
            fig_prop.add_trace(go.Histogram(
                x=df_f["propensity_score"],
                nbinsx=25,
                marker=dict(color=COLORS["primary"], opacity=0.8, line=dict(color='white', width=1)),
                hovertemplate="<b>Propensity Range:</b> %{x}<br><b>Users:</b> %{y}<extra></extra>"
            ))
            fig_prop.add_vline(x=0.5, line=dict(color=COLORS["accent"], width=2, dash="dash"))
            fig_prop.add_vline(x=ps_mean, line=dict(color=COLORS["teal"], width=2, dash="dot"))
            fig_prop.add_vrect(
                x0=0.45, x1=0.55,
                fillcolor=COLORS["teal"], opacity=0.1,
                layer="below", line_width=0
            )
            fig_prop.update_layout(
                title=dict(text="Assignment Prediction", x=0.5, xanchor='center', font=dict(size=14)),
                xaxis=dict(title="Predicted Prob. of Treatment", gridcolor=COLORS["grid"], range=[0.35, 0.65]),
                yaxis=dict(title="Users", gridcolor=COLORS["grid"]),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=320,
                showlegend=False,
                margin=dict(l=10, r=10, t=40, b=40)
            )
            st.plotly_chart(fig_prop, use_container_width=True)
            balance_score = 1 - abs(ps_mean - 0.5) * 2
            st.caption(f"🎯 Balance score: {balance_score:.3f} (mean ≈ 0.5 = perfect)")
        else:
            st.info("Propensity scores not available")
    
    st.markdown("---")

    st.subheader("Model & Experiment Validation")
    
    model_accuracy = df_f['uplift_model_performance'].iloc[0] if 'uplift_model_performance' in df_f.columns else 0.5
    variance_reduction = df_f['global_variance_reduction_pct'].iloc[0] if 'global_variance_reduction_pct' in df_f.columns else 0
    uplift_range = df_f[uplift_col].max() - df_f[uplift_col].min()
    
    randomization_balance = None
    if "propensity_score" in df_f.columns:
        ps_mean = df_f["propensity_score"].mean()
        randomization_balance = 1 - abs(ps_mean - 0.5) * 2
    
    st.markdown("""
    <style>
    .metric-card {
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 14px;
        padding: 14px 16px;
        background: linear-gradient(135deg, rgba(108,92,231,0.08), rgba(232,67,147,0.08));
        margin-bottom: 16px;
    }
    .metric-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    .metric-header {
        display: flex;
        align-items: center;
        margin-bottom: 12px;
    }
    .metric-icon {
        font-size: 24px;
        width: 36px;
        text-align: center;
        margin-right: 12px;
    }
    .metric-name {
        font-weight: 600;
        font-size: 14px;
        color: #6b7280;
        letter-spacing: 0.01em;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
    }
    .metric-implication {
        font-size: 12px;
        color: #6b7280;
        line-height: 1.4;
        font-style: italic;
    }
    .status-excellent { color: #059669; }
    .status-good { color: #2563eb; }
    .status-warning { color: #d97706; }
    .status-poor { color: #dc2626; }
    </style>
    """, unsafe_allow_html=True)
    
    def get_status_class(metric_type, value):
        if metric_type == "cuped":
            return "status-excellent" if value > 5 else "status-good" if value > 0 else "status-warning"
        elif metric_type == "model":
            return "status-excellent" if value > 0.55 else "status-good" if value > 0.52 else "status-warning"
        elif metric_type == "range":
            return "status-excellent" if value > 0.2 else "status-good" if value > 0.1 else "status-warning"
        elif metric_type == "randomization":
            return "status-excellent" if value >= 0.99 else "status-good" if value >= 0.95 else "status-warning"
        else:
            return "status-good"
    
    def get_statistical_implication(metric_type, value):
        if metric_type == "cuped":
            if value > 5:
                return "Strong precision → High confidence in results"
            elif value > 0:
                return "Good precision → Reliable statistical conclusions"
            else:
                return "No variance reduction → Results may be noisy"
        elif metric_type == "model":
            if value > 0.55:
                return "Strong targeting ability → Proceed with personalized rollout"
            elif value > 0.28:
                return "Moderate targeting → Selective personalization recommended"
            else:
                return "Limited targeting → Focus on broad rollout strategy"
        elif metric_type == "range":
            if value > 0.2:
                return "High diversity → Model captures user heterogeneity well"
            elif value > 0.1:
                return "Moderate diversity → Reasonable prediction spread"
            else:
                return "Low diversity → Limited personalization potential"
        elif metric_type == "randomization":
            if value >= 0.99:
                return "Strong balance → Unbiased experiment setup"
            elif value >= 0.95:
                return "Good balance → Reliable causal attribution"
            else:
                return "Review balance → Potential confounding factors"
        else:
            return "Key metric for validation"
    
    metrics_data = [
        ("🔄", "Statistical Precision (CUPED)", f"{variance_reduction:.1f}%", "cuped", variance_reduction),
        ("🎯", "Model Targeting Accuracy", f"{model_accuracy:.1%}", "model", model_accuracy),
        ("📈", "Prediction Diversity", f"{uplift_range:.1%}", "range", uplift_range),
        ("✅", "Randomization Balance", f"{randomization_balance:.3f}", "randomization", randomization_balance),
    ]
    
    if len(metrics_data) == 3:
        col1, col2, col3 = st.columns(3)
        columns = [col1, col2, col3]
    else:
        col1, col2, col3, col4 = st.columns(4)
        columns = [col1, col2, col3, col4]
    
    for i, (icon, name, value, metric_type, raw_value) in enumerate(metrics_data):
        status_class = get_status_class(metric_type, raw_value)
        implication = get_statistical_implication(metric_type, raw_value)
        with columns[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-header">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-name">{name}</div>
                </div>
                <div class="metric-value {status_class}">{value}</div>
                <div class="metric-implication">{implication}</div>
            </div>
            """, unsafe_allow_html=True)

# Tab 5: Recommendations
with tab5:
    st.subheader("Summary & Recommendations")
    
    effect_size = float(df_f["global_effect_size"].iloc[0])
    added_bookings = effect_size * impact_scale
    revenue_impact = added_bookings * booking_value
    
    st.markdown("**Key Takeaways:**")
    st.markdown(f"""
    - **{pct(effect_size,1)} conversion lift** (statistically supported)
    - Uplift model highlights who benefits most → targeted rollout
    - Projected impact: **{added_bookings:,.0f} bookings** ≈ **${revenue_impact:,.0f}** annually
    """)
    
    st.markdown("---")
    
    st.markdown("**Strategic Recommendations:**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="padding: 12px; border-left: 4px solid #6c5ce7; background: rgba(108,92,231,0.06); border-radius: 8px;">
        <h4 style="margin: 0; color: #6c5ce7;">🎯 Rollout</h4>
        <p style="margin: 8px 0 0; font-size: 0.9rem;">Start with top 25%; expand to 50% after validation, then full rollout</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="padding: 12px; border-left: 4px solid #e84393; background: rgba(232,67,147,0.06); border-radius: 8px;">
        <h4 style="margin: 0; color: #e84393;">📊 Monitoring</h4>
        <p style="margin: 8px 0 0; font-size: 0.9rem;">Track conversion, calibration, and revenue per user weekly</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="padding: 12px; border-left: 4px solid #00b894; background: rgba(0,184,148,0.06); border-radius: 8px;">
        <h4 style="margin: 0; color: #00b894;">⚠️ Risk Mgmt</h4>
        <p style="margin: 8px 0 0; font-size: 0.9rem;">Set conversion drop alerts, keep 10% holdout, recalibrate weekly</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="padding: 12px; border-left: 4px solid #fd79a8; background: rgba(253,121,168,0.06); border-radius: 8px;">
        <h4 style="margin: 0; color: #fd79a8;">🔄 Iteration</h4>
        <p style="margin: 8px 0 0; font-size: 0.9rem;">A/B test variants; retrain uplift model monthly</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Reference Materials")
    
    with st.expander("Glossary"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Core Terms**
            - **Treatment / Control:** Two groups in a randomized experiment — the *treatment* group experiences the new interface, the *control* group sees the current one.
            - **Lift:** Difference in conversion rates between treatment and control; can be expressed in absolute percentage points or relative percent.
            - **Uplift:** Predicted *incremental* improvement in conversion for an individual if they were in treatment vs control, based on the model.
            - **CUPED:** "Controlled Pre-Experiment Data" — a variance-reduction technique that uses pre-treatment behavior to make treatment–control comparisons more precise.
            """)
        with col2:
            st.markdown("""
            **Statistical Terms**
            - **Confidence Interval (CI):** Range of values that likely contains the true effect; a 95% CI means that if the experiment is repeated many times, 95% of the intervals would contain the true effect.
            - **Standardized Mean Difference (SMD):** A unitless measure of how different two groups are; SMD < 0.1 is generally considered well-balanced.
            - **Propensity Score:** The probability of a user being assigned to treatment, estimated from observed covariates; near 0.5 across users suggests good randomization.
            - **Z-statistic:** Test statistic that measures how many standard errors the observed effect is from zero; higher absolute values mean stronger evidence against the null hypothesis.
            """)

    with st.expander("Technical Implementation"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Machine Learning**
            - **Approach:** X-Learner uplift modeling with separate models for treatment and control outcomes.
            - **Base Learners:** Random Forest regressors (balanced depth/leaf size to avoid overfitting).
            - **Feature Space:** 50+ behavioral and demographic features — including booking history, browsing activity, session metrics, and device type.
            - **Validation:** Time-based cross-validation to avoid leakage from future to past behavior.
            - **Clipped Scores:** Extreme uplift predictions are clipped to reduce the influence of outliers when targeting.
            """)
        with col2:
            st.markdown("""
            **Statistical Methods**
            - **Experiment Design:** Parallel-group A/B test with random assignment at the user level.
            - **Significance Testing:** Two-sample z-tests for proportions; CUPED applied before test to reduce variance.
            - **Variance Reduction:** CUPED yields % reduction in outcome variance compared to raw analysis.
            - **Confidence Intervals:** Calculated at the 95% level using normal approximation for proportions.
            - **Multiple Testing:** Controlled false-positive risk across subgroup analyses.
            """)


st.divider()

# Footer
st.caption("Built with Streamlit & Plotly | Causify Experimentation Platform © 2025")
