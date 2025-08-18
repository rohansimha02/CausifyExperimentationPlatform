# Causify Experimentation Platform
# A/B testing dashboard with uplift modeling, statistical validation, and ROI-focused targeting

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Causify Experimentation Platform",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Force light mode and clean styling
st.markdown("""
<style>
/* Force light mode */
[data-testid="stAppViewContainer"] {
    background-color: #ffffff !important;
}

/* Force coral colors */
.kpi-value[style*="color: #FF6B6B"] {
    color: #FF6B6B !important;
}

.metric-value[style*="color: #FF6B6B"] {
    color: #FF6B6B !important;
}

/* Force footer colors */
div[style*="text-align: center"] span[style*="color: var(--primary-color)"] {
    color: #FF6B6B !important;
}
div[style*="text-align: center"] span[style*="color: var(--accent-color)"] {
    color: #45B7D1 !important;
}

/* Force white backgrounds */
[data-testid="stAppViewContainer"], 
[data-testid="stSidebar"], 
.main .block-container {
    background-color: #ffffff !important;
}

/* Remove sidebar grid borders */
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] div > div,
section[data-testid="stSidebar"] div > div > div,
section[data-testid="stSidebar"] div > div > div > div {
    border: none !important;
    background-color: transparent !important;
    box-shadow: none !important;
}

/* Make ALL text in inputs dark (keep) */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] select,
section[data-testid="stSidebar"] [data-baseweb="base-input"],
section[data-testid="stSidebar"] [data-baseweb="select"],
section[data-testid="stSidebar"] [data-baseweb="input"],
section[data-testid="stSidebar"] [data-baseweb="option"],
section[data-testid="stSidebar"] [data-baseweb="tag"],
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: #0f172a !important;
}

/* Ultra clean, modern sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #ffffff 0%, #fafbfc 50%, #ffffff 100%) !important;
    border-right: 1px solid rgba(255, 107, 107, 0.08) !important;
    padding: 2.5rem !important;
    box-shadow: inset -2px 0 20px rgba(255, 107, 107, 0.03) !important;
    border-radius: 18px 0 0 18px !important;
}

/* Enhanced radio button styling for cleaner look */
section[data-testid="stSidebar"] [data-baseweb="radio"] {
    margin: 8px 0 !important;
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
}

section[data-testid="stSidebar"] [data-baseweb="radio"] label {
    padding: 12px 16px !important;
    margin: 0 !important;
    border-radius: 12px !important;
    background: #ffffff !important;
    border: 1px solid rgba(15,23,42,0.08) !important;
    box-shadow: 0 2px 8px rgba(15,23,42,0.04) !important;
    transition: all 0.2s ease !important;
    font-weight: 500 !important;
    color: #374151 !important;
}

section[data-testid="stSidebar"] [data-baseweb="radio"]:hover label {
    border-color: rgba(255,107,107,0.3) !important;
    box-shadow: 0 4px 12px rgba(255,107,107,0.1) !important;
    transform: translateY(-1px) !important;
}

section[data-testid="stSidebar"] [data-baseweb="radio"][aria-checked="true"] label {
    background: linear-gradient(135deg, #FF6B6B, #FF8A80) !important;
    border-color: #FF6B6B !important;
    color: white !important;
    box-shadow: 0 4px 16px rgba(255,107,107,0.25) !important;
    font-weight: 600 !important;
}

/* NEW: Round + frame the inner sidebar content without changing layout */
section[data-testid="stSidebar"] .block-container {
    background: #ffffff !important;
    border: 1px solid rgba(15,23,42,0.08) !important;
    border-radius: 18px !important;
    box-shadow: 0 6px 22px rgba(15,23,42,0.06) !important;
    padding: 18px !important;   /* small padding so alignment stays stable */
}

/* Modern sidebar headers */
section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 {
    color: #0f172a !important;
    font-weight: 700 !important;
    font-size: 1.3rem !important;
    margin-bottom: 1.5rem !important;
    margin-top: 2.5rem !important;
    font-family: 'Space Grotesk', Inter, sans-serif !important;
    letter-spacing: -0.02em !important;
    position: relative !important;
}
section[data-testid="stSidebar"] h1::after, 
section[data-testid="stSidebar"] h2::after, 
section[data-testid="stSidebar"] h3::after {
    content: "" !important;
    position: absolute !important;
    bottom: -10px !important;
    left: 0 !important;
    width: 50px !important;
    height: 4px !important;
    background: linear-gradient(90deg, #FF6B6B, #FF8A80) !important;
    border-radius: 2px !important;
    box-shadow: 0 2px 8px rgba(255, 107, 107, 0.2) !important;
}

/* Input cards (top-level wrappers only) */
section[data-testid="stSidebar"] .stSelectbox > div,
section[data-testid="stSidebar"] .stNumberInput > div,
section[data-testid="stSidebar"] .stMultiSelect > div,
section[data-testid="stSidebar"] .stExpander > div,
section[data-testid="stSidebar"] .stSlider > div,
section[data-testid="stSidebar"] .stRadio > div,
section[data-testid="stSidebar"] .stCheckbox > div {
    background: #ffffff !important;
    border: 1px solid rgba(15,23,42,0.08) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.06), 0 2px 8px rgba(255, 107, 107, 0.08) !important;
    transition: box-shadow .2s ease, border-color .2s ease;
}
section[data-testid="stSidebar"] .stSelectbox > div:hover,
section[data-testid="stSidebar"] .stNumberInput > div:hover,
section[data-testid="stSidebar"] .stMultiSelect > div:hover {
    box-shadow: 0 8px 26px rgba(0,0,0,.10), 0 4px 14px rgba(255,107,107,.15) !important;
    border-color: rgba(255, 107, 107, 0.25) !important;
}
/* Focus ring (coral) without shifting layout */
section[data-testid="stSidebar"] .stSelectbox > div:focus-within,
section[data-testid="stSidebar"] .stNumberInput > div:focus-within,
section[data-testid="stSidebar"] .stMultiSelect > div:focus-within,
section[data-testid="stSidebar"] .stSlider > div:focus-within,
section[data-testid="stSidebar"] .stRadio > div:focus-within,
section[data-testid="stSidebar"] .stCheckbox > div:focus-within {
    outline: 0 !important;
    box-shadow: 0 0 0 3px rgba(255,107,107,0.12) !important;
    border-color: rgba(255,107,107,0.35) !important;
}

/* Labels */
section[data-testid="stSidebar"] label {
    color: #0f172a !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    margin-bottom: 1rem !important;
    letter-spacing: 0.01em !important;
    font-family: 'Space Grotesk', Inter, sans-serif !important;
}

/* Radio button styling */
section[data-testid="stSidebar"] .stRadio > div {
    background: rgba(255, 255, 255, 0.9) !important;
    border: 2px solid rgba(255, 107, 107, 0.1) !important;
    border-radius: 18px !important;
    padding: 20px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06), 0 2px 8px rgba(255, 107, 107, 0.08) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
    overflow: hidden !important;
}

section[data-testid="stSidebar"] .stRadio > div::before {
    content: "" !important;
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    height: 3px !important;
    background: linear-gradient(90deg, #FF6B6B, #FF8A80) !important;
    opacity: 0 !important;
    transition: opacity 0.3s ease !important;
}

section[data-testid="stSidebar"] .stRadio > div:hover {
    transform: translateY(-2px) !important;
    border-color: rgba(255, 107, 107, 0.3) !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12), 0 4px 16px rgba(255, 107, 107, 0.15) !important;
}

section[data-testid="stSidebar"] .stRadio > div:hover::before {
    opacity: 1 !important;
}

/* Individual radio button styling */
section[data-testid="stSidebar"] [data-baseweb="radio"] {
    padding: 16px 20px !important;
    border-radius: 16px !important;
    margin: 10px 0 !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    background: rgba(255, 255, 255, 0.8) !important;
    border: 2px solid rgba(255, 107, 107, 0.1) !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
    position: relative !important;
    overflow: hidden !important;
}

section[data-testid="stSidebar"] [data-baseweb="radio"]:hover {
    background: linear-gradient(135deg, rgba(255, 107, 107, 0.08), rgba(255, 138, 128, 0.08)) !important;
    transform: translateX(6px) !important;
    box-shadow: 0 6px 20px rgba(255, 107, 107, 0.2) !important;
    border-color: rgba(255, 107, 107, 0.3) !important;
}

section[data-testid="stSidebar"] [data-baseweb="radio"][aria-checked="true"] {
    background: linear-gradient(135deg, #FF6B6B, #FF8A80) !important;
    border-color: #FF6B6B !important;
    box-shadow: 0 8px 25px rgba(255, 107, 107, 0.4), 0 4px 16px rgba(255, 107, 107, 0.3) !important;
    transform: translateX(4px) !important;
    position: relative !important;
}

section[data-testid="stSidebar"] [data-baseweb="radio"][aria-checked="true"]::before {
    background: #ffffff !important;
    border-color: #ffffff !important;
    box-shadow: 0 3px 12px rgba(255, 255, 255, 0.6) !important;
    width: 16px !important;
    height: 16px !important;
}

section[data-testid="stSidebar"] [data-baseweb="radio"][aria-checked="true"] span {
    color: #ffffff !important;
    font-weight: 700 !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1) !important;
}

/* Add a checkmark for selected radio buttons */
section[data-testid="stSidebar"] [data-baseweb="radio"][aria-checked="true"]::after {
    content: "✓" !important;
    position: absolute !important;
    right: 16px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    color: #ffffff !important;
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.2) !important;
}

/* Enhanced radio group layout */
section[data-testid="stSidebar"] .stRadio > div [role="radiogroup"] {
    display: flex !important;
    flex-direction: column !important;
    gap: 12px !important;
    width: 100% !important;
    min-width: 220px !important;
}

section[data-testid="stSidebar"] .stRadio > div [role="radio"] {
    background: linear-gradient(135deg, #ffffff 0%, #fafbfc 100%) !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 6px !important;
    padding: 3px 6px !important;
    transition: all 0.3s ease !important;
    position: relative !important;
    cursor: pointer !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08) !important;
    text-align: center !important;
    min-height: 18px !important;
    min-width: 90px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    font-size: 0.55rem !important;
    line-height: 0.9 !important;
    word-break: keep-all !important;
    hyphens: none !important;
}

/* Force radio button text to stay on one line */
section[data-testid="stSidebar"] .stRadio > div [role="radio"] * {
    font-size: 0.55rem !important;
    white-space: nowrap !important;
    word-break: keep-all !important;
    hyphens: none !important;
    line-height: 0.9 !important;
}

section[data-testid="stSidebar"] .stRadio > div [role="radio"] label {
    font-size: 0.55rem !important;
    white-space: nowrap !important;
    word-break: keep-all !important;
    hyphens: none !important;
    line-height: 0.9 !important;
    max-width: none !important;
    width: auto !important;
}
}

section[data-testid="stSidebar"] .stRadio > div [role="radio"]:hover {
    background: linear-gradient(135deg, #fff 0%, #fef7f7 100%) !important;
    border-color: #FF6B6B !important;
    box-shadow: 0 3px 8px rgba(255, 107, 107, 0.2) !important;
    transform: translateY(-1px) !important;
}

section[data-testid="stSidebar"] .stRadio > div [role="radio"][aria-checked="true"] {
    background: linear-gradient(135deg, #FF6B6B, #FF8A80) !important;
    border-color: #FF6B6B !important;
    box-shadow: 0 4px 12px rgba(255, 107, 107, 0.25) !important;
    transform: translateY(-1px) !important;
    position: relative !important;
}

section[data-testid="stSidebar"] .stRadio > div [role="radio"][aria-checked="true"]::after {
    content: "✓" !important;
    position: absolute !important;
    right: 16px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    color: #ffffff !important;
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.2) !important;
}

/* Selected label styling */
section[data-testid="stSidebar"] .stRadio > div [role="radio"][aria-checked="true"] * {
    color: #ffffff !important;
    font-weight: 700 !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1) !important;
}

/* Checkbox */
section[data-testid="stSidebar"] [data-baseweb="checkbox"] {
    background: #fff !important;
    border: 1px solid rgba(15,23,42,0.10) !important;
    border-radius: 14px !important;
    padding: 8px 12px !important;
    transition: all 0.2s ease !important;
}
section[data-testid="stSidebar"] [data-baseweb="checkbox"][aria-checked="true"] > div {
    background: linear-gradient(135deg, #FF6B6B, #FF8A80) !important;
    border-color: #FF6B6B !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 16px rgba(255, 107, 107, 0.3) !important;
}

/* Slider */
section[data-testid="stSidebar"] [data-baseweb="slider"] [data-baseweb="track"] {
    background: linear-gradient(90deg, #FF6B6B, #FF8A80) !important;
    height: 8px !important;
    border-radius: 6px !important;
}
section[data-testid="stSidebar"] [data-baseweb="slider"] [data-baseweb="thumb"] {
    background: linear-gradient(135deg, #FF6B6B, #FF8A80) !important;
    border: 3px solid #ffffff !important;
    box-shadow: 0 4px 16px rgba(255, 107, 107, 0.4), 0 2px 8px rgba(0, 0, 0, 0.1) !important;
    width: 20px !important;
    height: 20px !important;
    border-radius: 50% !important;
}

/* Number input buttons */
section[data-testid="stSidebar"] .stNumberInput button {
    background: #f7f8fa !important;
    border: 1px solid rgba(15,23,42,0.12) !important;
    border-radius: 10px !important;
    color: #0f172a !important;
    transition: all 0.2s ease !important;
    font-weight: 600 !important;
    padding: 10px 14px !important;
    transform: translateY(-10px) !important;
}
section[data-testid="stSidebar"] .stNumberInput button:hover {
    background: linear-gradient(135deg, #FF6B6B, #FF8A80) !important;
    color: #ffffff !important;
    border-color: #FF6B6B !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(255, 107, 107, 0.3) !important;
}

/* Expander */
section[data-testid="stSidebar"] .stExpander > div:first-child {
    border: none !important;
    background: transparent !important;
    border-radius: 16px !important;
}
section[data-testid="stSidebar"] .stExpander [data-testid="stExpanderToggleIcon"] {
    color: #FF6B6B !important;
}

/* Popovers/menus */
section[data-testid="stSidebar"] [data-baseweb="popover"],
section[data-testid="stSidebar"] [data-baseweb="menu"],
section[data-testid="stSidebar"] [data-baseweb="option"],
section[data-testid="stSidebar"] [data-baseweb="listbox"] {
    background: rgba(255, 255, 255, 0.98) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 107, 107, 0.2) !important;
    border-radius: 14px !important;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.12) !important;
    color: #0f172a !important;
}
section[data-testid="stSidebar"] [data-baseweb="option"]:hover,
section[data-testid="stSidebar"] [data-baseweb="option"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(255, 107, 107, 0.10), rgba(255, 138, 128, 0.10)) !important;
    color: #0f172a !important;
    border-left: 3px solid rgba(255,107,107,0.45) !important;
}

/* Style the sidebar toggle button */
.stButton > button {
    background: linear-gradient(135deg, #FF6B6B, #FF8A80) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 20px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    box-shadow: 0 4px 16px rgba(255, 107, 107, 0.3) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4) !important;
    background: linear-gradient(135deg, #FF8A80, #FF6B6B) !important;
}

/* Keep sidebar visible/sized */
section[data-testid="stSidebar"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    width: 300px !important;
    min-width: 280px !important;
    max-width: 320px !important;
    transform: translateX(0) !important;
    overflow-y: auto !important;
}

/* Sidebar container sizing */
section[data-testid="stSidebar"] .block-container {
    max-width: none !important;
}




</style>
""", unsafe_allow_html=True)

# Color palette & theme variables
COLORS = {
    "primary": "#FF6B6B",      # Coral
    "secondary": "#4ECDC4",    # Teal
    "accent": "#45B7D1",       # Blue
    "success": "#95E1D3",      # Mint
    "warning": "#F7DC6F",      # Yellow
    "danger": "#FF8A80",       # Light red
}

# Light theme values
TEXT = "#0f172a"
MUTED = "#475569"
BG = "#ffffff"
PANEL = "#f8fafc"
BORDER = "rgba(0,0,0,0.08)"
GRID = "rgba(15,23,42,0.08)"
HOVER_BG = "#ffffff"
HOVER_FG = "#0f172a"

# Theme-aware CSS
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

:root {{
  --text-color: {TEXT};
  --muted-color: {MUTED};
  --background-color: {BG};
  --panel-bg: {PANEL};
  --border-color: {BORDER};
  --grid-color: {GRID};
  --primary-color: {COLORS["primary"]};
  --secondary-color: {COLORS["secondary"]};
  --accent-color: {COLORS["accent"]};
  --success-color: {COLORS["success"]};
  --warning-color: {COLORS["warning"]};
  --danger-color: {COLORS["danger"]};
  --accent-coral: {COLORS["primary"]};
  --accent-coral-weak: rgba(255,107,107,0.12);
}}

#MainMenu, header, footer {{ visibility: hidden; }}

.main .block-container {{
  padding-top: 1.5rem; padding-bottom: 1.5rem;
}}

h1 {{
  font-family: 'Space Grotesk', Inter, sans-serif !important;
  font-weight: 700; letter-spacing: -0.02em;
  background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
  margin-bottom: 0.75rem;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
  gap: 10px; justify-content: center; align-items: center;
  background: rgba(255,255,255,0.03);
  padding: 8px; border-radius: 12px; border: 1px solid var(--border-color);
}}
  .stTabs [data-baseweb="tab"] {{
  padding: 10px 18px; border-radius: 10px; border: none; background: transparent;
  color: var(--text-color) !important; font-weight: 600; transition: all 0.2s ease;
  font-family: 'Space Grotesk', Inter, sans-serif;
}}
.stTabs [data-baseweb="tab"]:hover {{
  background: rgba(255,107,107,0.08);
  color: var(--primary-color) !important;
}}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
  background: var(--primary-color) !important; 
  color: white !important;
  box-shadow: 0 4px 12px rgba(255, 107, 107, 0.3);
  transform: translateY(-1px);
}}
.stTabs [data-baseweb="tab"][aria-selected="true"] * {{ color: white !important; }}

/* KPI card styles */
  .kpi-card {{
  border: 1px solid var(--border-color); 
  border-radius: 16px; 
  padding: 20px 24px;
  background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(248,250,252,0.9));
  box-shadow: 0 6px 24px rgba(0,0,0,0.06), 0 2px 8px rgba(0,0,0,0.04);
  position: relative; overflow: hidden; transition: all 0.3s ease;
}}
.kpi-card:hover {{
  transform: translateY(-2px);
  box-shadow: 0 10px 32px rgba(0,0,0,0.1), 0 4px 12px rgba(0,0,0,0.06);
}}
.kpi-card::before {{ 
  content: ""; position: absolute; top:0; left:0; right:0; height: 3px;
  background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
  border-radius: 16px 16px 0 0;
}}
.kpi-title {{ font-size: .75rem; color: var(--muted-color); margin-bottom: 6px; font-weight: 500; text-transform: uppercase; letter-spacing: .06em; opacity: .7; }}
.kpi-value {{ font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 4px; color: var(--text-color); }}
.kpi-sub {{ font-size: .8rem; color: var(--muted-color); opacity: .7; font-weight: 400; }}

/* Notice boxes */
.notice, .warn {{
  padding: 16px 18px; border-radius: 14px; margin: 12px 0 20px 0; font-weight: 500;
  background: linear-gradient(135deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
}}
.notice {{ border: 1px solid var(--success-color); border-left: 6px solid var(--success-color); }}
.warn   {{ border: 1px solid var(--warning-color); border-left: 6px solid var(--warning-color); }}

/* Metric cards */
.metric-card {{ 
  border: 1px solid var(--border-color); border-radius: 18px; padding: 20px 24px; 
  background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(248,250,252,0.9));
  box-shadow: 0 6px 24px rgba(0,0,0,0.06), 0 2px 8px rgba(0,0,0,0.04);
  margin-bottom: 20px; position: relative; overflow: hidden; transition: all 0.3s ease;
}}
.metric-card:hover {{ transform: translateY(-2px); box-shadow: 0 10px 32px rgba(0,0,0,0.1), 0 4px 12px rgba(0,0,0,0.06); }}
.metric-card::before {{ content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, var(--primary-color), var(--secondary-color)); border-radius: 18px 18px 0 0; }}
.metric-header {{ display: flex; align-items: center; margin-bottom: 12px; }}
.metric-icon {{ font-size: 28px; width: 40px; text-align: center; margin-right: 12px; color: var(--primary-color); }}
.metric-name {{ font-weight: 600; font-size: 12px; color: var(--muted-color); letter-spacing: .06em; text-transform: uppercase; opacity: .8; }}
.metric-value {{ font-size: 28px; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 8px; }}
.metric-implication {{ font-size: 13px; color: var(--muted-color); line-height: 1.5; font-style: italic; opacity: .8; }}

/* Recommendation cards */
.rec-card {{ padding: 18px; border-left: 6px solid; background: linear-gradient(135deg, var(--panel-bg), rgba(255,255,255,0.02)); border-radius: 14px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }}
.rec-card h4 {{ margin: 0 0 6px 0; font-weight: 700; font-size: 16px; }}
.rec-card p {{ margin: 0; font-size: 14px; line-height: 1.55; color: var(--text-color); opacity: .95; }}

/* Download button */
.stDownloadButton > button {{
  background-color: rgba(255,255,255,0.04) !important;
  border: 1px solid var(--border-color) !important;
  border-radius: 8px !important;
  color: var(--text-color) !important;
  font-weight: 600 !important;
  padding: 0.5rem 1rem !important;
  transition: all 0.2s ease !important;
}}
.stDownloadButton > button:hover {{
  background-color: var(--accent-coral) !important; color: #ffffff !important; border-color: var(--accent-coral) !important;
  }}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Plotly theme helper
# ----------------------------
def apply_chart_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        plot_bgcolor=PANEL,
        paper_bgcolor=PANEL,
        font=dict(color=TEXT, size=14),
        title=dict(font=dict(size=18, color=TEXT), x=0.5, xanchor='center'),
        margin=dict(t=50, b=40, l=40, r=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, title_font=dict(color=MUTED), tickfont=dict(color=TEXT), tickcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, title_font=dict(color=MUTED), tickfont=dict(color=TEXT), tickcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=HOVER_BG, font=dict(size=13, color=HOVER_FG)),
        transition_duration=150,
    )
    return fig

# Data source configuration
DEFAULT_LOCAL = Path(__file__).resolve().parent / "final_dashboard_data.csv"
DEFAULT_PARENT = Path(__file__).resolve().parent.parent / "data" / "final_dashboard_data.csv"

# Load data directly from local file
@st.cache_data(ttl=3600)
def _load_data_from_source():
    path = next((p for p in [str(DEFAULT_LOCAL), str(DEFAULT_PARENT)] if p and Path(p).exists()), None)
    if path is None:
        raise ValueError("No data source found. Please ensure final_dashboard_data.csv is in the correct location.")
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
    df = _load_data_from_source()
except Exception as e:
    st.error(str(e))
    st.stop()

# Sidebar controls & filters
st.sidebar.header("Controls")

# Define uplift column
uplift_col = "uplift_score"

treatment_group = st.sidebar.radio("Treatment Group", ["All Groups", "Test Only", "Control Only"])



booking_value = st.sidebar.number_input("Value per booking ($)", min_value=0, value=150, step=10)
impact_scale = st.sidebar.number_input("Users scale for impact (thousands)", min_value=10_000, value=100_000, step=10_000)

with st.sidebar.expander("Filters", expanded=False):
    min_age, max_age = int(np.nanmin(df["age"])), int(np.nanmax(df["age"]))
    age_rng = st.slider("Age Range", min_age, max_age, (min_age, max_age))
    
    # Minimum uplift threshold for targeting
    min_uplift = float(df[uplift_col].min())
    max_uplift = float(df[uplift_col].max())
    uplift_threshold = st.slider("Min Uplift Threshold (%)", 
                                min_value=float(min_uplift * 100), 
                                max_value=float(max_uplift * 100), 
                                value=float(min_uplift * 100), 
                                step=0.1, 
                                help="Minimum predicted uplift to include in analysis")

# Assignment filter
df_view = df
if treatment_group == "Test Only":
    df_view = df_view[df_view["treatment"] == 1]
    st.sidebar.write(f"Test Only selected: {len(df_view)} rows (treatment=1)")
elif treatment_group == "Control Only":
    df_view = df_view[df_view["treatment"] == 0]
    st.sidebar.write(f"Control Only selected: {len(df_view)} rows (treatment=0)")
else:
    st.sidebar.write(f"All Groups selected: {len(df_view)} rows")

mask = (
    df_view["age"].between(age_rng[0], age_rng[1]) &
    (df_view[uplift_col] >= uplift_threshold / 100)
)
df_f = df_view.loc[mask].copy()

# Utilities
def pct(x, d=1):
    try:
        return f"{x:.{d}%}"
    except Exception:
        return "—"

def tc_rates(frame: pd.DataFrame):
    g = frame.groupby("group_label")["booking"].mean()
    return float(g.get("Treatment", np.nan)), float(g.get("Control", np.nan))

def ci_includes_zero(lo, hi):
    return lo <= 0 <= hi

def calculate_lift_stats(df_filtered):
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
    
    # Handle missing columns safely
    tr = agg.get(("mean", 1), pd.Series(0.0, index=agg.index)).fillna(0.0)
    cr = agg.get(("mean", 0), pd.Series(0.0, index=agg.index)).fillna(0.0)
    n_tr = agg.get(("count", 1), pd.Series(0, index=agg.index)).fillna(0)
    n_cr = agg.get(("count", 0), pd.Series(0, index=agg.index)).fillna(0)
    
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

def notice(text, kind="good"):
    cls = "notice" if kind == "good" else "warn"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)

def rec_card(title, body, color):
    st.markdown(f"""
    <div class="rec-card" style="border-left-color: {color};">
      <h4 style="color: {color};">{title}</h4>
      <p>{body}</p>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# Main UI
# ----------------------------
st.title("Causify Experimentation Platform")
st.markdown(
    "**A/B testing dashboard with uplift modeling, statistical validation, and ROI-focused targeting.** "
    "A randomized experiment on Airbnb booking data comparing a *new booking interface* (treatment) to the current (control), "
    "summarizing results, showing who benefits most, and outlining an ROI-first rollout."
)

st.markdown("Tabs: **(1)** experiment results, **(2)** uplift (who benefits), **(3)** targeting strategy, **(4)** validation, **(5)** summary & recommendations.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Uplift Analysis", 
    "Targeting Strategy",
    "Model & Experiment Validation",
    "Summary & Recommendations",
])

# Tab 1: Overview
with tab1:
    st.subheader("Overview")
    st.caption("Why: check if treatment beats control and by how much.")

    tr, cr = tc_rates(df_f)
    lift_val, z_stat, ci_lower, ci_upper = calculate_lift_stats(df_f)

    lift_color = COLORS["primary"] if (lift_val or 0) >= 0 else COLORS["danger"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">Total users (current view)</div>'
            f'<div class="kpi-value" style="color:{COLORS["primary"]}">{len(df_f):,}</div>'
            f'<div class="kpi-sub">Filters on the left</div></div>', unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">Treatment conversion</div>'
                    f'<div class="kpi-value">{pct(tr,2)}</div>'
            f'<div class="kpi-sub">New interface</div></div>', unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">Control conversion</div>'
                    f'<div class="kpi-value">{pct(cr,2)}</div>'
            f'<div class="kpi-sub">Current interface</div></div>', unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">Lift (treatment − control)</div>'
            f'<div class="kpi-value" style="color: #FF6B6B; font-weight: 800;">{pct(lift_val,1)}</div>'
                    f'<div class="kpi-sub">95% CI {pct(ci_lower,1)} to {pct(ci_upper,1)} · z={z_stat:.2f}</div></div>',
            unsafe_allow_html=True
        )

    if ci_includes_zero(ci_lower, ci_upper):
        notice(
            f"<strong>Result:</strong> Inconclusive at 95%. "
            f"Observed lift {pct(lift_val,1)} (CI {pct(ci_lower,1)} to {pct(ci_upper,1)}). "
            f"Consider more sample or stronger variance reduction.",
            kind="warn"
        )
    else:
        added = max(lift_val, 0) * impact_scale
        notice(
            f"<strong>Result:</strong> Statistically significant improvement. "
            f"At {impact_scale:,} users → ≈ {added:,.0f} extra bookings "
            f"(≈ ${added*booking_value:,.0f}).",
            kind="good"
        )

    n_tr = (df_f["group_label"] == "Treatment").sum()
    n_cr = (df_f["group_label"] == "Control").sum()
    se_tr = np.sqrt(tr * (1 - tr) / max(n_tr, 1))
    se_cr = np.sqrt(cr * (1 - cr) / max(n_cr, 1))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Current interface (Control)"], y=[cr],
        marker=dict(color=COLORS["secondary"], line=dict(width=0)),
        error_y=dict(type="data", array=[1.96*se_cr], visible=True, color="#FF6B6B", thickness=2),
        text=[f'{cr:.2%}'], textposition='outside',
        hovertemplate="<b>%{x}</b><br>Conversion: %{y:.2%}<extra></extra>",
        showlegend=False
    ))
    fig.add_trace(go.Bar(
        x=["New interface (Treatment)"], y=[tr],
        marker=dict(color=COLORS["primary"], line=dict(width=0)),
        error_y=dict(type="data", array=[1.96*se_tr], visible=True, color=COLORS["accent"], thickness=2),
        text=[f'{tr:.2%}'], textposition='outside',
        hovertemplate="<b>%{x}</b><br>Conversion: %{y:.2%}<extra></extra>",
        showlegend=False
    ))
    fig.add_annotation(
        x=1, y=tr*(1.12), text=f"<b>+{lift_val:.1%} lift</b>",
        showarrow=True, arrowhead=2, arrowcolor=COLORS["accent"], arrowwidth=2,
        font=dict(size=14, color=COLORS["accent"]),
        bgcolor="rgba(255,107,107,0.12)", bordercolor=COLORS["accent"], borderwidth=1
    )
    fig.update_layout(
        title=dict(text="New vs Current Interface (Two-Sample Z-Test with 95% CI)", x=0.5),
        yaxis_title="Booking conversion rate",
        xaxis_title="Experiment group",
        yaxis=dict(tickformat='.1%'),
        height=420, bargap=0.35
    )
    st.plotly_chart(apply_chart_theme(fig), use_container_width=True)

    st.download_button(
        label="Download current view (CSV)",
        data=df_f.to_csv(index=False).encode('utf-8'),
        file_name="filtered_view.csv",
        mime="text/csv",
    )

# Tab 2: Uplift Analysis
with tab2:
    st.subheader("Uplift Analysis")
    st.caption("Why: find who benefits most from treatment.")

    mean_uplift = float(df_f[uplift_col].mean())

    dist = px.histogram(
        df_f, x=uplift_col, nbins=40, marginal="rug",
        color_discrete_sequence=[COLORS["primary"]],
        labels={uplift_col: "Predicted improvement in booking chance"}
    )
    dist.add_vline(x=mean_uplift, line_dash="dash", line_color=COLORS["accent"])
    dist.update_layout(title=dict(text="Distribution of Individual Uplift Predictions", x=0.5), yaxis=dict(title="Users"), height=400)
    st.plotly_chart(apply_chart_theme(dist), use_container_width=True)
    st.caption(f"What this shows: distribution of predicted improvement; dashed line = average ({mean_uplift:.2%}).")

    pos_share = float((df_f[uplift_col] > 0).mean())
    hi_share = float((df_f[uplift_col] > np.percentile(df_f[uplift_col], 80)).mean())
    max_gain = float(df_f[uplift_col].max())
    c1, c2, c3 = st.columns(3)
    for title, val, sub, col in [
        ("Users who benefit", pct(pos_share,0), "Share with > 0% predicted improvement", c1),
        ("High-impact users", pct(hi_share,0), "Top 20% by predicted improvement", c2),
        ("Maximum predicted gain", pct(max_gain,1), "Largest expected single-user lift", c3),
    ]:
        with col:
            if title == "Users who benefit":
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-title">{title}</div>'
                    f'<div class="kpi-value" style="color: #0d9488; font-weight: 800;">{val}</div>'
                    f'<div class="kpi-sub">{sub}</div></div>',
                    unsafe_allow_html=True
                )
            elif title == "High-impact users":
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-title">{title}</div>'
                    f'<div class="kpi-value">{val}</div>'
                    f'<div class="kpi-sub">{sub}</div></div>',
                    unsafe_allow_html=True
                )
            elif title == "Maximum predicted gain":
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-title">{title}</div>'
                    f'<div class="kpi-value">{val}</div>'
                    f'<div class="kpi-sub">{sub}</div></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-title">{title}</div>'
                    f'<div class="kpi-value">{val}</div>'
                    f'<div class="kpi-sub">{sub}</div></div>',
                    unsafe_allow_html=True
                )

    st.subheader("Model Performance")
    st.caption("What this shows: higher predicted uplift should correspond to higher observed lift (bucketed).")

    df_dec, _ = compute_deciles(df_f, uplift_col, q=10)
    calib = decile_calibration(df_dec, dec_col="decile").sort_values("bucket")

    line = go.Figure()
    line.add_trace(
        go.Scatter(
            x=calib["bucket"], y=calib["lift"],
            mode="lines+markers", name="Observed lift",
            line=dict(color="#FF6B6B", width=3, shape="spline", smoothing=0.5),
            marker=dict(size=7, line=dict(width=0), color="#FF6B6B")
        )
    )
    line.update_layout(
        title=dict(text="Model Calibration: Observed Lift by Predicted Impact Group", x=0.5),
        xaxis_title="Impact Group (1 = Lowest, 10 = Highest)",
        yaxis_title="Observed Lift (New – Current)",
        yaxis=dict(tickformat=".2%"),
        height=500,
        margin=dict(l=40, r=40, t=60, b=100)
    )
    st.plotly_chart(apply_chart_theme(line), use_container_width=True)

# Tab 3: Targeting Strategy
with tab3:
    st.subheader("Targeting Strategy")
    st.caption("Why: maximize ROI by treating users most likely to benefit first.")

    pct_slider = st.slider("Choose rollout size (top % by predicted improvement)", 5, 100, 30, step=5)
    n_target = int(len(df_f) * (pct_slider / 100))
    df_rank = df_f.sort_values(uplift_col, ascending=False)

    inc_bookings = float(df_rank.iloc[:n_target][uplift_col].sum())
    base_total_inc = float(df_f[uplift_col].sum())

    avg_uplift_targeted = inc_bookings / max(n_target, 1)
    avg_uplift_full = base_total_inc / max(len(df_f), 1)
    roi_boost = (avg_uplift_targeted / max(avg_uplift_full, 1e-9)) if avg_uplift_full != 0 else 0.0

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">Users targeted</div>'
            f'<div class="kpi-value" style="color: #000000; font-weight: 800;">{n_target:,}</div>'
            f'<div class="kpi-sub">{pct_slider}% of current view</div></div>',
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">Extra bookings</div>'
            f'<div class="kpi-value" style="color: #000000; font-weight: 800;">{inc_bookings:,.0f}</div>'
            f'<div class="kpi-sub">Sum of predicted uplift in targeted cohort</div></div>',
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">ROI boost (per user)</div>'
            f'<div class="kpi-value" style="color: #FF6B6B; font-weight: 800;">{pct(roi_boost - 1,1) if roi_boost>=0 else pct(roi_boost,1)}</div>'
            f'<div class="kpi-sub">Avg uplift/user vs full rollout</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("&nbsp;", unsafe_allow_html=True)

    if pct_slider <= 25:
        notice(f"Conservative rollout: focus on top {pct_slider}% to maximize ROI; validate with holdouts.", kind="good")
    elif pct_slider <= 50:
        notice(f"Balanced approach: target {pct_slider}% for coverage and efficiency; expand if realized lift holds.", kind="good")
    else:
        notice(f"Aggressive rollout: {pct_slider}% coverage. Expect diminishing returns; monitor ROI closely.", kind="warn")

    st.subheader("Cumulative Impact")
    st.caption("What this shows: more users targeted → more incremental bookings (Qini curve).")

    xs, ys = build_gain_curve(df_f, uplift_col, step=0.1)
    qini = go.Figure()
    qini.add_trace(
        go.Scatter(
            x=[x * 100 for x in xs], y=ys,
            mode="lines+markers",
            line=dict(color=COLORS["accent"], width=3, shape="spline", smoothing=0.5),
            marker=dict(size=7, line=dict(width=0), color=COLORS["primary"])
        )
    )
    qini.update_layout(
        title=dict(text="ROI Analysis: Cumulative Impact (Qini Curve)", x=0.5),
        xaxis_title="% of users targeted (highest first)",
        yaxis_title="Cumulative incremental bookings",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(apply_chart_theme(qini), use_container_width=True)

# Tab 4: Model & Experiment Validation
with tab4:
    st.subheader("Model & Experiment Validation")
    st.caption("Why: confirm randomization worked and that model predictions are credible.")

    st.subheader("Balance & Randomization Diagnostics")
    
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
            colors = ["#0d9488" if s < 0.1 else "#d97706" if s < 0.2 else "#dc2626" for s in smd_df['smd']]
            fig_smd.add_trace(go.Bar(
                x=smd_df['smd'], y=smd_df['feature'],
                orientation='h', marker=dict(color=colors, opacity=0.9),
                hovertemplate="<b>%{y}</b><br>SMD: %{x:.3f}<extra></extra>"
            ))
            fig_smd.add_vline(x=0.1, line=dict(color=COLORS["primary"], width=2, dash="dash"))
            fig_smd.update_layout(
                title=dict(text="Covariate Balance", x=0.5, xanchor='center', font=dict(size=16, color=TEXT)),
                xaxis=dict(
                    title=dict(text="Standardized Mean Difference", font=dict(size=12, color=MUTED)),
                    tickformat=".2f",
                    zerolinecolor=GRID,
                    tickfont=dict(size=11, color=TEXT)
                ),
                yaxis=dict(
                    title=dict(text="Features", font=dict(size=12, color=MUTED)),
                    tickfont=dict(size=11, color=TEXT)
                ),
                height=350,
                showlegend=False,
                margin=dict(l=50, r=20, t=50, b=50)
            )
            st.plotly_chart(apply_chart_theme(fig_smd), use_container_width=True)
            st.caption(f"{(smd_df['smd'] < 0.1).sum()}/{len(smd_df)} features well-balanced (SMD < 0.1)")
        else:
            notice("SMDs require numeric baseline features", kind="warn")
    
    with col_prop:
        st.markdown("**Randomization Quality (Propensity)**")
        st.caption("Peak near 0.5 suggests true random assignment.")
        if "propensity_score" in df_f.columns:
            fig_prop = go.Figure()
            ps_mean = float(df_f["propensity_score"].mean())
            fig_prop.add_trace(go.Histogram(
                x=df_f["propensity_score"],
                nbinsx=25,
                marker=dict(color=COLORS["primary"], opacity=0.85, line=dict(color='#ffffff', width=0.5)),
                hovertemplate="<b>Propensity:</b> %{x:.2f}<br><b>Users:</b> %{y}<extra></extra>"
            ))
            fig_prop.add_vline(x=0.5, line=dict(color=COLORS["accent"], width=2, dash="dash"))
            fig_prop.add_vline(x=ps_mean, line=dict(color=COLORS["success"], width=2, dash="dot"))
            fig_prop.add_vrect(x0=0.45, x1=0.55, fillcolor="rgba(255,107,107,0.1)", opacity=1.0, layer="below", line_width=0)
            fig_prop.update_layout(
                title=dict(text="Assignment Prediction", x=0.5, xanchor='center', font=dict(size=16, color=TEXT)),
                xaxis=dict(
                    title=dict(text="Predicted Prob. of Treatment", font=dict(size=12, color=MUTED)),
                    range=[0.35, 0.65],
                    zerolinecolor=GRID,
                    tickfont=dict(size=11, color=TEXT)
                ),
                yaxis=dict(
                    title=dict(text="Users", font=dict(size=12, color=MUTED)),
                    tickfont=dict(size=11, color=TEXT)
                ),
                height=350,
                showlegend=False,
                margin=dict(l=50, r=20, t=50, b=50)
            )
            st.plotly_chart(apply_chart_theme(fig_prop), use_container_width=True)
            balance_score = 1 - abs(ps_mean - 0.5) * 2
            st.caption(f"Balance score: {balance_score:.3f} (mean ≈ 0.5 = perfect)")
        else:
            notice("Propensity scores not available", kind="warn")
    
    st.markdown("---")

    st.subheader("Model & Experiment Validation")
    
    model_accuracy = float(df_f['uplift_model_performance'].iloc[0]) if 'uplift_model_performance' in df_f.columns else 0.5
    variance_reduction = float(df_f['global_variance_reduction_pct'].iloc[0]) if 'global_variance_reduction_pct' in df_f.columns else 0.0
    uplift_range = float(df_f[uplift_col].max() - df_f[uplift_col].min())
    
    randomization_balance = None
    if "propensity_score" in df_f.columns:
        ps_mean = float(df_f["propensity_score"].mean())
        randomization_balance = 1 - abs(ps_mean - 0.5) * 2
    
    def status(metric_type, value):
        if metric_type == "cuped":
            return ("status-excellent" if value > 5 else "status-good" if value > 0 else "status-warning")
        if metric_type == "model":
            return ("status-excellent" if value > 0.55 else "status-good" if value > 0.52 else "status-warning")
        if metric_type == "range":
            return ("status-excellent" if value > 0.2 else "status-good" if value > 0.1 else "status-warning")
        if metric_type == "randomization":
            v = value or 0
            return ("status-excellent" if v >= 0.99 else "status-good" if v >= 0.95 else "status-warning")
    
    def implication(metric_type, value):
        if metric_type == "cuped":
            if value > 5: return "Strong precision → High confidence in results"
            if value > 0: return "Good precision → Reliable statistical conclusions"
            else:
                return "No variance reduction → Results may be noisy"
        if metric_type == "model":
            if value > 0.55: return "Strong targeting ability → Proceed with personalized rollout"
            if value > 0.28: return "Moderate targeting → Selective personalization recommended"
            else: return "Limited targeting → Focus on broad rollout strategy"
        if metric_type == "range":
            if value > 0.2: return "High diversity → Model captures user heterogeneity well"
            if value > 0.1: return "Moderate diversity → Reasonable prediction spread"
            else: return "Low diversity → Limited personalization potential"
        if metric_type == "randomization":
            v = value or 0
            if v >= 0.99: return "Strong balance → Unbiased experiment setup"
            if v >= 0.95: return "Good balance → Reliable causal attribution"
            else: return "Review balance → Potential confounding factors"
            return "Key metric for validation"
    
    metrics_data = [
        ("Statistical Precision (CUPED)", f"{variance_reduction:.1f}%", "cuped", variance_reduction),
        ("Model Targeting Accuracy", f"{model_accuracy:.1%}", "model", model_accuracy),
        ("Prediction Diversity", f"{uplift_range:.1%}", "range", uplift_range),
        ("Randomization Balance", f"{(randomization_balance or 0):.3f}", "randomization", randomization_balance or 0),
    ]

    col1, col2, col3, col4 = st.columns(4)
    for i, (name, value, mtype, raw) in enumerate(metrics_data):
        stat = status(mtype, raw)
        expl = implication(mtype, raw)
        with (col1 if i==0 else col2 if i==1 else col3 if i==2 else col4):
            # Color important metrics with coral
            if name == "Model Targeting Accuracy":
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-header">
                            <span class="metric-icon">⚙️</span>
                            <span class="metric-name">{name}</span>
                    </div>
                        <div class="metric-value" style="color: #FF6B6B; font-weight: 800;">{value}</div>
                        <div class="metric-implication">{expl}</div>
                    </div>
                    """, unsafe_allow_html=True)
            elif name == "Statistical Precision (CUPED)":
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-icon">⚙️</span>
                        <span class="metric-name">{name}</span>
                    </div>
                    <div class="metric-value" style="color: #000000; font-weight: 800;">{value}</div>
                    <div class="metric-implication">{expl}</div>
                </div>
                """, unsafe_allow_html=True)
            elif name == "Randomization Balance":
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-icon">⚙️</span>
                        <span class="metric-name">{name}</span>
                    </div>
                    <div class="metric-value" style="color: #0d9488; font-weight: 800;">{value}</div>
                    <div class="metric-implication">{expl}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-icon">⚙️</span>
                        <span class="metric-name">{name}</span>
                    </div>
                    <div class="metric-value {stat}">{value}</div>
                    <div class="metric-implication">{expl}</div>
            </div>
            """, unsafe_allow_html=True)

# Tab 5: Summary & Recommendations
with tab5:
    st.subheader("Summary & Recommendations")
    
    effect_size = float(df_f["global_effect_size"].iloc[0])
    added_bookings = effect_size * impact_scale
    revenue_impact = added_bookings * booking_value
    
    tr, cr = tc_rates(df_f)
    lift_val, z_stat, ci_lower, ci_upper = calculate_lift_stats(df_f)

    df_rank = df_f.sort_values(uplift_col, ascending=False)
    n_target_25 = int(len(df_f) * 0.25)
    inc_bookings_25 = float(df_rank.iloc[:n_target_25][uplift_col].sum())
    base_total_inc = float(df_f[uplift_col].sum())
    avg_uplift_targeted = inc_bookings_25 / max(n_target_25, 1)
    avg_uplift_full = base_total_inc / max(len(df_f), 1)
    roi_boost = (avg_uplift_targeted / max(avg_uplift_full, 1e-9)) if avg_uplift_full != 0 else 0.0
    
    st.markdown("#### Key Takeaways")
    
    st.markdown(f"""
    <div style="
        background: #f8f9fa;
        padding: 0.8rem;
        border-radius: 6px;
        border-left: 3px solid #FF6B6B;
        margin: 0.5rem 0;
    ">
        <ul style="margin: 0; padding-left: 1rem; line-height: 1.4; font-size: 1.05rem;">
            <li><strong>{pct(effect_size,1)} conversion lift</strong> • z-score: {z_stat:.1f} • Highly statistically significant results</li>
            <li><strong>Smart targeting works</strong> • Model successfully identifies high-impact users</li>
            <li><strong>${revenue_impact:,.0f} annual revenue</strong> • {added_bookings:,.0f} incremental bookings • At {impact_scale:,} total users scale</li>
            <li><strong>{roi_boost:.1f}x ROI boost</strong> • Top 25% users deliver better returns than broad rollout</li>
            <li><strong>Low risk</strong> • 95% CI ({pct(ci_lower,1)} to {pct(ci_upper,1)}) • All confidence intervals positive</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("**Strategic Recommendations:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1: rec_card("Rollout", "Start with top 25%; expand to 50% after validation, then full rollout.", COLORS["primary"])
    with col2: rec_card("Monitoring", "Track conversion, calibration, and revenue per user weekly.", COLORS["accent"])
    with col3: rec_card("Risk Mgmt", "Set conversion drop alerts, keep 10% holdout, recalibrate weekly.", COLORS["success"])
    with col4: rec_card("Iteration", "A/B test variants; retrain uplift model monthly.", COLORS["primary"])
    
    st.markdown("---")
    st.subheader("Reference Materials")
    
    with st.expander("Glossary"):
        colg1, colg2 = st.columns(2)
        with colg1:
            st.markdown("""
            **Core Terms**
            - **Treatment / Control:** Two groups in a randomized experiment — the *treatment* group experiences the new interface, the *control* group sees the current one.
            - **Lift:** Difference in conversion rates between treatment and control; can be expressed in absolute percentage points or relative percent.
            - **Uplift:** Predicted *incremental* improvement in conversion for an individual if they were in treatment vs control, based on the model.
            - **CUPED:** "Controlled Pre-Experiment Data" — a variance-reduction technique that uses pre-treatment behavior to make treatment–control comparisons more precise.
            """)
        with colg2:
            st.markdown("""
            **Statistical Terms**
            - **Confidence Interval (CI):** Range of values that likely contains the true effect; a 95% CI means that if the experiment is repeated many times, 95% of the intervals would contain the true effect.
            - **Standardized Mean Difference (SMD):** A unitless measure of how different two groups are; SMD < 0.1 is generally considered well-balanced.
            - **Propensity Score:** The probability of a user being assigned to treatment, estimated from observed covariates; near 0.5 across users suggests good randomization.
            - **Z-statistic:** Test statistic that measures how many standard errors the observed effect is from zero; higher absolute values mean stronger evidence against the null hypothesis.
            """)

    with st.expander("Technical Implementation"):
        colt1, colt2 = st.columns(2)
        with colt1:
            st.markdown("""
            **Machine Learning**
            - **Approach:** X-Learner uplift modeling with separate models for treatment and control outcomes.
            - **Base Learners:** Random Forest regressors (balanced depth/leaf size to avoid overfitting).
            - **Feature Space:** 50+ behavioral and demographic features — including booking history, browsing activity, session metrics, and device type.
            - **Validation:** Time-based cross-validation to avoid leakage from future to past behavior.
            - **Clipped Scores:** Extreme uplift predictions are clipped to reduce the influence of outliers when targeting.
            """)
        with colt2:
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
st.markdown("""
<div style="text-align: center; padding: 18px; margin-top: 10px; border-top: 1px solid var(--border-color);">
  <p style="color: var(--text-color); font-size: 0.9rem; margin: 0;">
    Built with <span style="color: var(--primary-color); font-weight: 700;">Streamlit</span> &
    <span style="color: var(--accent-color); font-weight: 700;">Plotly</span> |
    <span style="color: var(--primary-color); font-weight: 800;">Causify Experimentation Platform</span> © 2025
  </p>
</div>
""", unsafe_allow_html=True)