"""AI-Powered Data Analysis and Visualization Dashboard — Guided Wizard."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from src.data_loader import load_uploaded_file, load_sample_data
from src.data_profiler import profile_dataset, suggest_column_mapping, generate_cleaning_plan
from src.data_cleaner import apply_cleaning_actions, get_dynamic_filter_options, apply_dynamic_filters
from src.visualizations import (
    chart_target_distribution, chart_correlation_scatter,
    chart_boxplot_by_category, chart_category_breakdown,
    chart_trend_over_time, chart_correlation_heatmap,
)
from src.predictive_model import train_model, predict_single
from src.ai_narratives import get_openai_client, generate_data_summary, generate_visualization_insight, generate_predictive_insight, generate_full_report
from src.report_generator import generate_pdf_report, build_report_preview
from src.utils import compute_dynamic_stats, COLORS

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Professional CSS Styling ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Gradient Header Banner ── */
.main-header {
    background: linear-gradient(135deg, #0E7490 0%, #0891B2 40%, #06B6D4 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(6, 182, 212, 0.25);
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    border-radius: 50%;
}
.main-header::after {
    content: '';
    position: absolute;
    bottom: -30%; left: -10%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(52,211,153,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.main-header h1 {
    color: #FFFFFF; font-size: 2rem; font-weight: 800;
    margin: 0; letter-spacing: -0.02em; position: relative; z-index: 1;
}
.main-header p {
    color: rgba(255,255,255,0.85); font-size: 1rem;
    margin: 0.5rem 0 0 0; font-weight: 400; position: relative; z-index: 1;
}

/* ── Step Pills Row ── */
.step-pills {
    display: flex; gap: 0.5rem; flex-wrap: wrap;
    margin: 1.2rem 0; padding: 0;
}
.step-pill {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.35rem 0.9rem; border-radius: 20px;
    font-size: 0.8rem; font-weight: 600; transition: all 0.2s ease;
}
.step-active {
    background: linear-gradient(135deg, #0891B2, #06B6D4);
    color: white; box-shadow: 0 4px 15px rgba(6,182,212,0.4);
}
.step-done {
    background: rgba(52,211,153,0.15); color: #34D399;
    border: 1px solid rgba(52,211,153,0.3);
}
.step-pending {
    background: rgba(148,163,184,0.06); color: rgba(148,163,184,0.5);
    border: 1px solid rgba(148,163,184,0.12);
}

/* ── Metric Cards ── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(6,182,212,0.06) 0%, rgba(14,116,144,0.08) 100%);
    border: 1px solid rgba(6,182,212,0.18);
    border-radius: 12px; padding: 1rem 1.2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(6,182,212,0.18);
}
div[data-testid="stMetric"] label {
    color: rgba(148,163,184,0.8) !important;
    font-weight: 600; font-size: 0.75rem;
    text-transform: uppercase; letter-spacing: 0.06em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-weight: 700; font-size: 1.7rem; color: #F1F5F9;
}

/* ── Feature Cards Grid ── */
.feature-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 0.8rem; margin: 1.2rem 0;
}
.feature-card {
    background: rgba(30,41,59,0.6);
    border: 1px solid rgba(148,163,184,0.1);
    border-radius: 12px; padding: 1.2rem; text-align: center;
    transition: all 0.25s ease;
}
.feature-card:hover {
    background: rgba(6,182,212,0.08);
    border-color: rgba(6,182,212,0.3);
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(6,182,212,0.12);
}
.feature-card .icon { font-size: 1.8rem; margin-bottom: 0.4rem; }
.feature-card .title { font-weight: 600; font-size: 0.9rem; color: #F1F5F9; }
.feature-card .desc { font-size: 0.75rem; color: rgba(148,163,184,0.7); margin-top: 0.2rem; }

/* ── Upload Area ── */
div[data-testid="stFileUploader"] {
    border: 2px dashed rgba(6,182,212,0.3) !important;
    border-radius: 14px; padding: 0.5rem;
    transition: border-color 0.3s ease;
}
div[data-testid="stFileUploader"]:hover {
    border-color: rgba(6,182,212,0.7) !important;
}

/* ── Progress Bar Gradient ── */
div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #0E7490, #06B6D4, #34D399) !important;
    border-radius: 10px;
}

/* ── Primary Buttons ── */
button[kind="primary"] {
    background: linear-gradient(135deg, #0891B2 0%, #06B6D4 100%) !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; letter-spacing: 0.02em;
    padding: 0.6rem 1.5rem !important;
    box-shadow: 0 4px 15px rgba(6,182,212,0.3) !important;
    transition: all 0.2s ease !important;
}
button[kind="primary"]:hover {
    box-shadow: 0 6px 25px rgba(6,182,212,0.5) !important;
    transform: translateY(-1px);
}

/* ── Expanders ── */
div[data-testid="stExpander"] {
    border: 1px solid rgba(148,163,184,0.1);
    border-radius: 12px; overflow: hidden;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1120 0%, #0F1A2E 50%, #0B1120 100%);
    border-right: 1px solid rgba(6,182,212,0.08);
}
section[data-testid="stSidebar"] [data-testid="stMarkdown"] hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(6,182,212,0.15), transparent);
    margin: 0.8rem 0;
}

/* ── Sidebar Logo/Title ── */
.sidebar-logo {
    text-align: center; padding: 1.2rem 1rem 0.8rem 1rem;
    position: relative;
}
.sidebar-logo::after {
    content: '';
    position: absolute; bottom: 0; left: 15%; right: 15%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(6,182,212,0.2), transparent);
}
.sidebar-logo .logo-icon {
    font-size: 2.2rem; display: block; margin-bottom: 0.5rem;
    filter: drop-shadow(0 2px 8px rgba(6,182,212,0.3));
}
.sidebar-logo .logo-text {
    font-size: 1.15rem; font-weight: 800; letter-spacing: -0.01em;
    background: linear-gradient(135deg, #22D3EE 0%, #06B6D4 50%, #0891B2 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.sidebar-logo .logo-sub {
    font-size: 0.65rem; color: rgba(148,163,184,0.45);
    text-transform: uppercase; letter-spacing: 0.12em;
    margin-top: 0.2rem;
}

/* ── Sidebar Section Labels ── */
.sidebar-section-label {
    font-size: 0.65rem; font-weight: 700;
    color: rgba(6,182,212,0.5);
    text-transform: uppercase; letter-spacing: 0.1em;
    padding: 0.3rem 0.8rem 0.4rem 0.8rem;
    margin-top: 0.2rem;
}

/* ── Sidebar Progress Items ── */
.sidebar-step {
    display: flex; align-items: center; gap: 0.65rem;
    padding: 0.5rem 0.8rem; margin: 0.15rem 0.4rem;
    border-radius: 8px; font-size: 0.82rem;
    transition: all 0.25s ease;
    position: relative;
    letter-spacing: 0.01em;
}
.sidebar-step:hover {
    background: rgba(6,182,212,0.04);
}
.sidebar-step-done {
    color: rgba(52,211,153,0.85);
}
.sidebar-step-done:hover {
    color: #34D399;
}
.sidebar-step-done .step-icon {
    width: 24px; height: 24px; border-radius: 50%;
    background: rgba(52,211,153,0.12);
    border: 1px solid rgba(52,211,153,0.2);
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 0.65rem;
    flex-shrink: 0;
}
.sidebar-step-active {
    background: linear-gradient(135deg, rgba(6,182,212,0.1) 0%, rgba(14,116,144,0.08) 100%);
    color: #F1F5F9; font-weight: 600;
    border: 1px solid rgba(6,182,212,0.15);
    box-shadow: 0 2px 12px rgba(6,182,212,0.08), inset 0 0 0 1px rgba(6,182,212,0.05);
}
.sidebar-step-active .step-icon {
    width: 24px; height: 24px; border-radius: 50%;
    background: linear-gradient(135deg, #0E7490, #0891B2);
    box-shadow: 0 2px 8px rgba(6,182,212,0.3);
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 0.65rem; color: white; font-weight: 700;
    flex-shrink: 0;
}
.sidebar-step-pending {
    color: rgba(148,163,184,0.3);
}
.sidebar-step-pending .step-icon {
    width: 24px; height: 24px; border-radius: 50%;
    border: 1px solid rgba(148,163,184,0.12);
    background: rgba(148,163,184,0.03);
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 0.65rem;
    flex-shrink: 0;
}

/* ── Sidebar Inputs ── */
section[data-testid="stSidebar"] [data-testid="stTextInput"] input {
    background: rgba(6,182,212,0.04) !important;
    border: 1px solid rgba(6,182,212,0.12) !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    transition: border-color 0.2s ease !important;
}
section[data-testid="stSidebar"] [data-testid="stTextInput"] input:focus {
    border-color: rgba(6,182,212,0.35) !important;
    box-shadow: 0 0 0 2px rgba(6,182,212,0.08) !important;
}
section[data-testid="stSidebar"] .stSubheader {
    font-size: 0.75rem !important; font-weight: 700 !important;
    color: rgba(6,182,212,0.5) !important;
    text-transform: uppercase !important; letter-spacing: 0.08em !important;
}

/* ── Sidebar Metrics ── */
section[data-testid="stSidebar"] div[data-testid="stMetric"] {
    background: rgba(6,182,212,0.04) !important;
    border: 1px solid rgba(6,182,212,0.08) !important;
    padding: 0.6rem 0.8rem !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stMetric"]:hover {
    transform: none !important;
    background: rgba(6,182,212,0.06) !important;
    border-color: rgba(6,182,212,0.15) !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.2rem !important;
}

/* ── Sidebar Buttons ── */
section[data-testid="stSidebar"] button {
    background: rgba(6,182,212,0.06) !important;
    border: 1px solid rgba(6,182,212,0.12) !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    color: rgba(148,163,184,0.8) !important;
    transition: all 0.2s ease !important;
}
section[data-testid="stSidebar"] button:hover {
    background: rgba(6,182,212,0.1) !important;
    border-color: rgba(6,182,212,0.25) !important;
    color: #F1F5F9 !important;
    transform: none !important;
    box-shadow: 0 2px 8px rgba(6,182,212,0.1) !important;
}

/* ── Dataframe ── */
div[data-testid="stDataFrame"] {
    border-radius: 12px; overflow: hidden;
    border: 1px solid rgba(148,163,184,0.1);
}

/* ── Download Buttons ── */
button[data-testid="stDownloadButton"] {
    border-radius: 10px !important; font-weight: 600 !important;
}

/* ══════════════════════════════════════════════════════════════════════════
   GLOBAL HOVER POP EFFECTS
   ══════════════════════════════════════════════════════════════════════════ */

/* ── All Buttons ── */
button {
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
button:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 6px 20px rgba(6,182,212,0.25) !important;
}
button:active {
    transform: translateY(0) scale(0.98) !important;
}

/* ── Download Buttons pop ── */
button[data-testid="stDownloadButton"] button {
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease !important;
}
button[data-testid="stDownloadButton"] button:hover {
    transform: translateY(-3px) scale(1.03) !important;
    box-shadow: 0 8px 25px rgba(6,182,212,0.2) !important;
    border-color: rgba(6,182,212,0.5) !important;
}

/* ── Select boxes / Dropdowns ── */
div[data-testid="stSelectbox"],
div[data-testid="stMultiSelect"] {
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stSelectbox"]:hover,
div[data-testid="stMultiSelect"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(6,182,212,0.12);
}
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stMultiSelect"] > div > div {
    transition: border-color 0.2s ease !important;
}
div[data-testid="stSelectbox"]:hover > div > div,
div[data-testid="stMultiSelect"]:hover > div > div {
    border-color: rgba(6,182,212,0.5) !important;
}

/* ── Text Inputs / Number Inputs ── */
div[data-testid="stTextInput"],
div[data-testid="stNumberInput"] {
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stTextInput"]:hover,
div[data-testid="stNumberInput"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(6,182,212,0.12);
}
div[data-testid="stTextInput"]:hover input,
div[data-testid="stNumberInput"]:hover input {
    border-color: rgba(6,182,212,0.5) !important;
}

/* ── File Uploader ── */
div[data-testid="stFileUploader"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(6,182,212,0.12);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

/* ── Checkboxes ── */
div[data-testid="stCheckbox"] {
    transition: transform 0.15s ease;
    border-radius: 8px;
    padding: 2px 4px;
}
div[data-testid="stCheckbox"]:hover {
    transform: translateX(4px) scale(1.02);
    background: rgba(6,182,212,0.06);
}

/* ── Sliders ── */
div[data-testid="stSlider"] {
    transition: transform 0.2s ease;
}
div[data-testid="stSlider"]:hover {
    transform: translateY(-2px);
}

/* ── Expanders pop ── */
div[data-testid="stExpander"] {
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
div[data-testid="stExpander"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(6,182,212,0.1);
    border-color: rgba(6,182,212,0.25) !important;
}

/* ── Plotly Charts ── */
div[data-testid="stPlotlyChart"] {
    border-radius: 12px;
    border: 1px solid rgba(148,163,184,0.08);
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}
div[data-testid="stPlotlyChart"]:hover {
    transform: translateY(-3px) scale(1.005);
    box-shadow: 0 12px 35px rgba(6,182,212,0.15);
    border-color: rgba(6,182,212,0.3);
}

/* ── Dataframes pop ── */
div[data-testid="stDataFrame"] {
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
div[data-testid="stDataFrame"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(6,182,212,0.1);
    border-color: rgba(6,182,212,0.25) !important;
}

/* ── Metric Cards (enhance existing) ── */
div[data-testid="stMetric"]:hover {
    transform: translateY(-4px) scale(1.02);
    box-shadow: 0 12px 35px rgba(6,182,212,0.22);
    border-color: rgba(6,182,212,0.4) !important;
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    transition: transform 0.2s ease, color 0.2s ease !important;
}
button[data-baseweb="tab"]:hover {
    transform: translateY(-2px) !important;
    color: #06B6D4 !important;
}

/* ── Sidebar items (handled by sidebar-specific rules above) ── */

/* ── Step pills interactive feel ── */
.step-pill {
    transition: all 0.2s ease;
    cursor: default;
}
.step-pill:hover {
    transform: translateY(-2px) scale(1.05);
    box-shadow: 0 4px 15px rgba(6,182,212,0.2);
}
.step-done:hover {
    background: rgba(52,211,153,0.25);
    box-shadow: 0 4px 15px rgba(52,211,153,0.25);
}
.step-active:hover {
    box-shadow: 0 6px 20px rgba(6,182,212,0.5);
}

/* ── Feature cards (enhance) ── */
.feature-card {
    cursor: default;
}
.feature-card:hover {
    transform: translateY(-5px) scale(1.03);
    box-shadow: 0 12px 30px rgba(6,182,212,0.18);
}

/* ── Forms ── */
div[data-testid="stForm"] {
    border-radius: 12px;
    border: 1px solid rgba(148,163,184,0.1);
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
div[data-testid="stForm"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(6,182,212,0.1);
    border-color: rgba(6,182,212,0.25);
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(15,23,42,0.5); }
::-webkit-scrollbar-thumb { background: rgba(6,182,212,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(6,182,212,0.6); }

/* ── Section Dividers ── */
.section-divider {
    height: 1px; margin: 1.5rem 0;
    background: linear-gradient(90deg, transparent, rgba(6,182,212,0.25), transparent);
}

</style>
""", unsafe_allow_html=True)

# ── Session State Initialization ─────────────────────────────────────────────
STEPS = ["Upload", "Profile & Map", "Clean", "Visualize", "Predict", "AI Insights", "Report"]
STEP_ICONS = ["📁", "🔍", "🧹", "📈", "🔮", "🤖", "📄"]

if "current_step" not in st.session_state:
    st.session_state.current_step = 0
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "profile" not in st.session_state:
    st.session_state.profile = None
if "mapping" not in st.session_state:
    st.session_state.mapping = None
if "cleaning_actions" not in st.session_state:
    st.session_state.cleaning_actions = None
if "clean_df" not in st.session_state:
    st.session_state.clean_df = None
if "cleaning_log" not in st.session_state:
    st.session_state.cleaning_log = None
if "model_results" not in st.session_state:
    st.session_state.model_results = None
if "transition_target" not in st.session_state:
    st.session_state.transition_target = None

STEP_DESCRIPTIONS = [
    "Preparing upload area...",
    "Analyzing your data quality...",
    "Setting up cleaning tools...",
    "Building interactive charts...",
    "Preparing ML pipeline...",
    "Connecting to AI engine...",
    "Assembling your report...",
]


def go_to_step(step_index):
    st.session_state.transition_target = step_index


def _render_transition_overlay(target_step):
    """Render a loading overlay when transitioning between steps."""
    icon = STEP_ICONS[target_step]
    label = STEPS[target_step]
    desc = STEP_DESCRIPTIONS[target_step]

    # Build step dots
    dots_html = ""
    for i in range(len(STEPS)):
        if i > 0:
            conn_class = "done" if i <= target_step else ""
            dots_html += f'<div class="t-conn {conn_class}"></div>'
        if i < target_step:
            dots_html += '<div class="t-dot done"></div>'
        elif i == target_step:
            dots_html += '<div class="t-dot active"></div>'
        else:
            dots_html += '<div class="t-dot"></div>'

    import streamlit.components.v1 as components
    components.html(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@600;700&display=swap');
        @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
        @keyframes fadeOut {{ from {{ opacity: 1; }} to {{ opacity: 0; }} }}
        @keyframes pulseIcon {{
            0%, 100% {{ transform: scale(1); opacity: 0.8; }}
            50% {{ transform: scale(1.15); opacity: 1; }}
        }}
        @keyframes slideUp {{
            from {{ transform: translateY(20px); opacity: 0; }}
            to {{ transform: translateY(0); opacity: 1; }}
        }}
        @keyframes progressFill {{ from {{ width: 0%; }} to {{ width: 100%; }} }}
        @keyframes shimmer {{
            0% {{ background-position: -200% center; }}
            100% {{ background-position: 200% center; }}
        }}
        .t-overlay {{
            position: fixed; top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(15, 23, 42, 0.94);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            z-index: 99999;
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            animation: fadeIn 0.25s ease-out;
            font-family: 'Inter', sans-serif;
        }}
        .t-overlay.exit {{ animation: fadeOut 0.4s ease-in forwards; }}
        .t-icon {{
            font-size: 3.5rem;
            animation: pulseIcon 1.2s ease-in-out infinite;
            margin-bottom: 1.2rem;
            filter: drop-shadow(0 0 20px rgba(6,182,212,0.4));
        }}
        .t-label {{
            font-size: 1.2rem; font-weight: 700; color: #F1F5F9;
            animation: slideUp 0.4s ease-out 0.1s both;
            margin-bottom: 0.4rem;
        }}
        .t-sublabel {{
            font-size: 0.85rem; color: rgba(148,163,184,0.7);
            animation: slideUp 0.4s ease-out 0.2s both;
            margin-bottom: 1.5rem;
        }}
        .t-progress-track {{
            width: 220px; height: 4px;
            background: rgba(148,163,184,0.15);
            border-radius: 4px; overflow: hidden;
            animation: slideUp 0.4s ease-out 0.3s both;
        }}
        .t-progress-bar {{
            height: 100%; border-radius: 4px;
            background: linear-gradient(90deg, #0891B2, #06B6D4, #34D399);
            background-size: 200% auto;
            animation: progressFill 1.2s ease-out forwards, shimmer 1.5s linear infinite;
        }}
        .t-dots {{
            display: flex; align-items: center; gap: 0.5rem;
            margin-top: 1.5rem;
            animation: slideUp 0.4s ease-out 0.35s both;
        }}
        .t-dot {{
            width: 10px; height: 10px; border-radius: 50%;
            background: rgba(148,163,184,0.2);
        }}
        .t-dot.done {{
            background: #34D399; box-shadow: 0 0 8px rgba(52,211,153,0.4);
        }}
        .t-dot.active {{
            background: #06B6D4; box-shadow: 0 0 10px rgba(6,182,212,0.5);
            width: 12px; height: 12px;
        }}
        .t-conn {{
            width: 16px; height: 2px;
            background: rgba(148,163,184,0.15);
        }}
        .t-conn.done {{ background: rgba(52,211,153,0.4); }}
    </style>
    <div class="t-overlay" id="stepTransition">
        <div class="t-icon">{icon}</div>
        <div class="t-label">Step {target_step + 1}: {label}</div>
        <div class="t-sublabel">{desc}</div>
        <div class="t-progress-track">
            <div class="t-progress-bar"></div>
        </div>
        <div class="t-dots">{dots_html}</div>
    </div>
    <script>
        // Escape the iframe and overlay on the parent document
        var overlay = document.getElementById('stepTransition');
        var parent = window.parent.document;
        var clone = overlay.cloneNode(true);
        // Copy styles to parent
        var styles = document.querySelector('style');
        var parentStyles = parent.createElement('style');
        parentStyles.textContent = styles.textContent;
        parentStyles.id = 'transition-styles';
        // Remove old ones if exist
        var old = parent.getElementById('transition-styles');
        if (old) old.remove();
        var oldOverlay = parent.getElementById('stepTransition');
        if (oldOverlay) oldOverlay.remove();
        parent.body.appendChild(parentStyles);
        parent.body.appendChild(clone);
        // Auto-dismiss
        setTimeout(function() {{
            var el = parent.getElementById('stepTransition');
            if (el) el.classList.add('exit');
        }}, 1100);
        setTimeout(function() {{
            var el = parent.getElementById('stepTransition');
            if (el) el.remove();
            var s = parent.getElementById('transition-styles');
            if (s) s.remove();
        }}, 1500);
    </script>
    """, height=0)


# ── Handle step transitions ─────────────────────────────────────────────────
if st.session_state.transition_target is not None:
    target = st.session_state.transition_target
    st.session_state.current_step = target
    st.session_state.transition_target = None
    _render_transition_overlay(target)


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div class="sidebar-logo">
    <span class="logo-icon">📊</span>
    <span class="logo-text">AI Analytics</span><br>
    <span class="logo-sub">Data Intelligence Platform</span>
</div>
""", unsafe_allow_html=True)

# Progress tracker
st.sidebar.markdown('<div class="sidebar-section-label">Workflow</div>', unsafe_allow_html=True)

steps_html = ""
for i, step_name in enumerate(STEPS):
    icon = STEP_ICONS[i]
    if i < st.session_state.current_step:
        steps_html += (
            f'<div class="sidebar-step sidebar-step-done">'
            f'<span class="step-icon">✓</span> {icon} {step_name}</div>'
        )
    elif i == st.session_state.current_step:
        steps_html += (
            f'<div class="sidebar-step sidebar-step-active">'
            f'<span class="step-icon">{i+1}</span> {icon} {step_name}</div>'
        )
    else:
        steps_html += (
            f'<div class="sidebar-step sidebar-step-pending">'
            f'<span class="step-icon">{i+1}</span> {icon} {step_name}</div>'
        )
st.sidebar.markdown(steps_html, unsafe_allow_html=True)

# Navigation buttons
st.sidebar.markdown("---")
if st.session_state.current_step > 0:
    if st.sidebar.button("← Previous Step"):
        go_to_step(st.session_state.current_step - 1)
        st.rerun()

# Quick jump (only to completed steps)
if st.session_state.current_step > 1:
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sidebar-section-label">Jump To</div>', unsafe_allow_html=True)
    for i, step_name in enumerate(STEPS):
        if i < st.session_state.current_step:
            if st.sidebar.button(f"{STEP_ICONS[i]} {step_name}", key=f"jump_{i}"):
                go_to_step(i)
                st.rerun()

# AI Settings
st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-section-label">AI Configuration</div>', unsafe_allow_html=True)
api_key = st.sidebar.text_input("OpenAI API Key", type="password",
                                 help="Required for AI-generated insights in Steps 6 & 7",
                                 label_visibility="collapsed",
                                 placeholder="sk-... paste your API key")
ai_client = get_openai_client(api_key) if api_key else None

# Dataset info
if st.session_state.raw_df is not None:
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sidebar-section-label">Dataset</div>', unsafe_allow_html=True)
    df_info = st.session_state.raw_df
    info_cols = st.sidebar.columns(2)
    with info_cols[0]:
        st.metric("Rows", f"{len(df_info):,}")
    with info_cols[1]:
        st.metric("Columns", f"{len(df_info.columns)}")
    if st.session_state.clean_df is not None:
        st.sidebar.metric("Cleaned Rows", f"{len(st.session_state.clean_df):,}")

# ── Header ───────────────────────────────────────────────────────────────────
step_idx = st.session_state.current_step
step_label = STEPS[step_idx]
step_icon = STEP_ICONS[step_idx]

st.markdown(f"""
<div class="main-header">
    <h1>📊 AI-Powered Data Analysis & Visualization</h1>
    <p>Transform raw data into actionable insights with machine learning and AI narratives</p>
</div>
""", unsafe_allow_html=True)

# Step pills
pills_html = '<div class="step-pills">'
for i, (name, icon) in enumerate(zip(STEPS, STEP_ICONS)):
    if i < step_idx:
        pills_html += f'<span class="step-pill step-done">✓ {icon} {name}</span>'
    elif i == step_idx:
        pills_html += f'<span class="step-pill step-active">{icon} {name}</span>'
    else:
        pills_html += f'<span class="step-pill step-pending">{icon} {name}</span>'
pills_html += '</div>'
st.markdown(pills_html, unsafe_allow_html=True)

st.progress(step_idx / (len(STEPS) - 1))
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.current_step == 0:
    st.subheader("📁 Upload Your Dataset")
    st.markdown("Welcome! This dashboard guides you through a complete data analysis workflow in 7 steps.")

    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="icon">📁</div>
            <div class="title">Upload</div>
            <div class="desc">CSV or Excel files</div>
        </div>
        <div class="feature-card">
            <div class="icon">🔍</div>
            <div class="title">Profile</div>
            <div class="desc">Auto-detect issues</div>
        </div>
        <div class="feature-card">
            <div class="icon">🧹</div>
            <div class="title">Clean</div>
            <div class="desc">Fix data quality</div>
        </div>
        <div class="feature-card">
            <div class="icon">📈</div>
            <div class="title">Visualize</div>
            <div class="desc">Interactive charts</div>
        </div>
        <div class="feature-card">
            <div class="icon">🔮</div>
            <div class="title">Predict</div>
            <div class="desc">ML predictions</div>
        </div>
        <div class="feature-card">
            <div class="icon">🤖</div>
            <div class="title">AI Insights</div>
            <div class="desc">GPT narratives</div>
        </div>
        <div class="feature-card">
            <div class="icon">📄</div>
            <div class="title">Report</div>
            <div class="desc">PDF export</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("Upload any dataset — sales, education, health, finance, or anything else!")

    col_upload, col_sample = st.columns(2)

    with col_upload:
        st.markdown("**Option A: Upload your own file**")
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="Upload any tabular dataset"
        )
        if uploaded_file:
            with st.spinner("Reading file..."):
                df, error = load_uploaded_file(uploaded_file)
                if error:
                    st.error(error)
                else:
                    st.session_state.raw_df = df
                    st.session_state.file_name = uploaded_file.name
                    st.success(f"Loaded **{uploaded_file.name}** — {len(df):,} rows × {len(df.columns)} columns")
                    st.dataframe(df.head(10), use_container_width=True)

    with col_sample:
        st.markdown("**Option B: Use sample dataset**")
        st.markdown("Try the built-in education dataset with 1,485 student records.")
        if st.button("📚 Load Sample Education Data", use_container_width=True):
            sample = load_sample_data()
            if sample is not None:
                st.session_state.raw_df = sample
                st.session_state.file_name = "education_data.csv (sample)"
                st.success(f"Loaded sample dataset — {len(sample):,} rows × {len(sample.columns)} columns")
                st.rerun()
            else:
                st.error("Sample dataset not found. Run `python data/generate_dataset.py` first.")

    if st.session_state.raw_df is not None:
        st.markdown("---")
        st.markdown("**Preview:**")
        st.dataframe(st.session_state.raw_df.head(10), use_container_width=True)

        if st.button("➡️ Continue to Data Profiling", type="primary", use_container_width=True):
            go_to_step(1)
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: PROFILE & MAP COLUMNS
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_step == 1:
    df = st.session_state.raw_df
    st.subheader("🔍 Data Profiling & Column Mapping")
    st.markdown("We've analyzed your dataset and identified potential issues. Review the findings below and map your columns.")

    # Profile the data
    with st.spinner("Profiling your dataset..."):
        profile = profile_dataset(df)
        suggestions = suggest_column_mapping(df)
        st.session_state.profile = profile

    # Overview metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Rows", f"{profile['shape']['rows']:,}")
    with m2:
        st.metric("Total Columns", f"{profile['shape']['columns']}")
    with m3:
        color = "inverse" if profile["summary"]["missing_pct"] > 5 else "normal"
        st.metric("Missing Values", f"{profile['summary']['total_missing']:,}",
                  delta=f"{profile['summary']['missing_pct']}%", delta_color="inverse")
    with m4:
        st.metric("Duplicate Rows", f"{profile['summary']['duplicate_rows']:,}")

    # Issues summary
    if profile["issues"]:
        st.markdown("### ⚠️ Data Issues Found")
        issues_df = pd.DataFrame(profile["issues"])
        severity_colors = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        issues_df["Severity"] = issues_df["severity"].map(severity_colors)
        display_issues = issues_df[["Severity", "column", "type", "description", "fix"]].copy()
        display_issues.columns = ["Severity", "Column", "Issue Type", "Description", "Suggested Fix"]
        st.dataframe(display_issues, use_container_width=True, hide_index=True)
    else:
        st.success("No data issues detected! Your dataset looks clean.")

    # Column details
    with st.expander("📋 Column Details", expanded=False):
        for col_name, col_info in profile["columns"].items():
            col_left, col_right = st.columns([1, 2])
            with col_left:
                st.markdown(f"**{col_name}**")
                st.caption(f"Type: {col_info['dtype']} → Inferred: {col_info['inferred_type']}")
                st.caption(f"Non-null: {col_info['non_null']} | Unique: {col_info['unique_count']}")
            with col_right:
                if "stats" in col_info:
                    s = col_info["stats"]
                    st.caption(f"Range: [{s['min']}, {s['max']}] | Mean: {s['mean']} | Median: {s['median']}")
                elif "top_values" in col_info:
                    top = list(col_info["top_values"].items())[:5]
                    st.caption(f"Top values: {', '.join(f'{k} ({v})' for k, v in top)}")
            st.markdown("---")

    # Column mapping
    st.markdown("### 🗂️ Map Your Columns")
    st.markdown("Tell us what each column represents so we can build the right visualizations and predictions.")

    all_cols = ["(none)"] + list(df.columns)

    map_col1, map_col2 = st.columns(2)

    with map_col1:
        id_col = st.selectbox(
            "ID / Identifier column (optional)",
            all_cols,
            index=all_cols.index(suggestions["id_column"]) if suggestions["id_column"] in all_cols else 0,
            help="A column that uniquely identifies each record (e.g., student_id, order_id)"
        )

        target_col = st.selectbox(
            "Target / Outcome column *",
            all_cols,
            index=all_cols.index(suggestions["target_column"]) if suggestions["target_column"] in all_cols else 0,
            help="The main column you want to analyze and predict (e.g., grade, sales, passed)"
        )

        time_col = st.selectbox(
            "Time / Period column (optional)",
            all_cols,
            index=all_cols.index(suggestions["date_column"]) if suggestions["date_column"] in all_cols else 0,
            help="A column representing time periods for trend analysis (e.g., semester, month, year)"
        )

    with map_col2:
        available_numeric = [c for c in df.columns if c not in [id_col, target_col, time_col] or c == "(none)"]
        default_numeric = [c for c in suggestions["numeric_columns"] if c in available_numeric and c != target_col]
        numeric_cols = st.multiselect(
            "Numeric feature columns",
            [c for c in available_numeric if c != "(none)"],
            default=default_numeric,
            help="Numeric columns to use for analysis and prediction"
        )

        available_categorical = [c for c in df.columns if c not in [id_col] or c == "(none)"]
        default_categorical = [c for c in suggestions["categorical_columns"] if c in available_categorical and c != target_col]
        categorical_cols = st.multiselect(
            "Categorical feature columns",
            [c for c in available_categorical if c != "(none)" and c not in numeric_cols],
            default=default_categorical,
            help="Categorical columns for grouping and breakdown analysis"
        )

    # Validate mapping
    if target_col == "(none)":
        st.warning("Please select a target/outcome column to continue.")
    elif len(numeric_cols) + len(categorical_cols) == 0:
        st.warning("Please select at least one feature column (numeric or categorical).")
    else:
        mapping = {
            "id_column": id_col if id_col != "(none)" else None,
            "target_column": target_col,
            "date_column": time_col if time_col != "(none)" else None,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
        }
        st.session_state.mapping = mapping

        st.markdown("---")
        st.markdown("**Mapping Summary:**")
        summary_items = [
            f"- **Target:** {target_col}",
            f"- **Numeric features:** {', '.join(numeric_cols) if numeric_cols else 'None'}",
            f"- **Categorical features:** {', '.join(categorical_cols) if categorical_cols else 'None'}",
        ]
        if mapping["id_column"]:
            summary_items.insert(0, f"- **ID:** {mapping['id_column']}")
        if mapping["date_column"]:
            summary_items.append(f"- **Time:** {mapping['date_column']}")
        st.markdown("\n".join(summary_items))

        if st.button("➡️ Continue to Data Cleaning", type="primary", use_container_width=True):
            go_to_step(2)
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: CLEAN
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_step == 2:
    df = st.session_state.raw_df
    profile = st.session_state.profile
    st.subheader("🧹 Data Cleaning")
    st.markdown("Review the suggested cleaning actions below. Check the ones you want to apply.")

    if profile is None:
        profile = profile_dataset(df)
        st.session_state.profile = profile

    plan = generate_cleaning_plan(profile)

    if not plan:
        st.success("Your dataset looks clean! No cleaning actions needed.")
        st.session_state.clean_df = df.copy()
        st.session_state.cleaning_log = ["No cleaning needed — dataset was already clean."]
    else:
        st.markdown(f"**{len(plan)} cleaning actions suggested:**")

        selected_actions = []
        for i, action in enumerate(plan):
            col_name = action.get("column", "Dataset")
            label = action["description"]
            checked = action.get("auto", False)

            col_check, col_desc, col_impact = st.columns([1, 3, 2])
            with col_check:
                include = st.checkbox("Apply", value=checked, key=f"clean_{i}")
            with col_desc:
                st.markdown(f"**{label}**")
            with col_impact:
                st.caption(action.get("impact", ""))

            if include:
                selected_actions.append(action)

        st.markdown("---")

        # Additional options
        with st.expander("⚙️ Additional Cleaning Options"):
            # Convert numeric columns
            mapping = st.session_state.mapping
            if mapping:
                for col in mapping.get("numeric_columns", []):
                    if col in df.columns and df[col].dtype == object:
                        if st.checkbox(f"Convert '{col}' to numeric", value=True, key=f"convert_{col}"):
                            selected_actions.append({
                                "action": "convert_numeric",
                                "column": col,
                                "description": f"Convert '{col}' to numeric type",
                            })

        if st.button("🧹 Apply Selected Cleaning Actions", type="primary", use_container_width=True):
            with st.spinner("Cleaning data..."):
                cleaned_df, log = apply_cleaning_actions(df, selected_actions)
                st.session_state.clean_df = cleaned_df
                st.session_state.cleaning_log = log
                st.session_state.cleaning_actions = selected_actions
            st.rerun()

    # Show results if cleaning was done
    if st.session_state.cleaning_log:
        st.markdown("### ✅ Cleaning Results")
        for entry in st.session_state.cleaning_log:
            st.markdown(f"- {entry}")

        clean_df = st.session_state.clean_df
        raw_df = st.session_state.raw_df

        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Rows Before", f"{len(raw_df):,}")
        with r2:
            st.metric("Rows After", f"{len(clean_df):,}",
                      delta=f"{len(clean_df) - len(raw_df):,}" if len(clean_df) != len(raw_df) else None)
        with r3:
            missing_after = clean_df.isnull().sum().sum()
            st.metric("Remaining Missing", f"{missing_after:,}")

        with st.expander("Preview Cleaned Data"):
            st.dataframe(clean_df.head(20), use_container_width=True)

        if st.button("➡️ Continue to Visualizations", type="primary", use_container_width=True):
            go_to_step(3)
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: VISUALIZE
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_step == 3:
    df = st.session_state.clean_df
    mapping = st.session_state.mapping
    st.subheader("📈 Data Visualization")
    st.markdown("Explore your data through interactive charts. Use the sidebar filters to narrow down the data.")

    target_col = mapping["target_column"]
    numeric_cols = [c for c in mapping["numeric_columns"] if c in df.columns]
    categorical_cols = [c for c in mapping["categorical_columns"] if c in df.columns]
    time_col = mapping.get("date_column")
    if time_col and time_col not in df.columns:
        time_col = None

    if target_col not in df.columns:
        st.error(f"Target column `{target_col}` was removed during cleaning. Please go back to Step 2 and re-map your columns.")
        if st.button("⬅️ Go Back to Column Mapping", type="primary"):
            go_to_step(1)
            st.rerun()
        st.stop()

    # Filters in an expander at the top
    with st.expander("🔍 Filter Data", expanded=False):
        filter_opts = get_dynamic_filter_options(df, categorical_cols, numeric_cols)
        filters = {}

        filter_columns = st.columns(min(len(categorical_cols), 3)) if categorical_cols else []
        for idx, col in enumerate(categorical_cols):
            if col in filter_opts:
                with filter_columns[idx % len(filter_columns)]:
                    selected = st.multiselect(
                        col.replace("_", " ").title(),
                        filter_opts[col], default=[], key=f"filter_{col}"
                    )
                    if selected:
                        filters[col] = selected

        for col in numeric_cols:
            if col in filter_opts:
                info = filter_opts[col]
                range_val = st.slider(
                    col.replace("_", " ").title(),
                    float(info["min"]), float(info["max"]),
                    (float(info["min"]), float(info["max"])),
                    key=f"filter_range_{col}"
                )
                if range_val != (float(info["min"]), float(info["max"])):
                    filters[f"{col}_range"] = range_val

        filtered_df = apply_dynamic_filters(df, filters, categorical_cols, numeric_cols)
        st.caption(f"Showing {len(filtered_df):,} of {len(df):,} records")
        st.session_state["_filtered_df"] = filtered_df

    filtered_df = st.session_state.get("_filtered_df", df)

    # Metrics row
    stats = compute_dynamic_stats(filtered_df, mapping)
    metric_cols = st.columns(4)
    metric_items = [("Records", f"{stats['total_records']:,}")]
    if "unique_entities" in stats:
        metric_items.append(("Unique IDs", f"{stats['unique_entities']:,}"))
    if "target_mean" in stats:
        metric_items.append(("Avg " + target_col.replace("_", " ").title(), f"{stats['target_mean']:.2f}"))
    if "positive_rate" in stats:
        metric_items.append(("Positive Rate", f"{stats['positive_rate']:.1f}%"))
    elif "target_median" in stats:
        metric_items.append(("Median " + target_col.replace("_", " ").title(), f"{stats['target_median']:.2f}"))

    for i, (label, value) in enumerate(metric_items[:4]):
        with metric_cols[i]:
            st.metric(label, value)

    st.markdown("---")

    # Chart 1: Target Distribution
    st.markdown("#### 1. Target Variable Distribution")
    st.markdown(f"*How is `{target_col}` distributed across your dataset?*")
    fig1 = chart_target_distribution(filtered_df, target_col)
    st.plotly_chart(fig1, use_container_width=True)

    # Chart 2: Correlation Scatter (if we have 2+ numeric cols)
    if len(numeric_cols) >= 1:
        st.markdown("---")
        st.markdown("#### 2. Correlation Analysis")
        scatter_cols = st.columns([1, 1, 1])
        with scatter_cols[0]:
            x_axis = st.selectbox("X-axis", numeric_cols, key="scatter_x")
        with scatter_cols[1]:
            y_options = [target_col] + [c for c in numeric_cols if c != x_axis]
            y_axis = st.selectbox("Y-axis", y_options, key="scatter_y")
        with scatter_cols[2]:
            color_options = ["(none)"] + categorical_cols
            scatter_color = st.selectbox("Color by", color_options, key="scatter_color")

        fig2 = chart_correlation_scatter(
            filtered_df, x_axis, y_axis,
            color_col=scatter_color if scatter_color != "(none)" else None,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Chart 3: Box plot by category
    if categorical_cols and (numeric_cols or target_col):
        st.markdown("---")
        st.markdown("#### 3. Distribution by Category")
        box_cols = st.columns(2)
        with box_cols[0]:
            box_numeric = st.selectbox("Numeric variable",
                                       [target_col] + numeric_cols, key="box_num")
        with box_cols[1]:
            box_category = st.selectbox("Group by", categorical_cols, key="box_cat")

        fig3 = chart_boxplot_by_category(filtered_df, box_numeric, box_category)
        st.plotly_chart(fig3, use_container_width=True)

    # Chart 4: Category breakdown
    if categorical_cols:
        st.markdown("---")
        st.markdown("#### 4. Category Breakdown")
        breakdown_cat = st.selectbox("Category to analyze",
                                     categorical_cols, key="breakdown_cat")
        fig4 = chart_category_breakdown(filtered_df, breakdown_cat, target_col)
        st.plotly_chart(fig4, use_container_width=True)

    # Chart 5: Trend over time
    if time_col:
        st.markdown("---")
        st.markdown("#### 5. Trends Over Time")
        group_options = ["(none)"] + categorical_cols
        trend_group = st.selectbox("Group lines by", group_options, key="trend_group")
        fig5 = chart_trend_over_time(
            filtered_df, time_col, target_col,
            group_col=trend_group if trend_group != "(none)" else None,
        )
        st.plotly_chart(fig5, use_container_width=True)

    # Bonus: Correlation heatmap
    if len(numeric_cols) >= 2:
        st.markdown("---")
        st.markdown("#### Correlation Heatmap")
        heatmap_cols = [c for c in ([target_col] + numeric_cols if target_col not in numeric_cols else numeric_cols) if c in filtered_df.columns]
        fig_heat = chart_correlation_heatmap(filtered_df, heatmap_cols)
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")
    if st.button("➡️ Continue to Predictions", type="primary", use_container_width=True):
        go_to_step(4)
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_step == 4:
    df = st.session_state.clean_df
    mapping = st.session_state.mapping
    st.subheader("🔮 Predictive Model")
    st.markdown(f"Build a machine learning model to predict **{mapping['target_column']}** based on your features.")

    target_col = mapping["target_column"]
    numeric_cols = [c for c in mapping["numeric_columns"] if c in df.columns]
    categorical_cols = [c for c in mapping["categorical_columns"] if c in df.columns]

    if target_col not in df.columns:
        st.error(f"Target column `{target_col}` was removed during cleaning. Please go back to Step 2 and re-map your columns.")
        if st.button("⬅️ Go Back to Column Mapping", type="primary"):
            go_to_step(1)
            st.rerun()
        st.stop()

    # Feature selection
    st.markdown("#### Select Features for Prediction")
    feat_col1, feat_col2 = st.columns(2)
    with feat_col1:
        pred_numeric = st.multiselect("Numeric features", numeric_cols,
                                       default=numeric_cols, key="pred_num")
    with feat_col2:
        pred_categorical = st.multiselect("Categorical features", categorical_cols,
                                           default=categorical_cols, key="pred_cat")

    if not pred_numeric and not pred_categorical:
        st.warning("Select at least one feature to train the model.")
    else:
        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            with st.spinner("Training logistic regression model..."):
                pipeline, metrics, importances, feature_config = train_model(
                    df, target_col, pred_numeric, pred_categorical
                )
                st.session_state.model_results = {
                    "pipeline": pipeline,
                    "metrics": metrics,
                    "importances": importances,
                    "feature_config": feature_config,
                }
            st.rerun()

        if st.session_state.model_results:
            results = st.session_state.model_results
            metrics = results["metrics"]
            importances = results["importances"]
            pipeline = results["pipeline"]
            feature_config = results["feature_config"]

            if "error" in metrics:
                st.error(metrics["error"])
            else:
                st.markdown("### Model Performance")

                perf_col1, perf_col2 = st.columns([2, 3])

                with perf_col1:
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
                        st.metric("Precision", f"{metrics['precision']:.1%}")
                    with m2:
                        st.metric("Recall", f"{metrics['recall']:.1%}")
                        st.metric("F1 Score", f"{metrics['f1']:.1%}")

                    st.caption(f"Train: {metrics['train_size']} | Test: {metrics['test_size']}")

                    # Confusion matrix
                    st.markdown("**Confusion Matrix**")
                    labels = metrics.get("target_labels", ["Class 0", "Class 1"])
                    cm = np.array(metrics["confusion_matrix"])
                    fig_cm = px.imshow(
                        cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=labels, y=labels,
                        color_continuous_scale="Blues", text_auto=True,
                    )
                    fig_cm.update_layout(height=300, margin=dict(t=10, b=10))
                    st.plotly_chart(fig_cm, use_container_width=True)

                with perf_col2:
                    # Feature importance
                    st.markdown("**Feature Importance (Coefficients)**")
                    imp_df = pd.DataFrame(
                        sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True),
                        columns=["Feature", "Coefficient"],
                    )
                    fig_imp = px.bar(
                        imp_df, x="Coefficient", y="Feature", orientation="h",
                        color="Coefficient", color_continuous_scale="RdBu",
                        color_continuous_midpoint=0,
                    )
                    fig_imp.update_layout(height=max(350, len(imp_df) * 30), margin=dict(t=10))
                    st.plotly_chart(fig_imp, use_container_width=True)

                # Individual prediction
                st.markdown("---")
                st.markdown("### 🧪 Individual Prediction")
                st.markdown("Enter values to get a prediction:")

                with st.form("prediction_form"):
                    input_cols = st.columns(min(len(pred_numeric) + len(pred_categorical), 3))
                    input_data = {}

                    all_pred_features = pred_numeric + pred_categorical
                    for idx, col in enumerate(all_pred_features):
                        col_idx = idx % len(input_cols)
                        with input_cols[col_idx]:
                            if col in pred_numeric:
                                num_series = pd.to_numeric(df[col], errors="coerce").dropna()
                                input_data[col] = st.number_input(
                                    col.replace("_", " ").title(),
                                    min_value=float(num_series.min()),
                                    max_value=float(num_series.max()),
                                    value=float(num_series.median()),
                                    key=f"pred_input_{col}",
                                )
                            else:
                                unique_vals = sorted(df[col].dropna().unique().tolist())
                                input_data[col] = st.selectbox(
                                    col.replace("_", " ").title(),
                                    unique_vals, key=f"pred_input_{col}",
                                )

                    submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

                if submitted and pipeline is not None:
                    prediction, probability = predict_single(
                        pipeline, input_data, pred_numeric, pred_categorical
                    )
                    labels = metrics.get("target_labels", ["Negative", "Positive"])

                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        label = labels[prediction] if prediction < len(labels) else str(prediction)
                        if prediction == 1:
                            st.success(f"### Predicted: {label}")
                        else:
                            st.error(f"### Predicted: {label}")
                    with res_col2:
                        st.metric("Confidence", f"{probability:.1%}")

                    # Gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=probability * 100,
                        title={"text": "Prediction Confidence"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": COLORS["success"] if prediction == 1 else COLORS["danger"]},
                            "steps": [
                                {"range": [0, 40], "color": "#FFEBEE"},
                                {"range": [40, 60], "color": "#FFF9C4"},
                                {"range": [60, 100], "color": "#E8F5E9"},
                            ],
                            "threshold": {"line": {"color": "black", "width": 2}, "value": 50},
                        },
                        number={"suffix": "%"},
                    ))
                    fig_gauge.update_layout(height=280, margin=dict(t=50, b=10))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                st.markdown("---")
                if st.button("➡️ Continue to AI Insights", type="primary", use_container_width=True):
                    go_to_step(5)
                    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: AI INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_step == 5:
    df = st.session_state.clean_df
    mapping = st.session_state.mapping
    st.subheader("🤖 AI-Generated Insights")

    if not ai_client:
        st.warning("Enter your OpenAI API key in the sidebar to enable AI-generated insights.")
        st.info("""
        **Without an API key, you can still:**
        - Review all visualizations from Step 4
        - Use the predictive model from Step 5
        - Download cleaned data and basic stats in Step 7

        **With an API key, you unlock:**
        - Comprehensive narrative data summaries
        - Per-chart AI interpretations
        - Predictive model explanations
        - Full analytical report with recommendations
        """)
        st.markdown("---")
        if st.button("➡️ Skip to Report", type="primary", use_container_width=True):
            go_to_step(6)
            st.rerun()
    else:
        stats = compute_dynamic_stats(df, mapping)

        # Full data summary
        st.markdown("#### 📝 Comprehensive Data Analysis")
        if st.button("🤖 Generate AI Analysis", use_container_width=True, type="primary"):
            with st.spinner("AI is analyzing your data... This may take a moment."):
                summary = generate_data_summary(ai_client, stats)
                st.session_state["ai_summary"] = summary
            st.rerun()

        if "ai_summary" in st.session_state:
            st.markdown(st.session_state["ai_summary"])

        # Model interpretation
        if st.session_state.model_results and "error" not in st.session_state.model_results.get("metrics", {}):
            st.markdown("---")
            st.markdown("#### 🔮 Predictive Model Interpretation")
            if st.button("🤖 Generate Model Insight", use_container_width=True):
                with st.spinner("Interpreting model results..."):
                    results = st.session_state.model_results
                    insight = generate_predictive_insight(
                        ai_client, results["metrics"], results["importances"]
                    )
                    st.session_state["ai_model_insight"] = insight
                st.rerun()

            if "ai_model_insight" in st.session_state:
                st.markdown(st.session_state["ai_model_insight"])

        # Quick chart insights
        st.markdown("---")
        st.markdown("#### 📊 Quick Visualization Insights")

        numeric_cols = mapping.get("numeric_columns", [])
        target_col = mapping["target_column"]

        insight_topics = []
        if numeric_cols:
            for col in numeric_cols[:3]:
                corr_key = f"{col}_target_corr"
                if corr_key in stats:
                    insight_topics.append((
                        f"{col} vs {target_col}",
                        {"column": col, "target": target_col, "correlation": stats[corr_key],
                         f"{col}_mean": stats.get(f"{col}_mean")}
                    ))

        for cat_col in mapping.get("categorical_columns", [])[:2]:
            breakdown_key = f"{cat_col}_breakdown"
            if breakdown_key in stats:
                insight_topics.append((
                    f"Breakdown by {cat_col}",
                    {"category": cat_col, "breakdown": stats[breakdown_key]}
                ))

        for topic_name, topic_stats in insight_topics:
            with st.expander(f"📊 {topic_name}"):
                key = f"ai_viz_{topic_name}"
                if key not in st.session_state:
                    if st.button(f"Generate Insight", key=f"btn_{topic_name}"):
                        with st.spinner("Generating..."):
                            st.session_state[key] = generate_visualization_insight(
                                ai_client, topic_name, topic_stats
                            )
                        st.rerun()
                if key in st.session_state:
                    st.markdown(st.session_state[key])

        st.markdown("---")
        if st.button("➡️ Continue to Report", type="primary", use_container_width=True):
            go_to_step(6)
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7: REPORT
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_step == 6:
    df = st.session_state.clean_df
    mapping = st.session_state.mapping
    st.subheader("📄 Report & Export")
    st.markdown("Review your analysis report below, then download when ready.")

    stats = compute_dynamic_stats(df, mapping)
    model_metrics = {}
    feat_imp = {}
    if st.session_state.model_results and "error" not in st.session_state.model_results.get("metrics", {}):
        model_metrics = st.session_state.model_results["metrics"]
        feat_imp = st.session_state.model_results["importances"]

    # ── AI Narrative Generation ──────────────────────────────────────────────
    if ai_client and "report_narrative" not in st.session_state:
        if st.button("🤖 Generate AI Narrative for Report", use_container_width=True, type="primary"):
            with st.spinner("AI is writing your report narrative... This may take a moment."):
                narrative = generate_full_report(ai_client, stats, model_metrics, feat_imp)
                st.session_state["report_narrative"] = narrative
            st.rerun()
    elif not ai_client and "report_narrative" not in st.session_state:
        st.info("💡 Enter an OpenAI API key in the sidebar to add AI-generated narrative to your report.")

    narrative = st.session_state.get("report_narrative", None)

    # ── Live Report Preview ──────────────────────────────────────────────────
    st.markdown("---")
    preview_md = build_report_preview(stats, model_metrics, narrative)
    with st.expander("📋 Report Preview — Review Before Downloading", expanded=True):
        st.markdown(preview_md)

    # ── Downloads ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⬇️ Downloads")

    dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)

    with dl_col1:
        # Generate PDF on-the-fly from current preview content
        pdf_narrative = narrative or "AI-generated narrative not available. Provide an OpenAI API key for full analysis."
        pdf_bytes = generate_pdf_report(pdf_narrative, stats, model_metrics)
        st.download_button(
            label="📄 PDF Report",
            data=pdf_bytes,
            file_name=f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    with dl_col2:
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📊 Cleaned Data (CSV)",
            data=csv_data,
            file_name="cleaned_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with dl_col3:
        if st.session_state.cleaning_log:
            log_text = "\n".join(st.session_state.cleaning_log)
            st.download_button(
                label="📝 Cleaning Log",
                data=log_text.encode("utf-8"),
                file_name="cleaning_log.txt",
                mime="text/plain",
                use_container_width=True,
            )

    with dl_col4:
        import json
        stats_json = json.dumps(stats, indent=2, default=str)
        st.download_button(
            label="📈 Statistics (JSON)",
            data=stats_json.encode("utf-8"),
            file_name="statistics.json",
            mime="application/json",
            use_container_width=True,
        )

    # ── Workflow Summary ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ✅ Workflow Complete!")
    st.markdown(f"""
    **What was accomplished:**
    - Uploaded and profiled **{len(st.session_state.raw_df):,}** rows of data
    - Identified and addressed **{len(st.session_state.cleaning_actions or [])}** data quality issues
    - Cleaned dataset: **{len(df):,}** rows ready for analysis
    - Created interactive visualizations across **{len(mapping.get('numeric_columns', [])) + len(mapping.get('categorical_columns', []))}** features
    {"- Trained predictive model with **" + f"{model_metrics.get('accuracy', 0):.1%}" + "** accuracy" if model_metrics else ""}
    {"- Generated AI-powered narrative insights" if narrative else ""}
    """)

    if st.button("🔄 Start Over with New Dataset", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
