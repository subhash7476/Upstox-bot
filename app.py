# app.py
import sys
import os
import json
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

sys.path.append(os.path.dirname(__file__))   # ← keeps imports working
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

import streamlit as st

# ← ONLY HERE we set page config — once for the entire app
st.set_page_config(
    page_title="Trading Bot Pro",
    page_icon="Chart_increasing",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────
st.sidebar.image("https://i.imgur.com/8Q2X2.png", width=200)  # optional logo later
st.sidebar.title("Trading Bot Control Panel")
st.sidebar.markdown("**Portable • Profitable • Unstoppable**")

pages = {
    "Login & Instruments"       : "pages/1_Login_&_Instruments.py",
    "Fetch & Manage Data"       : "pages/2_Fetch_&_Manage_Data.py",
    "Supertrend Backtester"     : "pages/3_Supertrend_Backtester.py",
    "AI Loss Killer"            : "pages/4_AI_Loss_Filter.py",
    "Live Trading (AI Protected)":"pages/5_Live_Trading_With_AI.py",
    "Adaptive Kalman Supertrend": "pages/6_Adaptive_Kalman_Supertrend.py",
}

selection = st.sidebar.radio("Navigate →", list(pages.keys()))

# Load the selected page
with open(pages[selection], encoding="utf-8") as f:
    exec(f.read())
    