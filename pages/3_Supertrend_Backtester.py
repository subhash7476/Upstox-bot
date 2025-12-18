import streamlit as st
import pandas as pd
from pathlib import Path
import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.engine import run_backtest
from core.metrics import compute_metrics, expectancy
from core.diagnostics import analyze_trend_quality
from core.indicators import compute_supertrend

try:
    from core.quant_advanced import AdvancedQuantEngine
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False
    class AdvancedQuantEngine: pass 

st.set_page_config(layout="wide", page_title="Supertrend Backtester")
st.title("3 — Supertrend Backtester")

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.header("Strategy Engine")
    
    strategy_mode = st.radio(
        "Select Engine:",
        ("Standard (Indicators)", "Advanced (Math/Regime)"),
        index=0
    )

    # --- DYNAMIC PARAMETERS ---
    er_input = 0.25
    z_input = 0.20
    
    if strategy_mode == "Advanced (Math/Regime)":
        st.success("🧮 Math Tuner Active")
        er_input = st.slider("Efficiency Ratio (Noise Filter)", 0.05, 1.0, 0.25, 0.01, help="Higher = Stricter. Lower = More Trades.")
        z_input = st.slider("Z-Score (Momentum)", 0.0, 3.0, 0.2, 0.1, help="Required statistical deviation to enter.")

    st.divider()
    st.header("Risk Settings")
    initial_capital = st.number_input("Initial Capital", value=100000.0, step=10000.0)
    risk_pct = st.number_input("Risk % per Trade", value=1.0, step=0.1)
    sl_points = st.number_input("Stop Loss (points)", value=20.0, step=1.0)
    rr = st.number_input("Risk : Reward", value=2.0, step=0.1)
    direction = st.selectbox("Trade Direction", ["Both", "Long", "Short"])
    enable_costs = st.checkbox("Include Slippage & Costs", value=True)

# ==================================================
# DATA LOADING
# ==================================================
st.subheader("📂 Backtest Data")
DATA_DIR = Path("data/derived")
if not DATA_DIR.exists(): DATA_DIR = Path("data/processed")
files = sorted(DATA_DIR.rglob("*.parquet"))
df = None

if "backtest_df" in st.session_state:
    df = st.session_state["backtest_df"]
    st.info(f"Using loaded data: {len(df)} candles")

if files:
    selected_file = st.selectbox("Select Parquet File", files, format_func=lambda x: x.name)
    if st.button("📥 Load Selected Data"):
        try:
            df = pd.read_parquet(selected_file)
            if not isinstance(df.index, pd.DatetimeIndex):
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df.set_index("timestamp", inplace=True)
            required_cols = {"Open", "High", "Low", "Close"}
            df.columns = [c.title() if c.lower() in [x.lower() for x in required_cols] else c for c in df.columns]
            st.session_state["backtest_df"] = df
            st.success(f"Loaded {len(df):,} candles")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to load: {e}")

if df is None:
    st.warning("⚠️ Please load a data file to proceed.")
    st.stop()

# ==================================================
# RUN BACKTEST
# ==================================================
st.divider()
if st.button("▶ Run Backtest", type="primary"):
    with st.spinner(f"Running {strategy_mode} Backtest..."):
        
        # STANDARD MODE
        if strategy_mode == "Standard (Indicators)":
            if "Trend" not in df.columns:
                df = compute_supertrend(df)
            trades_df, equity_curve = run_backtest(df, initial_capital, risk_pct, sl_points, rr, direction, enable_costs)
            metrics = compute_metrics(trades_df)

        # ADVANCED MODE (With Sliders)
        else:
            engine = AdvancedQuantEngine(df)
            # PASS THE SLIDER VALUES HERE
            df_advanced = engine.generate_signals(er_threshold=er_input, z_threshold=z_input)
            
            with st.expander("🔍 Inspect Math (Regime & Efficiency)"):
                st.write(f"Filtering with ER > {er_input} and Z-Score > {z_input}")
                st.dataframe(df_advanced[['Close', 'er', 'z_score', 'regime', 'signal']].tail(50))

            df_advanced['Trend'] = df_advanced['signal'].replace(0, method='ffill')
            trades_df, equity_curve = run_backtest(df_advanced, initial_capital, risk_pct, sl_points, rr, direction, enable_costs)
            metrics = compute_metrics(trades_df)
            final_capital = equity_curve[-1] if equity_curve else initial_capital

    # DISPLAY RESULTS
    st.divider()
    final_capital = equity_curve[-1] if equity_curve else initial_capital
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Capital", f"{final_capital:,.2f}")
    c2.metric("Total Trades", metrics.get("Trades", 0))
    c3.metric("Win Rate %", f"{metrics.get('Win Rate %', 0)}%")
    c4.metric("Profit Factor", f"{metrics.get('Profit Factor', 0)}")

    if equity_curve: st.line_chart(pd.Series(equity_curve))
    with st.expander("🧾 Trade Logs"):
        if not trades_df.empty: st.dataframe(trades_df, use_container_width=True)
        else: st.write("No trades generated.")