import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import sys, os

from backtest.indicators import supertrend
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics
from backtest.hmm_regime import supertrend_with_hmm, analyze_regime_distribution, compare_signals

BASE_DIR = Path(__file__).resolve().parents[1]
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(layout="wide")
st.title("📊 Backtester — Derived Data Only")

DERIVED_ROOT = Path("data/derived")

# ---------------- Sidebar ----------------
st.sidebar.header("Backtest Settings")

symbol = st.sidebar.text_input("Symbol", value="RELIANCE").upper()
tf = st.sidebar.selectbox("Timeframe", ["5minute","15minute","30minute","60minute"])
capital = st.sidebar.number_input("Starting Capital", value=200000)
risk_pct = st.sidebar.slider("Risk % per trade", 0.5, 5.0, 2.0)
sl_points = st.sidebar.number_input("SL (points)", value=20.0)
rr = st.sidebar.slider("Risk:Reward", 1.0, 5.0, 2.0)
direction = st.sidebar.selectbox("Direction", ["Both","Long","Short"])

# NEW: Advanced options
st.sidebar.markdown("---")
st.sidebar.subheader("Advanced Options")
supertrend_type = st.sidebar.radio("Supertrend Type", ["Standard", "Adaptive"])
enable_costs = st.sidebar.checkbox("Enable Slippage & Costs", value=True)
min_trend_bars = st.sidebar.slider("Min Trend Duration (bars)", 0, 10, 0, 
                                    help="Filter out trends shorter than this")
show_diagnostics = st.sidebar.checkbox("Show Diagnostics", value=True)
# HMM Options
st.sidebar.markdown("---")
st.sidebar.subheader("🎭 HMM Regime Filter")
use_hmm = st.sidebar.checkbox("Enable HMM Filtering", value=False,
                               help="Use Hidden Markov Model to detect market regimes and filter signals")
if use_hmm:
    hmm_confidence = st.sidebar.slider("Min Regime Confidence", 0.0, 1.0, 0.6, 0.05,
                                       help="Minimum confidence to allow signals in ranging markets")
else:
    hmm_confidence = 0.6

# ---------------- File select ----------------
files = []
folder = DERIVED_ROOT / symbol / tf
if folder.exists():
    files = sorted([f.name for f in folder.glob("*.parquet")])

file = st.selectbox("Derived Data File", files)

# ---------------- Run ----------------
if st.button("▶ Run Backtest") and file:
    df = pd.read_parquet(folder / file)
    
    # Show data info
    st.info(f"Loaded {len(df):,} candles from {df.index[0]} to {df.index[-1]}")
    
    # Apply HMM if enabled
    if use_hmm:
        with st.spinner("🎭 Training HMM model..."):
            df = supertrend_with_hmm(df, period=10, mult=3.0, min_confidence=hmm_confidence)
            df['Trend'] = df['FilteredTrend']  # Use filtered signals
        
        st.success("✅ HMM Regime Filter Applied")
        
        # Show regime analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎭 Regime Distribution")
            regime_stats = analyze_regime_distribution(df)
            st.dataframe(regime_stats, use_container_width=True)
        
        with col2:
            st.subheader("🚦 Signal Filtering")
            signal_comp = compare_signals(df)
            for k, v in signal_comp.items():
                st.metric(k, v)
            
            if signal_comp.get('Block Rate %', 0) > 30:
                st.info(f"ℹ️ HMM blocked {signal_comp['Block Rate %']:.0f}% of signals (likely low-quality trades)")
    
    # Apply selected supertrend (if HMM not used)
    else:
        if supertrend_type == "Adaptive":
            from backtest.indicators import supertrend_adaptive
            df = supertrend_adaptive(df)
            st.info("Using Adaptive Supertrend (volatility-adjusted)")
        else:
            df = supertrend(df)
            st.info("Using Standard Supertrend")
    
    # Show diagnostics if enabled
    if show_diagnostics:
        from backtest.diagnostics import analyze_trend_quality
        stats = analyze_trend_quality(df)
        
        st.subheader("📊 Trend Quality Analysis")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Trend Changes", stats["Trend Changes"])
        col2.metric("Avg Duration", f"{stats['Avg Trend Duration']} bars")
        col3.metric("Whipsaw Ratio", stats["Whipsaw Ratio"])
        col4.metric("Risk Level", stats["Whipsaw Risk"])
        
        if stats["Whipsaw Risk"] == "HIGH":
            st.warning("⚠️ High whipsaw detected! Consider using Adaptive Supertrend or increasing min trend duration.")
        elif stats["Whipsaw Risk"] == "MEDIUM":
            st.info("⚡ Medium whipsaw - acceptable but can be improved with filters.")
        else:
            st.success("✅ Low whipsaw - good trend following behavior.")

    trades = run_backtest(
            df=df,
            capital=capital,
            risk_pct=risk_pct,
            sl_points=sl_points,
            rr=rr,
            direction=direction,
            enable_costs=enable_costs
        )
    st.write("Trend distribution:")
    st.write(df["Trend"].value_counts())

    # =========================================================
    # SAVE TRADES TO CSV
    # =========================================================
    TRADES_DIR = BASE_DIR / "data" / "trades"
    TRADES_DIR.mkdir(parents=True, exist_ok=True)


    if not trades_df.empty:
        start = df.index.min().strftime("%Y%m%d")
        end = df.index.max().strftime("%Y%m%d")

        csv_file = TRADES_DIR / f"{symbol}_{tf}_trades_{start}_{end}.csv"
        trades.to_csv(csv_file, index=False)

        st.success(f"📄 Trades saved to CSV")
        st.code(str(csv_file))

        # Optional: Download button
        with open(csv_file, "rb") as f:
            st.download_button(
                label="⬇️ Download Trades CSV",
                data=f,
                file_name=csv_file.name,
                mime="text/csv"
            )
    else:
        st.warning("No trades generated — CSV not created.")

    st.write(type(trades_df), type(equity_curve))


# Show regime timeline if HMM used
    if use_hmm and 'Regime' in df.columns:
        st.subheader("📈 Regime Timeline")
        
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        # Map regimes to colors
        regime_colors = {
            'trending_up': 'green',
            'trending_down': 'red',
            'ranging': 'yellow',
            'volatile': 'orange',
            'unknown': 'gray'
        }
        
        # Plot price
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='Price',
            line=dict(color='blue', width=1)
        ))
        
        # Add regime background colors
        for regime in df['Regime'].unique():
            if regime in ['unknown', 'error']:
                continue
            
            mask = df['Regime'] == regime
            segments = df[mask].index
            
            if len(segments) > 0:
                fig.add_trace(go.Scatter(
                    x=segments,
                    y=df.loc[segments, 'Close'],
                    mode='markers',
                    name=regime,
                    marker=dict(
                        color=regime_colors.get(regime, 'gray'),
                        size=3,
                        opacity=0.5
                    )
                ))
        
        fig.update_layout(
            height=300,
            showlegend=True,
            xaxis_title="Time",
            yaxis_title="Price",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Now show metrics
    metrics = compute_metrics(trades)

    col1, col2, col3, col4 = st.columns(4)
    for (k, v), c in zip(metrics.items(), [col1,col2,col3,col4]):
        c.metric(k, v)

    # Equity curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=equity, mode="lines", name="Equity"))
    fig.update_layout(height=400, title="Equity Curve")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Trades")
    st.dataframe(trades, use_container_width=True)
