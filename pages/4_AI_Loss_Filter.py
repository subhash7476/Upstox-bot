import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys, os

# --- PATH SETUP ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.quant import generate_signals, DEFAULTS
from core.metrics import compute_metrics

st.set_page_config(layout="wide", page_title="AI Loss Filter")

# ==============================================================================
# 1. SIDEBAR & SETUP
# ==============================================================================
DATA_DIR = Path("data/derived")
if not DATA_DIR.exists():
    DATA_DIR = Path("data/processed")

files = sorted(DATA_DIR.rglob("*.parquet"))

with st.sidebar:
    st.title("⚙️ AI Settings")
    if files:
        selected_file = st.selectbox("Select File", files, format_func=lambda x: x.name)
    else:
        st.warning("No data found.")
        st.stop()
        
    st.divider()
    st.caption("Strategy Parameters")
    atr_p = st.number_input("ATR Period", value=DEFAULTS['atr_period'])
    mult = st.number_input("Supertrend Mult", value=DEFAULTS['mult'])
    
    st.divider()
    st.caption("AI Filters")
    ntz_thresh = st.slider("No Trade Zone", 0.5, 3.0, DEFAULTS['ntz_atr_mult'])
    conf_thresh = st.slider("Min Confidence", 0.0, 1.0, DEFAULTS['conf_thresh'])
    
    params = DEFAULTS.copy()
    params['atr_period'] = int(atr_p)
    params['mult'] = mult
    params['ntz_atr_mult'] = ntz_thresh
    params['conf_thresh'] = conf_thresh
    
    run_btn = st.button("▶️ RUN ANALYSIS", type="primary", use_container_width=True)

# Load Data
df_raw = pd.read_parquet(selected_file)
if "timestamp" in df_raw.columns:
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw.set_index("timestamp", inplace=True)

# ==============================================================================
# 2. BACKTEST ENGINE (V2 - Mandatory Exits)
# ==============================================================================
def robust_backtest(df, use_ai_filter=False):
    balance = 100000
    position = 0
    entry_price = 0
    trades = []
    equity = []
    
    # Fast Arrays
    opens = df['Open'].values
    trends = df['Trend'].values
    signals = df['FinalSignal'].values if use_ai_filter else df['Signal'].values
    dates = df.index
    
    for i in range(len(df)-1):
        curr_trend = trends[i]
        curr_sig = signals[i]
        exec_price = opens[i+1]
        exec_date = dates[i+1]
        
        # --- EXIT LOGIC (Always obey Trend) ---
        if position == 1 and curr_trend == -1:
            pnl = (exec_price - entry_price) * 10 
            balance += pnl
            trades.append({'Exit Date': exec_date, 'PnL': pnl, 'Type': 'Long Close'})
            position = 0   
        elif position == -1 and curr_trend == 1:
            pnl = (entry_price - exec_price) * 10
            balance += pnl
            trades.append({'Exit Date': exec_date, 'PnL': pnl, 'Type': 'Short Close'})
            position = 0

        # --- ENTRY LOGIC ---
        if position == 0:
            if curr_sig == 1:
                position = 1
                entry_price = exec_price
            elif curr_sig == -1:
                position = -1
                entry_price = exec_price
        
        equity.append(balance)
    
    return pd.DataFrame(trades), equity

# ==============================================================================
# 3. DASHBOARD UI
# ==============================================================================
if run_btn:
    with st.spinner("Analyzing..."):
        df_ai = generate_signals(df_raw, params=params)
        trades_std, eq_std = robust_backtest(df_ai, use_ai_filter=False)
        trades_ai, eq_ai = robust_backtest(df_ai, use_ai_filter=True)
        
        m_std = compute_metrics(pd.DataFrame({'pnl': trades_std['PnL']})) if not trades_std.empty else {}
        m_ai = compute_metrics(pd.DataFrame({'pnl': trades_ai['PnL']})) if not trades_ai.empty else {}
        
        # --- HEADER ---
        st.header(f"📊 Analysis: {selected_file.stem}")
        
        # --- ROW 1: PRIMARY METRICS ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔴 Standard PnL", f"{m_std.get('Total PnL', 0):.2f}")
        c2.metric("Standard Trades", m_std.get('Trades', 0))
        
        delta_pnl = m_ai.get('Total PnL', 0) - m_std.get('Total PnL', 0)
        c3.metric("🟢 AI Net PnL", f"{m_ai.get('Total PnL', 0):.2f}", delta=f"{delta_pnl:.2f}")
        
        delta_trades = int(m_ai.get('Trades', 0) - m_std.get('Trades', 0))
        c4.metric("AI Trades", m_ai.get('Trades', 0), delta=delta_trades, delta_color="inverse")

        st.divider()

        # --- ROW 2: ADVANCED STATS (Sharpe, Profit Factor, etc) ---
        st.subheader("🧠 Advanced Risk Metrics")
        
        # Create a clean comparison dataframe
        metrics_df = pd.DataFrame({
            "Metric": ["Win Rate %", "Profit Factor", "Sharpe Ratio", "Avg Win/Loss Ratio", "Max Drawdown %"],
            "Standard Strategy": [
                f"{m_std.get('Win Rate %', 0)}%",
                m_std.get('Profit Factor', 0),
                m_std.get('Sharpe Ratio', 0),
                m_std.get('Avg Win/Loss', 0),
                f"{m_std.get('Max Drawdown %', 0)}%"
            ],
            "AI Filtered Strategy": [
                f"{m_ai.get('Win Rate %', 0)}%",
                m_ai.get('Profit Factor', 0),
                m_ai.get('Sharpe Ratio', 0),
                m_ai.get('Avg Win/Loss', 0),
                f"{m_ai.get('Max Drawdown %', 0)}%"
            ]
        })
        
        # Display as a styled table
        st.dataframe(
            metrics_df, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Metric": st.column_config.TextColumn("Metric", width="medium"),
                "Standard Strategy": st.column_config.TextColumn("🔴 Standard"),
                "AI Filtered Strategy": st.column_config.TextColumn("🟢 AI Filtered")
            }
        )

        st.divider()

        # --- ROW 3: CHART ---
        #st.subheader("📈 Equity Curve")
        #chart_df = pd.DataFrame({"Standard Strategy": eq_std, "AI Filtered": eq_ai})
        #st.line_chart(chart_df, color=["#FF5252", "#00C853"]) # Red vs Green

        # --- ROW 4: BLOCKED TRADES ---
        st.subheader("🛑 What did the AI Block?")
        
        if not trades_std.empty and not trades_ai.empty:
            std_dates = set(trades_std['Exit Date'])
            ai_dates = set(trades_ai['Exit Date'])
            blocked_dates = std_dates - ai_dates
            
            if blocked_dates:
                # Get the trades that exist in Standard but NOT in AI
                blocked_df = trades_std[trades_std['Exit Date'].isin(blocked_dates)].copy()
                blocked_df = blocked_df[['Exit Date', 'Type', 'PnL']]
                blocked_df = blocked_df.sort_values('Exit Date')

                def highlight_row(row):
                    if row['PnL'] > 0:
                        return ['background-color: #dbf6df; color: black'] * len(row) # Green (Bad Block)
                    else:
                        return ['background-color: #ffcdd2; color: black'] * len(row) # Red (Good Block)

                st.write(f"The AI filtered out **{len(blocked_df)}** trades.")
                st.dataframe(blocked_df.style.apply(highlight_row, axis=1), use_container_width=True)
                
                saved_loss = abs(blocked_df[blocked_df['PnL'] < 0]['PnL'].sum())
                missed_profit = blocked_df[blocked_df['PnL'] > 0]['PnL'].sum()
                st.caption(f"🛡️ Loss Avoided: **{saved_loss:.2f}** | 💸 Profit Missed: **{missed_profit:.2f}**")
            else:
                st.success("✅ No trades were completely blocked.")