import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import plotly.graph_objects as go

st.title("Supertrend Backtester v3 — Stable & Filtered")

PROCESSED_DIR = "data/processed"
TRADES_DB = "data/trades_memory.json"

os.makedirs("data", exist_ok=True)
if not os.path.exists(TRADES_DB):
    with open(TRADES_DB, "w") as f:
        json.dump({"trades": []}, f, indent=2)

# ===========================
# LOAD DATA
# ===========================
files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".parquet")]
if not files:
    st.error("No parquet files found in data/processed")
    st.stop()

file = st.selectbox("Data File", sorted(files, reverse=True))
df = pd.read_parquet(f"{PROCESSED_DIR}/{file}")

st.success(f"Loaded {len(df):,} candles")

# ===========================
# PARAMETERS
# ===========================
col1, col2, col3 = st.columns(3)
with col1:
    atr_period = st.slider("ATR Period", 7, 20, 10)
    multiplier = st.slider("Supertrend Multiplier", 1.0, 5.0, 3.0, 0.1)
with col2:
    capital_start = st.number_input("Starting Capital", 10000, 10000000, 200000)
    risk_pct = st.number_input("Risk per Trade (%)", 0.1, 20.0, 2.0)
with col3:
    min_volume_factor = st.slider("Min Volume Factor", 0.1, 2.0, 0.8, 0.1)
    allowed_hours = st.multiselect("Allowed Hours", list(range(24)), default=list(range(9, 16)))

MAX_QTY = 500
MAX_POSITIONS = 1
MAX_VAL = 1e12

# ===========================
# SUPERTREND (WITH ATR)
# ===========================
def compute_supertrend(df, period, mult):
    high, low, close = df["High"], df["Low"], df["Close"]

    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, adjust=False).mean()

    hl2 = (high + low) / 2
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr

    trend = pd.Series(1, index=df.index)
    st_line = lower.copy()

    for i in range(1, len(df)):
        if close.iloc[i] > upper.iloc[i-1]:
            trend.iloc[i] = 1
        elif close.iloc[i] < lower.iloc[i-1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]

        st_line.iloc[i] = lower.iloc[i] if trend.iloc[i] == 1 else upper.iloc[i]

    out = df.copy()
    out["Supertrend"] = st_line
    out["Trend"] = trend == 1
    out["ATR"] = atr
    return out

df = compute_supertrend(df, atr_period, multiplier)

# ===========================
# ADVANCED FILTERS
# (Now ATR exists → SAFE)
# ===========================

# Trend Strength (EMA20 > EMA50)
df["EMA20"] = df["Close"].ewm(span=20).mean()
df["EMA50"] = df["Close"].ewm(span=50).mean()
df["TrendStrong"] = df["EMA20"] > df["EMA50"]

# ATR Regime Filter
atr_ma = df["ATR"].rolling(50).mean()
df["ATR_OK"] = df["ATR"] > atr_ma * 0.8

# Candle Body Strength
body = (df["Close"] - df["Open"]).abs()
range_ = (df["High"] - df["Low"]).replace(0, 1e-6)
df["StrongCandle"] = (body / range_) > 0.4

# Distance from Supertrend
df["ST_Dist"] = ((df["Close"] - df["Supertrend"]).abs() / df["Close"])

# Hour & Volume Filters
df["Hour"] = df.index.hour
vol_ma = df["Volume"].rolling(20).mean()
df["VolOK"] = df["Volume"] > (min_volume_factor * vol_ma)

# ===========================
# SIGNAL GENERATION
# ===========================
df["PrevTrend"] = df["Trend"].shift(1)
df["PrevTrend"] = df["PrevTrend"].fillna(False).infer_objects(copy=False)
df["Signal"] = 0

# Trend Flip
df.loc[(df["Trend"] == True) & (df["PrevTrend"] == False), "Signal"] = 1
df.loc[(df["Trend"] == False) & (df["PrevTrend"] == True), "Signal"] = -1

# Momentum Confirmation
df.loc[(df["Trend"] == True) & (df["Close"] > df["Supertrend"]), "Signal"] = 1
df.loc[(df["Trend"] == False) & (df["Close"] < df["Supertrend"]), "Signal"] = -1

# ===========================
# APPLY FILTERS TO SIGNALS
# ===========================
df.loc[df["TrendStrong"] == False, "Signal"] = 0
df.loc[df["ATR_OK"] == False, "Signal"] = 0
df.loc[df["StrongCandle"] == False, "Signal"] = 0
df.loc[df["ST_Dist"] < 0.002, "Signal"] = 0
df.loc[df["VolOK"] == False, "Signal"] = 0
df.loc[~df["Hour"].isin(allowed_hours), "Signal"] = 0

# ===========================
# BACKTEST ENGINE
# ===========================
def backtest(df, capital, risk_pct):
    cash = capital
    positions = []
    closed = []
    equity_curve = []

    for i in range(len(df)):
        price = df["Close"].iloc[i]
        date = str(df.index[i])

        # Close positions (SL + trailing)
        for pos in positions[:]:
            qty = pos["qty"]
            entry_price = pos["entry_price"]
            side = pos["side"]

            # Stoploss
            if (side == "LONG" and price <= pos["stoploss"]) or (side == "SHORT" and price >= pos["stoploss"]):
                pnl_pct = ((price - entry_price) / entry_price * 100) if side == "LONG" else ((entry_price - price) / entry_price * 100)
                cash += qty * price
                closed.append({
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "qty": qty,
                    "pnl_pct": round(pnl_pct, 2),
                    "type": side,
                    "reason": "stoploss"
                })
                positions.remove(pos)
                continue

            # Trailing via Supertrend
            st_val = df["Supertrend"].iloc[i]
            if side == "LONG":
                pos["stoploss"] = max(pos["stoploss"], st_val)
            else:
                pos["stoploss"] = min(pos["stoploss"], st_val)

        # New Entry
        signal = int(df["Signal"].iloc[i])

        if signal != 0 and len(positions) < MAX_POSITIONS:
            risk_amt = cash * (risk_pct / 100)
            atr = df["ATR"].iloc[i]
            sl_dist = max(0.002 * price, 2 * atr)

            qty = int(risk_amt / sl_dist)
            qty = max(1, min(qty, MAX_QTY))

            if signal == 1:
                positions.append({
                    "entry_price": price,
                    "qty": qty,
                    "entry_date": date,
                    "entry_idx": i,
                    "side": "LONG",
                    "stoploss": price - sl_dist,
                })
                cash -= qty * price

            elif signal == -1:
                positions.append({
                    "entry_price": price,
                    "qty": qty,
                    "entry_date": date,
                    "entry_idx": i,
                    "side": "SHORT",
                    "stoploss": price + sl_dist,
                })
                margin = qty * price * 0.3
                cash -= margin

        unrealized = sum([(p["qty"] * price) for p in positions])
        equity = min(MAX_VAL, max(-MAX_VAL, cash + unrealized))
        equity_curve.append(equity)

    # Close all positions at end
    last_price = df["Close"].iloc[-1]
    last_date = str(df.index[-1])

    for pos in positions:
        entry_price = pos["entry_price"]
        qty = pos["qty"]
        side = pos["side"]
        pnl_pct = ((last_price - entry_price) / entry_price * 100) if side == "LONG" else ((entry_price - last_price) / entry_price * 100)
        closed.append({
            "entry_date": pos["entry_date"],
            "exit_date": last_date,
            "entry_price": entry_price,
            "exit_price": last_price,
            "qty": qty,
            "pnl_pct": round(pnl_pct, 2),
            "type": side,
            "reason": "end"
        })

    # Save trades
    with open(TRADES_DB) as f:
        mem = json.load(f)

    mem["trades"].extend(closed)

    with open(TRADES_DB, "w") as f:
        json.dump(mem, f, indent=2)

    return equity_curve, closed

# ===========================
# RUN BACKTEST
# ===========================
if st.button("Run Backtest v3"):
    equity, closed = backtest(df, capital_start, risk_pct)

    st.metric("Trades", len(closed))
    st.metric("Losing Trades", sum(t["pnl_pct"] < 0 for t in closed))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[:len(equity)], y=equity, name="Equity"))
    fig.update_layout(title="Equity Curve", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(pd.DataFrame(closed))
    st.success("Backtest v3 Complete")
