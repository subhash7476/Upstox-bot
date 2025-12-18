# pages/5_Live_Trading_With_AI.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
from datetime import datetime

st.title("Live Trading — AI Protected")
st.markdown("**AI IS LIVE — BLOCKING YOUR PAST LOSSES IN REAL TIME**")

MODEL_PATH = "data/loss_killer_model.pkl"
SCALER_PATH = "data/loss_killer_scaler.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("AI model not found! Train it first in 'AI Loss Killer page")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Load latest data
files = [f for f in os.listdir("data/processed") if f.endswith(".parquet")]
file = st.selectbox("Live Symbol", sorted(files, reverse=True))
df = pd.read_parquet(f"data/processed/{file}")

# Supertrend
def get_supertrend(df, p=10, m=3.0):
    high, low, close = df['High'], df['Low'], df['Close']
    tr = pd.DataFrame({'a': high-low, 'b': (high-close.shift()).abs(), 'c': (low-close.shift()).abs()}).max(axis=1)
    atr = tr.ewm(alpha=1/p, adjust=False).mean()
    hl2 = (high + low)/2
    upper = hl2 + m*atr
    lower = hl2 - m*atr
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
    df = df.copy()
    df['Supertrend'] = st_line
    df['Trend'] = trend == 1
    return df

df = get_supertrend(df, 10, 3.0)

# AI FILTER — BLOCK LOSING PATTERNS
def is_trade_allowed(candle):
    features = np.array([[
        candle['Open'],
        candle['High'] - candle['Low'],
        abs(candle['Close'] - candle['Open']) / (candle['High'] - candle['Low'] + 1e-6),
        candle.name.hour,
        candle.name.dayofweek,
        candle['Volume'] / df['Volume'].rolling(20).mean().iloc[-1]
    ]])
    scaled = scaler.transform(features)
    prob = model.predict_proba(scaled)[0][1]
    return prob < 0.6  # <60% chance of loss = ALLOW

# Signal + AI Filter
latest = df.iloc[-1]
signal = "BUY" if (df['Trend'].iloc[-2] == False and df['Trend'].iloc[-1] == True) else \
         "SELL" if (df['Trend'].iloc[-2] == True and df['Trend'].iloc[-1] == False) else "HOLD"

if signal != "HOLD":
    allowed = is_trade_allowed(latest)
    color = "green" if allowed else "red"
    st.markdown(f"### SUPER TREND SIGNAL: **{signal}**")
    st.markdown(f"### AI DECISION: **{'TRADE ALLOWED' if allowed else 'TRADE BLOCKED — PAST LOSER PATTERN'}**", 
                unsafe_allow_html=True)
    if not allowed:
        st.error("AI BLOCKED THIS TRADE — IT LOOKS LIKE ONE OF YOUR PAST LOSERS")
    else:
        st.success("AI APPROVED — HIGH PROBABILITY WINNER")
else:
    st.info("No signal — waiting for trend change")

# Chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price"))
fig.add_trace(go.Scatter(x=df.index, y=df['Supertrend'], name="Supertrend", line=dict(width=3)))
fig.update_layout(title=f"{file} — Live + AI Protected", height=600)
st.plotly_chart(fig, use_container_width=True)

st.balloons()
st.success("LIVE TRADING WITH AI PROTECTION IS ACTIVE")