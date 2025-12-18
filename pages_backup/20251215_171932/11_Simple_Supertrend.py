# pages/11_Simple_Supertrend.py
import streamlit as st
from core.data_utils import load_parquet
from core.indicators import compute_supertrend
from pathlib import Path

st.set_page_config(page_title="Simple Supertrend", layout="centered")
st.title("📈 Simple Supertrend Viewer")

proc = Path("data/processed")
files = sorted(list(proc.glob("*.parquet")), reverse=True)
if not files:
    st.warning("No processed data.")
    st.stop()

sel = st.selectbox("Select file", files, format_func=lambda p: p.name)
df = load_parquet(sel)
st.write(f"Loaded {len(df)} rows from {sel.name}")

period = st.slider("ATR Period", 5, 30, 10)
mult = st.slider("Multiplier", 1.0, 6.0, 3.0)

if st.button("Compute & Preview"):
    df_st = compute_supertrend(df.tail(500), atr_period=period, m=mult)
    st.dataframe(df_st[['Open','High','Low','Close','Supertrend','Trend']].tail(100))
