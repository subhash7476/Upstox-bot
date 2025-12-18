# pages/5_Live_Trading_With_AI.py
import streamlit as st
from core.api.upstox_client import UpstoxClient
from core.config import get_access_token
from core.options import select_best_option
from core.quant import generate_signals
from core.data_utils import load_parquet
from datetime import date

st.set_page_config(page_title="Live Trading With AI", layout="wide")
st.title("⚡ Live Trading (Paper Mode) — AI Filtered")

token = get_access_token()
if not token:
    st.warning("No access token. Go to Login page to set token.")
    st.stop()

client = UpstoxClient(token)
st.sidebar.header("Live Setup")
underlying = st.sidebar.selectbox("Underlying", ["NIFTY","BANKNIFTY","RELIANCE"])
expiry = st.sidebar.text_input("Expiry (YYYY-MM-DD)", value="")
preview_days = st.sidebar.number_input("Load last N days", min_value=1, max_value=500, value=30)

# Simulated live: load processed file and run signal engine on last N days
from pathlib import Path
proc = Path("data/processed")
files = sorted(list(proc.glob("*.parquet")), reverse=True)
if not files:
    st.warning("No processed files to simulate live feed.")
    st.stop()
df = load_parquet(files[0])
st.info(f"Using {files[0].name} for simulated live feed")

# run signals on last preview_days * assuming daily or minute based
df_tail = df.tail(preview_days*50) if 'Close' in df.columns else df.tail(preview_days)
df_sig = generate_signals(df_tail)
st.subheader("Latest Signals")
st.dataframe(df_sig[['FinalSignal','Close']].tail(20))

st.subheader("Option candidates (if expiry provided)")
if expiry:
    try:
        chain = client.get_option_chain(underlying, expiry)
        spot = client.fetch_ltp(client.get_instrument_key_local(underlying) or f"NSE_INDEX|{underlying}") or df_sig['Close'].iloc[-1]
        top = select_best_option(chain, spot, direction="BUY" if df_sig['FinalSignal'].iloc[-1] == 1 else "SELL")
        st.write("Spot:", spot)
        st.write("Top option picks:", top)
    except Exception as e:
        st.error(f"Option fetch failed: {e}")
