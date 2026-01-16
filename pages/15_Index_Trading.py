# pages/15_Index_Trading.py

import streamlit as st
import pandas as pd
from datetime import datetime

from core.live_trading_manager import LiveTradingManager
from core.quant import generate_signals
from core.config import get_access_token
from core.database import get_db

st.set_page_config(
    page_title="Index Trading",
    layout="wide"
)

st.title("📈 Index Trading – Live MTF Scanner")
st.caption("NIFTY | BANKNIFTY | FINNIFTY (WebSocket powered)")

# -------------------------------------------------------------------
# SESSION STATE
# -------------------------------------------------------------------

if "index_live_manager" not in st.session_state:
    st.session_state["index_live_manager"] = LiveTradingManager()

live_manager = st.session_state["index_live_manager"]
db = get_db()

INDEX_UNIVERSE = {
    "NIFTY 50": "NSE_INDEX|Nifty 50",
    "BANKNIFTY": "NSE_INDEX|Nifty Bank",
    "FINNIFTY": "NSE_INDEX|Nifty Fin Service",
    "MIDCPNIFTY": "NSE_INDEX|NIFTY MID SELECT",
}


access_token = get_access_token()
if access_token:
    live_manager.start_websocket_if_needed(access_token)

st.subheader("🔌 Live Data Status")

if live_manager.ws_connected:
    st.success(
        f"🟢 WebSocket Connected (since {live_manager.ws_started_at.strftime('%H:%M:%S')})"
    )
else:
    st.warning("🟡 WebSocket not connected (REST fallback)")


st.divider()
st.subheader("📊 Live MTF Index Scanner")

selected_indices = st.multiselect(
    "Select Indices",
    options=list(INDEX_INSTRUMENTS.keys()),
    default=list(INDEX_INSTRUMENTS.keys())
)

scan_btn = st.button("🚀 Scan Live Indices", use_container_width=True)

results = []

if scan_btn and selected_indices:
    with st.spinner("Scanning live index data..."):
        for name in selected_indices:
            inst_key = INDEX_INSTRUMENTS[name]

            df_60m, df_15m, df_5m = live_manager.get_live_mtf_data(inst_key)

            if df_60m is None or df_15m is None or df_5m is None:
                continue

            # Run your existing signal engine
            sig_60 = generate_signals(df_60m).iloc[-1]
            sig_15 = generate_signals(df_15m).iloc[-1]
            sig_5 = generate_signals(df_5m).iloc[-1]

            alignment = sum([
                sig_60["FinalSignal"],
                sig_15["FinalSignal"],
                sig_5["FinalSignal"]
            ])

            signal = "LONG" if alignment >= 2 else "SHORT" if alignment <= -2 else "NO_TRADE"

            results.append({
                "Index": name,
                "Signal": signal,
                "Alignment": alignment,
                "Price": df_5m["Close"].iloc[-1],
                "Instrument Key": inst_key,
                "Time": datetime.now().strftime("%H:%M:%S")
            })

if results:
    df = pd.DataFrame(results).sort_values("Alignment", ascending=False)

    st.session_state["index_scan_results"] = df

    st.subheader("📌 Live Index Signals")
    st.dataframe(df, use_container_width=True)


def normalize_index_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Symbol"] = out["Index"]
    out["Signal"] = out["Signal"]
    out["Alignment Score"] = out["Alignment"]
    out["Price"] = out["Price"]
    out["Entry"] = out["Price"]
    out["SL"] = out["Price"] * 0.995
    out["TP"] = out["Price"] * 1.01
    out["RSI"] = "-"
    out["60m Bias"] = "INDEX"
    out["Reasons"] = "Index MTF Alignment"
    out["Bars Ago"] = 0
    out["Instrument Key"] = out["Instrument Key"]
    out["asset_class"] = "INDEX"
    return out


st.divider()

if st.button("💾 Save Index Signals to EHMA Universe", use_container_width=True):
    df = st.session_state.get("index_scan_results")

    if df is None or df.empty:
        st.warning("No index signals to save")
    else:
        from pages.utils import save_signals_to_universe  # reuse existing save logic
        norm = normalize_index_signals(df)
        saved = save_signals_to_universe(norm)
        st.success(f"✅ Saved {saved} INDEX signals to EHMA Universe")
