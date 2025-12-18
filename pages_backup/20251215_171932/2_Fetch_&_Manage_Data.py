# pages/2_Fetch_&_Manage_Data.py
import streamlit as st
import sys, os
from pathlib import Path
from datetime import date

# ensure root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.api.historical import fetch_and_save_data  # wrapper (keeps backward compat)
from data.data_manager import (
    fetch_historical_range,
    fetch_batch,
    detect_missing_days,
    preflight_plan,
    merge_partitions_to_parquet,
    merge_partitions_to_zip,
)
import pandas as pd

st.set_page_config(page_title="Fetch & Manage Data", layout="wide")
st.title("📥 Fetch & Manage Data — Enhanced")

# Load symbol list
LIST_FILE = Path("data/Nifty100list.csv")
if LIST_FILE.exists():
    try:
        df_list = pd.read_csv(LIST_FILE)
        if "Symbol" in df_list.columns:
            symbol_list = df_list["Symbol"].dropna().astype(str).unique().tolist()
        elif "symbol" in df_list.columns:
            symbol_list = df_list["symbol"].dropna().astype(str).unique().tolist()
        else:
            symbol_list = df_list.iloc[:, 0].dropna().astype(str).unique().tolist()
    except Exception as e:
        st.error(f"Error loading Nifty100list.csv: {e}")
        symbol_list = []
else:
    symbol_list = []

col1, col2 = st.columns([3, 1])
with col1:
    if symbol_list:
        symbols = st.multiselect("Symbols (multi-select supported)", symbol_list, default=symbol_list[:1])
        symbol = symbols[0] if len(symbols)==1 else None
    else:
        symbols = [st.text_input("Symbol", value="RELIANCE").upper()]

    segment = st.selectbox("Segment", ["NSE_EQ", "NSE_FO", "NSE_INDEX"], index=0)
    interval = st.selectbox("Interval", ["1minute", "5minute", "15minute", "30minute", "60minute", "1day"], index=0)
    from_dt = st.date_input("From", value=date(2025, 1, 1))
    to_dt = st.date_input("To", value=date.today())

with col2:
    force = st.checkbox("Force re-download (overwrite)", value=False)
    gb = st.checkbox("Show gap detection", value=True)
    if st.button("Preflight"):
        try:
            # Use first selected for preflight
            target = symbols[0] if symbols else (symbol or "RELIANCE")
            pp = preflight_plan(target, interval)
            st.info(f"Last saved: {pp.get('last_saved')}, next day to fetch: {pp.get('next_day_to_fetch')}")
        except Exception as e:
            st.error(f"Preflight failed: {e}")

st.markdown("---")
st.header("Batch / Partition Downloader")

status_box = st.empty()
prog = st.progress(0)
log_area = st.empty()

def st_callback(msg: str):
    # append log and update progress loosely
    current = log_area.text_area("Log (latest at bottom)", value=(log_area.text_area("x") if False else "") + f"{msg}\n", height=200)
    # show last message too
    status_box.info(msg)

# ---------- Detect gaps ----------
if gb and symbols:
    sample = symbols[0]
    if st.button("Detect missing days for first symbol"):
        with st.spinner("Detecting missing days..."):
            missing = detect_missing_days(sample, interval, from_dt, to_dt)
            if not missing:
                st.success("No missing partition days found in the range.")
            else:
                st.warning(f"Missing {len(missing)} days. Example: {missing[:5]}")

# ---------- Run batch fetch ----------
if st.button("Fetch as partitions (batch)"):
    if not symbols:
        st.error("Provide symbols.")
    else:
        total_symbols = len(symbols)
        all_results = {}
        i = 0
        for s in symbols:
            i += 1
            status_box.info(f"[{i}/{total_symbols}] Starting {s}")
            prog.progress(int(((i-1)/total_symbols)*100))
            try:
                # use fetch_historical_range with progress callback
                saved = fetch_historical_range(s, segment, interval, from_dt, to_dt, force=force, progress_callback=lambda m: status_box.info(f"{s}: {m}"))
                all_results[s] = {"ok": True, "saved": saved}
                status_box.success(f"[{i}/{total_symbols}] Done {s} (saved {len(saved)} partitions)")
            except Exception as e:
                all_results[s] = {"ok": False, "error": str(e)}
                status_box.error(f"[{i}/{total_symbols}] Error {s}: {e}")
        prog.progress(100)
        st.write("Batch results (sample):")
        st.json({k: {"ok": v["ok"], "saved_count": len(v["saved"]) if v.get("saved") else 0, "error": v.get("error")} for k,v in all_results.items()})

# ---------- Merge partitions UI ----------
st.markdown("---")
st.header("Merge Partitions")
merge_sym = st.selectbox("Symbol to merge", symbol_list or [s.upper() for s in symbols])
merge_interval = st.selectbox("Interval to merge", ["1minute","5minute","15minute","30minute","60minute","1day"], index=0)
out_name = st.text_input("Output filename (parquet)", value=f"merged_{merge_sym}_{merge_interval}.parquet")
if st.button("Merge to parquet"):
    try:
        out_path = Path("data") / out_name
        merge_partitions_to_parquet(merge_sym, merge_interval, out_path)
        st.success(f"Saved merged parquet: {out_path}")
    except Exception as e:
        st.error(f"Merge failed: {e}")

if st.button("Merge to zip"):
    try:
        out_zip = Path("data") / (out_name.replace(".parquet", ".zip"))
        merge_partitions_to_zip(merge_sym, merge_interval, out_zip)
        st.success(f"Saved zipped parquet: {out_zip}")
    except Exception as e:
        st.error(f"Zip merge failed: {e}")

st.markdown("---")
st.write("Notes: batch fetch runs sequentially to keep Streamlit progress updates reliable. You can run multiple symbols at once (multi-select).")
