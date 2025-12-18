import streamlit as st
from datetime import date, datetime
from pathlib import Path
import pandas as pd
import os
from data.data_manager import fetch_historical_v3, preflight_plan, list_partitions

st.set_page_config(layout="wide")
st.title("2 — Fetch & Manage Data (Partition-friendly, dual mode)")

PARTITION_ROOT = Path("data/stocks")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

col1, col2 = st.columns([2,1])

# Load Nifty100 symbols list
NIFTY100_CSV = r"D:\bot\instruments\Nifty100list.csv"

def load_nifty100_symbols():
    try:
        df = pd.read_csv(NIFTY100_CSV)
        if "Symbol" not in df.columns:
            st.error("CSV missing 'Symbol' column")
            return ["RELIANCE"]
        symbols = (
            df["Symbol"]
            .astype(str)
            .str.strip()
            .tolist()
        )
        # Clean list
        symbols = [s for s in symbols if s not in ("", "nan", "None", "Symbol")]
        symbols = list(dict.fromkeys(symbols))  # drop duplicates
        return symbols[:100]                    # enforce exactly 100
    except Exception as e:
        st.error(f"Failed to read Nifty100 CSV: {e}")
        return ["RELIANCE"]

with col1:
    symbols_list = load_nifty100_symbols()
    symbol = st.selectbox("Select Trading Symbol", symbols_list, index=0)
    segment = st.text_input("Segment (e.g., NSE_EQ, NSE_INDEX)", value="NSE_EQ")
    interval = st.selectbox("Interval", ["1minute","5minute","15minute","30minute","60minute","1day"], index=0)
    from_dt = st.date_input("From", value=date(2025,1,1))
    to_dt = st.date_input("To", value=date.today())
    force = st.checkbox("Force re-download (overwrite existing day partitions)", value=False)

with col2:
    st.markdown("**Actions**")
    fetch_part_btn = st.button("Fetch as partitions (recommended)")
    fetch_merge_btn = st.button("Fetch & Merge → single parquet for backtest (legacy)")
    show_parts_btn = st.button("Show known partitions for symbol/TF")

st.markdown("### Preflight plan")
try:
    plan = preflight_plan(symbol, interval)
    st.json({"last_saved": str(plan.get("last_saved")), "next_day_to_fetch": str(plan.get("next_day_to_fetch"))})
except Exception as e:
    st.error(f"Preflight failed: {e}")

if show_parts_btn:
    parts = list_partitions(symbol, interval)
    if not parts:
        st.info("No partitions found for this symbol/TF yet.")
    else:
        st.write(f"Found {len(parts)} partitions (showing last 50):")
        st.write(parts[-50:])

def merge_partitions_to_single(symbol: str, interval: str, out_path: Path):
    parts = list_partitions(symbol, interval)
    if not parts:
        raise RuntimeError("No partitions available to merge. Run full organize first.")
    frames = []
    for d in parts:
        p = PARTITION_ROOT / symbol / interval / f"year={d.year:04d}" / f"month={d.month:02d}" / f"day={d.day:02d}" / "data.parquet"
        if p.exists():
            try:
                df = pd.read_parquet(p)
                if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                frames.append(df)
            except Exception as e:
                st.warning(f"Failed to read partition {p}: {e}")
    if not frames:
        raise RuntimeError("No readable partitions found.")
    merged = pd.concat(frames).sort_index().drop_duplicates()
    merged.to_parquet(out_path)
    return merged

if fetch_part_btn:
    st.info("Starting partitioned fetch — this will append day-level parquet files under data/stocks/...")
    try:
        appended = fetch_and_save_data(symbol, segment, interval, from_date=from_dt, to_date=to_dt, incremental=False if force else True, verbose=True)
        if appended is None:
            st.warning("Fetcher returned no data (None). Check logs or token validity.")
        elif isinstance(appended, list):
            st.success(f"Appended partitions: {appended}")
            st.write(f"Total appended days: {len(appended)}")
        elif hasattr(appended, "shape"):
            st.success(f"Fetcher returned DataFrame with {len(appended)} rows. You can merge partitions via the button.")
            out = PROCESSED_DIR / f"{symbol}_{interval}_{from_dt.strftime('%Y%m%d')}_{to_dt.strftime('%Y%m%d')}.parquet"
            appended.to_parquet(out)
            st.success(f"Saved merged file for convenience: {out}")
        else:
            st.info(f"Fetcher returned: {type(appended)} — check logs.")
            st.balloon
    except Exception as e:
        st.error(f"Partitioned fetch failed: {e}")

if fetch_merge_btn:
    st.info("Running legacy fetch & merge — this will fetch in 30-day chunks then merge (legacy behaviour).")
    try:
        # Legacy behaviour: fetch 30-day windows and create partitions, then merge existing partitions for the requested range.
        appended = fetch_and_save_data(symbol, segment, interval, from_date=from_dt, to_date=to_dt, incremental=False, verbose=True)
        out_file = PROCESSED_DIR / f"{symbol}_{interval}_{from_dt.strftime('%Y%m%d')}_{to_dt.strftime('%Y%m%d')}.parquet"
        merged = merge_partitions_to_single(symbol, interval, out_file)
        st.success(f"Merged parquet created: {out_file} ({len(merged)} rows)")
    except Exception as e:
        st.error(f"Fetch & Merge failed: {e}")

st.markdown("---")
st.markdown("**Notes / Troubleshooting**")
st.markdown(r"""
- If Upstox returns no candles for minute-level windows, try shorter recent windows or use 1-day which is generally available historically.
- For 222 symbols, use the `scripts/daily_update.py` to run incremental appends overnight rather than full rebuilds.
- If tokens expire, visit Page 1 to login and refresh credentials.
""")
