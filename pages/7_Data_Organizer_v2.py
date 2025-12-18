import streamlit as st
import pandas as pd
from pathlib import Path
import sys, os

# ensure root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.resampler import (
    list_1m_partitions,
    load_1m_data,
    resample_from_1m,
    build_derived_parquet,
)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(layout="wide", page_title="Data Organizer v2")
st.title("📦 Data Organizer v2 — 1min → Higher Timeframes")

RAW_ROOT = Path("data/stocks")
DERIVED_ROOT = Path("data/derived")

# =========================================================
# SYMBOL LOADER (Same as Page 2)
# =========================================================
LIST_FILE = Path("data/Nifty100list.csv")
symbol_list = []

if LIST_FILE.exists():
    try:
        df_list = pd.read_csv(LIST_FILE)
        # Handle variations in column names
        if "Symbol" in df_list.columns:
            symbol_list = df_list["Symbol"].dropna().astype(str).unique().tolist()
        elif "symbol" in df_list.columns:
            symbol_list = df_list["symbol"].dropna().astype(str).unique().tolist()
        else:
            symbol_list = df_list.iloc[:, 0].dropna().astype(str).unique().tolist()
    except Exception as e:
        st.error(f"Error loading Nifty100list.csv: {e}")

# =========================================================
# UI
# =========================================================
col1, col2 = st.columns([2, 1])

with col1:
    # If list exists, show selectbox, otherwise text input
    if symbol_list:
        # Default to RELIANCE if available, else first item
        default_idx = symbol_list.index("RELIANCE") if "RELIANCE" in symbol_list else 0
        symbol = st.selectbox("Symbol", symbol_list, index=default_idx)
    else:
        symbol = st.text_input("Symbol (Trading Symbol)", value="RELIANCE").strip().upper()

    target_tf = st.selectbox(
        "Target Timeframe",
        ["5minute", "15minute", "30minute", "60minute"]
    )

    overwrite = st.checkbox(
        "Overwrite existing derived file (rebuild)",
        value=False
    )

with col2:
    st.markdown("### Actions")
    st.info("This will merge all 1-min partitions into a single file for backtesting.")
    run_btn = st.button(
        "▶️ Generate Derived Data",
        use_container_width=True
    )

st.divider()
status = st.empty()

# =========================================================
# PRECHECKS
# =========================================================
if symbol:
    parts = list_1m_partitions(symbol)
    if parts:
        st.success(f"📁 Found **{len(parts)}** day-level 1-minute partitions for {symbol}")
    else:
        st.warning(f"⚠ No 1-minute partitions found for {symbol}. Please go to Page 2 and fetch data first.")

# =========================================================
# RUN PIPELINE
# =========================================================
if run_btn:
    try:
        status.info(f"🔍 Loading & cleaning 1-minute data for {symbol}...")
        df_1m = load_1m_data(symbol)

        status.success(f"✔ Loaded {len(df_1m):,} one-minute candles")

        status.info(f"🔄 Resampling to {target_tf}...")
        df_tf = resample_from_1m(df_1m, target_tf)

        if df_tf.empty:
            raise RuntimeError("Resampled dataframe is empty")

        status.success(f"✔ Generated {len(df_tf):,} {target_tf} candles")

        status.info("💾 Saving merged parquet...")
        out_file = build_derived_parquet(
            symbol=symbol,
            target_tf=target_tf,
            overwrite=overwrite
        )

        status.success("✅ Derived data created successfully!")
        st.code(str(out_file))

        with st.expander("🔍 Preview (last 20 rows)"):
            st.dataframe(
                df_tf.tail(20),
                use_container_width=True
            )

    except FileExistsError as e:
        status.warning(str(e))

    except Exception as e:
        status.error(f"❌ Failed: {e}")