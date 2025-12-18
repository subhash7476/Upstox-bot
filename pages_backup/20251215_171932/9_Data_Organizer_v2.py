import streamlit as st
from pathlib import Path

from data.resampler import (
    list_1m_partitions,
    load_1m_data,
    resample_from_1m,
    build_derived_parquet,
)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(layout="wide")
st.title("📦 Data Organizer v2 — 1min → Higher Timeframes")

RAW_ROOT = Path("data/stocks")
DERIVED_ROOT = Path("data/derived")

# =========================================================
# UI
# =========================================================
col1, col2 = st.columns([2, 1])

with col1:
    symbol = st.text_input(
        "Symbol (Trading Symbol)",
        value="RELIANCE"
    ).strip().upper()

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
        st.info(f"📁 Found **{len(parts)}** day-level 1-minute partitions")
    else:
        st.warning("⚠ No 1-minute partitions found for this symbol")

# =========================================================
# RUN PIPELINE
# =========================================================
if run_btn:
    try:
        status.info("🔍 Loading & cleaning 1-minute data...")
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
