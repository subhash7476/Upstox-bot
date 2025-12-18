# pages/13_Parquet_Viewer.py
import streamlit as st
from pathlib import Path
from core.data_utils import load_parquet

st.set_page_config(page_title="Parquet Viewer", layout="wide")
st.title("🗂 Parquet File Browser")

proc = Path("root/data/stocks/ABB/1minute/year=2025/month=01/day=01")
files = sorted([p for p in proc.glob("*.parquet")], reverse=True)
if not files:
    st.info("No .parquet files found in data/processed.")
    st.stop()

sel = st.selectbox("Select file", files, format_func=lambda p: p.name)
df = load_parquet(sel)
st.success(f"Loaded {len(df):,} rows × {len(df.columns):,} columns")
st.subheader("Preview (first 50 rows)")
st.dataframe(df.head(50), use_container_width=True)

with st.expander("Show columns"):
    st.write(list(df.columns))
with st.expander("Summary"):
    st.write(df.describe(include='all'))
