# pages/4_AI_Loss_Filter.py
import streamlit as st
from core.data_utils import load_parquet
from core.quant import generate_signals
from pathlib import Path

st.set_page_config(page_title="AI Loss Filter", layout="centered")
st.title("🧠 AI Loss Filter (Diagnostics)")

DATA_DIR = Path("data/processed")
files = sorted([p for p in DATA_DIR.glob("*.parquet")], reverse=True)

if not files:
    st.warning("No processed files available.")
    st.stop()

sel = st.selectbox("Select file for analysis", files, format_func=lambda p: p.name)
df = load_parquet(sel)

st.sidebar.header("Filter Parameters")
ml_thresh = st.sidebar.slider("ML Threshold", 0.0, 1.0, 0.6)
params = {"ml_thresh": ml_thresh}

if st.button("Run Filter & Show Diagnostics"):
    df_sig = generate_signals(df, params=params)
    st.write("Signals distribution:", df_sig["FinalSignal"].value_counts().to_dict())
    st.dataframe(df_sig.tail(100))
    st.success("Diagnostics displayed. Integrate this into Live to decide filters.")
