# pages/12_Signal_Generator.py
import streamlit as st
from core.data_utils import load_parquet
from core.quant import generate_signals
from pathlib import Path

st.set_page_config(page_title="Signal Generator", layout="wide")
st.title("🔔 Signal Generator — Module")

proc = Path("data/processed")
files = sorted(list(proc.glob("*.parquet")), reverse=True)
if not files:
    st.warning("No processed data.")
    st.stop()

sel = st.selectbox("Select file", files, format_func=lambda p: p.name)
df = load_parquet(sel)

st.sidebar.header("Signal Params")
ml_thresh = st.sidebar.slider("ML threshold", 0.0, 1.0, 0.6)
params = {"ml_thresh": ml_thresh}

if st.button("Generate Signals"):
    df_sig = generate_signals(df, params=params)
    st.write("Signals count:", df_sig["FinalSignal"].value_counts().to_dict())
    st.dataframe(df_sig.tail(200))
