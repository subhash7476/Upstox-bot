# pages/10_Options_Pro_Terminal.py
import streamlit as st
from core.api.upstox_client import UpstoxClient
from core.config import get_access_token
from core.options import select_best_option

st.set_page_config(page_title="Options Pro Terminal", layout="wide")
st.title("📈 Options Pro Terminal — Clean UI")

token = get_access_token()
client = UpstoxClient(token) if token else None

st.sidebar.header("Config")
underlying = st.sidebar.selectbox("Underlying", ["NIFTY","BANKNIFTY","RELIANCE"])
expiry = st.sidebar.text_input("Expiry (YYYY-MM-DD)", value="")
if st.sidebar.button("Fetch Chain"):
    if not token:
        st.error("No token. Go to Login.")
    elif not expiry:
        st.error("Provide expiry date.")
    else:
        try:
            inst_key = client.get_instrument_key_local(underlying)
            chain = client.get_option_chain(inst_key or underlying, expiry)
            st.success(f"Loaded CE:{len(chain.get('CE',[]))} PE:{len(chain.get('PE',[]))}")
            st.write("Sample CE:", chain.get("CE", [])[:5])
            st.write("Sample PE:", chain.get("PE", [])[:5])
        except Exception as e:
            st.error(f"Failed: {e}")

st.markdown("---")
st.header("Quick Helpers")
if st.button("Show expiries (local)"):
    try:
        exps = client.get_expiries_for_underlying(underlying)
        st.write(exps[:20])
    except Exception as e:
        st.error(f"Error: {e}")
