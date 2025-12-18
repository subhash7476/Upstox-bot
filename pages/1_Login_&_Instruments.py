# pages/1_Login_&_Instruments.py
import streamlit as st
from core.config import get_access_token, save_access_token, load_config
from core.api.instruments import download_and_split_instruments, list_segments
from urllib.parse import urlparse, parse_qs
import webbrowser
from core.indicators import compute_supertrend
from core.config import get_access_token

st.set_page_config(page_title="Login & Instruments", layout="centered")
st.title("🔐 Login & Instruments")

st.markdown("""
This page centralizes login and instrument updates.
- Use **Get Fresh Access Token** (opens Upstox auth dialog)
- Or paste redirected URL (fallback)
""")

cfg = load_config()
if get_access_token():
    st.success("✅ Access token present")
else:
    st.warning("No access token found — please login")

col1, col2 = st.columns(2)
with col1:
    if st.button("Get Fresh Access Token (open browser)"):
        # Build auth URL if credentials available
        creds = cfg
        api_key = creds.get("api_key")
        redirect = creds.get("redirect_uri", "http://localhost:8501")
        if not api_key:
            st.error("API key missing in config/credentials.json")
        else:
            auth_url = f"https://api.upstox.com/v2/login/authorization/dialog?client_id={api_key}&redirect_uri={redirect}&response_type=code"
            st.info("Opening browser for login...")
            webbrowser.open(auth_url)

with col2:
    url_input = st.text_input("Paste full redirected URL (fallback)", placeholder="http://localhost:8501?code=xxxx")

    if st.button("Submit Redirected URL"):
        if "code=" not in url_input:
            st.error("URL missing ?code= parameter")
        else:
            try:
                # Extract ?code=xxxx
                parsed = urlparse(url_input)
                code = parse_qs(parsed.query).get("code", [None])[0]

                if not code:
                    st.error("Could not extract code from URL.")
                    st.stop()

                # Load credentials
                api_key = cfg.get("api_key")
                client_secret = cfg.get("api_secret")
                redirect_uri = cfg.get("redirect_uri")

                if not api_key or not client_secret:
                    st.error("API key or secret missing in config/credentials.json")
                    st.stop()

                # Exchange auth code → access token
                token_url = "https://api.upstox.com/v2/login/authorization/token"
                payload = {
                    "code": code,
                    "client_id": api_key,
                    "client_secret": client_secret,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code"
                }

                headers = {"Content-Type": "application/x-www-form-urlencoded"}

                import requests
                resp = requests.post(token_url, data=payload, headers=headers, timeout=10)

                if resp.status_code != 200:
                    st.error(f"HTTP {resp.status_code}: {resp.text}")
                    st.stop()

                data = resp.json()
                token = data.get("access_token")

                if not token:
                    st.error("Response did not contain access_token.")
                    st.stop()

                save_access_token(token)
                st.success("Token saved successfully!")
                #st.experimental_rerun()

            except Exception as e:
                st.error(f"Failed to exchange code: {e}")


st.markdown("---")
st.header("Instruments")
st.write("Update local instrument parquet files (keeps your analysis fast and local).")
if st.button("Download & Split Instruments (safe)"):
    try:
        ok = download_and_split_instruments()
        if ok:
            st.success("Instruments updated.")
        else:
            st.warning("Instruments download returned False/None (check logs).")
    except Exception as e:
        st.error(f"Failed to update instruments: {e}")

segments = list_segments()
st.info(f"Local segments found: {segments[:10]}")