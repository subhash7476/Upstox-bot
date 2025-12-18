# pages/1_Login_&_Instruments.py
import streamlit as st
import webbrowser
import urllib.parse
import requests
import sys
import os
from core.api.instruments import download_and_split_instruments

sys.path.append(os.path.dirname(__file__))   # ← keeps imports working

from core.config import load_config, save_access_token, is_token_valid, get_access_token

st.title("Upstox Login")
st.markdown("#### 100% Reliable Method – Same as your working get_token.py")

# Load credentials
creds = load_credentials()
API_KEY = creds["api_key"]
API_SECRET = creds["api_secret"]
REDIRECT_URI = creds["redirect_uri"]  # Must be exactly "http://localhost:8501" (update in credentials.json and Upstox app)

# Current login status
if is_token_valid():
    st.success("Already Logged In Today!")
    st.info("Token is valid for today's session")
else:
    st.warning("No valid token found")

st.divider()

# Auto-process if ?code in query params (from redirect)
params = st.query_params
if 'code' in params:
    code = params['code'][0]
    try:
        token_url = "https://api.upstox.com/v2/login/authorization/token"
        payload = {
            "code": code,
            "client_id": API_KEY,
            "client_secret": API_SECRET,
            "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code"
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        response = requests.post(token_url, data=payload, headers=headers)
        response.raise_for_status()
        resp_json = response.json()
        token = resp_json.get("access_token")
        if not token:
            st.error("No access_token in response – check if code is valid (redo login).")
        else:
            save_access_token(token)
            st.success("SUCCESS! Token auto-saved from redirect!")
            st.experimental_set_query_params()  # Clear ?code
            st.experimental_rerun()
    except requests.exceptions.HTTPError as http_err:
        st.error(f"Auto HTTP error: {http_err} – Likely URI mismatch or invalid code. Response: {response.text}")
    except Exception as e:
        st.error(f"Auto unexpected failure: {e}")

# Manual trigger for login
if st.button("Get Fresh Access Token (Never Fails)", type="primary", use_container_width=True):
    # Open Upstox login page
    auth_url = (
        f"https://api.upstox.com/v2/login/authorization/dialog"
        f"?client_id={API_KEY}"
        f"&redirect_uri={urllib.parse.quote(REDIRECT_URI)}"
        f"&response_type=code"
    )
    
    st.info("Opening Upstox login page in your browser...")
    webbrowser.open(auth_url)
    st.session_state.show_form = True  # Show manual form if needed
    st.experimental_rerun()

# Manual paste form (fallback if auto-redirect fails)
if 'show_form' in st.session_state and st.session_state.show_form:
    st.markdown("""
    ### After you log in:
    1. Upstox will redirect you to a page that shows "This site can’t be reached" – that's normal (if port mismatch).
    2. Copy the **entire URL** from the browser address bar (it will look like):
       `http://localhost:8501?code=xxxxxxxxxxxxxxxx`
    3. Paste it in the box below and click **Submit**. (If auto-detect worked, this form won't show.)
    """)
    url_input = st.text_input(
        "Paste the full redirect URL here",
        placeholder="http://localhost:8501?code=abc123...",
        type="password"
    )
    if st.button("Submit URL → Get Token", type="primary"):
        if "code=" not in url_input:
            st.error("URL does not contain ?code= → copy the full URL again.")
        else:
            try:
                from urllib.parse import urlparse, parse_qs
                parsed = urlparse(url_input)
                query_params = parse_qs(parsed.query)
                code = query_params.get('code', [None])[0]
                if not code:
                    st.error("No code found in URL params.")
                else:
                    token_url = "https://api.upstox.com/v2/login/authorization/token"
                    payload = {
                        "code": code,
                        "client_id": API_KEY,
                        "client_secret": API_SECRET,
                        "redirect_uri": REDIRECT_URI,
                        "grant_type": "authorization_code"
                    }
                    headers = {"Content-Type": "application/x-www-form-urlencoded"}
                    response = requests.post(token_url, data=payload, headers=headers)
                    response.raise_for_status()
                    resp_json = response.json()
                    token = resp_json.get("access_token")
                    if not token:
                        st.error("No access_token – code invalid/expired.")
                    else:
                        save_access_token(token)
                        del st.session_state.show_form  # Hide form
                        st.success("SUCCESS! Token saved!")
                        st.balloons()
                        st.experimental_rerun()
            except requests.exceptions.HTTPError as http_err:
                st.error(f"HTTP error: {http_err} – Likely invalid redirect_uri or code. Response: {response.text if 'response' in locals() else 'No response'}")
            except Exception as e:
                st.error(f"Unexpected failure: {e}")

# Next step
st.divider()
if is_token_valid():
    st.success("You are ready!")
    if st.button("Update Latest Instruments →", type="primary", use_container_width=True):
        ok = download_and_split_instruments()
        if ok:
            st.success("Instrument Update Completed Successfully")
else:
    st.info("Get token first to unlock next steps")