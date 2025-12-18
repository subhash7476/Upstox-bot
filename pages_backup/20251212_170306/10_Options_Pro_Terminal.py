import streamlit as st
import pandas as pd
import os
import sys
import requests
from datetime import datetime

# ---------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------
DATA_DIR = r"D:\bot\instruments\segment_wise"
if r"D:\bot\scripts" not in sys.path:
    sys.path.append(r"D:\bot\scripts")

st.set_page_config(layout="wide", page_title="Options Pro Terminal")

# ---------------------------------------------
# 2. SMART DATA LOADERS
# ---------------------------------------------
@st.cache_data
def load_symbols_smart(segment):
    """
    Loads symbols. 
    """
    # NSE_INDEX: Standard List
    if segment == "NSE_INDEX":
        return ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]

    # NSE_FO: Extract from 'instrument_type' column
    path = os.path.join(DATA_DIR, f"{segment}.parquet")
    if not os.path.exists(path):
        return ["RELIANCE", "TCS", "SBIN", "ADANIENT"] # Fallback

    try:
        df = pd.read_parquet(path)
        # Filter for Futures to get clean Underlying names
        if 'instrument_type' in df.columns and 'trading_symbol' in df.columns:
            futures = df[df['instrument_type'].astype(str).str.contains("FUT", case=False)]
            if not futures.empty:
                raw = futures['trading_symbol'].astype(str).tolist()
                # "RELIANCE FUT..." -> "RELIANCE"
                return sorted(list(set([x.split(' ')[0] for x in raw])))
        
        # Fallback
        col = next((c for c in ['trading_symbol', 'symbol'] if c in df.columns), None)
        if col:
            raw = df[col].astype(str).unique().tolist()
            return sorted(list(set([x.split()[0] for x in raw if "FUT" in x])))
    except: pass
    return []

@st.cache_data
def get_expiries_locally(symbol):
    """
    CRITICAL FIX: Always looks in NSE_FO.parquet for expiries, 
    even for Indices (NIFTY/BANKNIFTY).
    """
    # 1. Always target the F&O file for dates
    fo_path = os.path.join(DATA_DIR, "NSE_FO.parquet")
    if not os.path.exists(fo_path):
        return [datetime.now().strftime("%Y-%m-%d")]

    try:
        df = pd.read_parquet(fo_path)
        
        # 2. Strict Filter for Symbol
        # We search for "SYMBOL " (with space) or exact match to avoid "NIFTYIT" showing up for "NIFTY"
        # For NIFTY, trading_symbol is like "NIFTY 25 JAN 23000 CE"
        mask = df['trading_symbol'].astype(str).str.startswith(f"{symbol} ")
        
        filtered = df[mask]
        
        # 3. Extract Expiry
        if not filtered.empty and 'expiry' in filtered.columns:
            # Get unique expiry timestamps
            expiries = filtered['expiry'].dropna().unique()
            
            readable_dates = set()
            for ts in expiries:
                try:
                    # Handle Milliseconds (13 digits) vs Seconds (10 digits)
                    ts = float(ts)
                    if ts > 100000000000: ts = ts / 1000 
                    date_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    readable_dates.add(date_str)
                except: pass
            
            # 4. Sort and Filter Future Dates
            sorted_dates = sorted(list(readable_dates))
            today = datetime.now().strftime("%Y-%m-%d")
            future_dates = [d for d in sorted_dates if d >= today]
            
            if future_dates:
                return future_dates

    except Exception as e:
        print(f"Expiry Error: {e}")
    
    # Default fallback
    return ["2025-12-30"]

# ---------------------------------------------
# 3. API & KEY FUNCTIONS
# ---------------------------------------------
@st.cache_data
def get_instrument_key_locally(symbol, segment):
    # INDICES MAP (Always prioritize these for Indices)
    if segment == "NSE_INDEX" or symbol in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]:
        return {
            "NIFTY": "NSE_INDEX|Nifty 50",
            "BANKNIFTY": "NSE_INDEX|Nifty Bank",
            "FINNIFTY": "NSE_INDEX|Nifty Fin Service",
            "MIDCPNIFTY": "NSE_INDEX|Nifty Midcap Select"
        }.get(symbol)

    # STOCKS (NSE_EQ.parquet)
    eq_file = os.path.join(DATA_DIR, "NSE_EQ.parquet")
    if not os.path.exists(eq_file): return None

    try:
        df = pd.read_parquet(eq_file)
        df.columns = [c.lower().strip() for c in df.columns]
        
        sym_col = next((c for c in df.columns if 'symbol' in c), None)
        key_col = next((c for c in df.columns if 'key' in c), None)

        if sym_col and key_col:
            target = str(symbol).strip().upper()
            match = df[df[sym_col].astype(str).str.strip().str.upper() == target]
            if not match.empty: return match.iloc[0][key_col]
            match = df[df[sym_col].astype(str).str.contains(target, case=False)]
            if not match.empty: return match.iloc[0][key_col]
    except: pass
    return None

def get_spot_price(key, access_token):
    if not key or not access_token: return 0, {}
    url = "https://api.upstox.com/v2/market-quote/ltp"
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
    try:
        resp = requests.get(url, params={"instrument_key": key}, headers=headers, timeout=5)
        raw = resp.json()
        if resp.status_code == 200:
            data = raw.get("data", {})
            if data: return next(iter(data.values())).get("last_price", 0), raw
        return 0, raw
    except Exception as e: return 0, {"error": str(e)}

def get_option_chain(key, expiry, access_token):
    if not key or not access_token: return None
    url = "https://api.upstox.com/v2/option/chain"
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
    try:
        resp = requests.get(url, params={"instrument_key": key, "expiry_date": expiry}, headers=headers, timeout=8)
        if resp.status_code == 200: return resp.json().get("data", [])
    except: pass
    return None

# ---------------------------------------------
# 4. MAIN APP
# ---------------------------------------------
def app():
    try:
        from auth.auth_manager import get_access_token
        access_token = get_access_token()
    except:
        st.error("⚠️ Auth Manager Missing.")
        return

    st.title("Options Pro Terminal — V8 (Index Expiry Fix)")

    # --- SIDEBAR CONFIG ---
    st.sidebar.header("Configuration")
    segment = st.sidebar.selectbox("Segment", ["NSE_FO", "NSE_INDEX"])
    
    # 1. LOAD SYMBOLS
    symbols = load_symbols_smart(segment)
    
    # 2. SELECT SYMBOL
    # Ensure correct default selection
    default_sym = "NIFTY" if segment == "NSE_INDEX" else ("ADANIENT" if "ADANIENT" in symbols else symbols[0])
    try:
        def_idx = symbols.index(default_sym)
    except:
        def_idx = 0
        
    symbol = st.sidebar.selectbox("Select Symbol", symbols, index=def_idx)
    
    # 3. SELECT EXPIRY (AUTO-POPULATED)
    # Now correctly pulls NIFTY dates from NSE_FO file
    available_expiries = get_expiries_locally(symbol)
    
    # Logic to select nearest date
    expiry = st.sidebar.selectbox("Select Expiry", available_expiries)

    # --- MAIN LOGIC ---
    inst_key = get_instrument_key_locally(symbol, segment)
    spot_price, debug_json = get_spot_price(inst_key, access_token)
    
    # Top Metrics
    c1, c2 = st.columns([1, 2])
    c1.metric(f"{symbol} Spot Price", spot_price)
    
    # Debug Info
    with st.expander("🛠 Debugger", expanded=(spot_price==0)):
        st.write(f"**Resolved Key:** `{inst_key}`")
        if spot_price == 0: st.json(debug_json)

    # Option Chain Display
    if inst_key:
        with st.spinner(f"Fetching Option Chain for {expiry}..."):
            chain_data = get_option_chain(inst_key, expiry, access_token)
            
        if chain_data:
            st.success(f"✅ Loaded {len(chain_data)} contracts for {expiry}")
            
            ce_data = []
            pe_data = []
            
            for c in chain_data:
                strike = c["strike_price"]
                if c.get("call_options"):
                    row = c["call_options"]["market_data"]
                    row["strike"] = strike
                    ce_data.append(row)
                if c.get("put_options"):
                    row = c["put_options"]["market_data"]
                    row["strike"] = strike
                    pe_data.append(row)
            
            df_ce = pd.DataFrame(ce_data)
            df_pe = pd.DataFrame(pe_data)
            
            # Simple Display
            col1, col2 = st.columns(2)
            col1.subheader("Calls (CE)")
            col1.dataframe(df_ce, hide_index=True)
            col2.subheader("Puts (PE)")
            col2.dataframe(df_pe, hide_index=True)
            
        else:
            if spot_price > 0:
                st.warning(f"No options found for **{expiry}**. Indices often have Weekly Expiries (e.g. Thursdays). Try checking the next few dates.")
    else:
        st.error(f"Could not resolve Instrument Key for {symbol}")

if __name__ == "__main__":
    app()