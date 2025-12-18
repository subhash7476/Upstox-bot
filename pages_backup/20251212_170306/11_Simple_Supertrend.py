"""
Simple Supertrend — LIVE TRADING ENGINE — AUTO-RECONNECT
Fixes:
1. Auto-Reconnect: Automatically reconnects if internet/server drops.
2. Connection Status: Shows "🟢 Connected" or "🔴 Reconnecting" in the UI.
3. Smart Column Mapping: Handles 'trading_symbol' vs 'tradingsymbol' automatically.
"""

# =====================================================================
# IMPORTS
# =====================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import threading
import asyncio
import websockets
import requests
import json
import time
import queue
import os
from google.protobuf.json_format import MessageToDict
import MarketDataFeed_pb2 as pb
from streamlit.runtime.scriptrunner import add_script_run_ctx

from auth.auth_manager import get_access_token

# =====================================================================
# CONFIGURATION
# =====================================================================
st.set_page_config(page_title="Supertrend Live Engine", layout="wide")
st.title("⚡ Supertrend Live Engine — Signal Generator")

# SIGNAL PARAMETERS
ATR_PERIOD = 10
ATR_MULT = 3.0
HTF_RESAMPLE = "15min" 

# =====================================================================
# INDICATOR LOGIC
# =====================================================================
def calculate_indicators(df):
    if len(df) < 50: return df 

    df = df.copy()
    
    # Supertrend
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/ATR_PERIOD, adjust=False).mean()
    
    hl2 = (high + low) / 2.0
    upper_basic = hl2 + ATR_MULT * atr
    lower_basic = hl2 - ATR_MULT * atr
    upper = upper_basic.copy()
    lower = lower_basic.copy()
    
    trend = np.ones(len(df))
    upper_final = np.zeros(len(df))
    lower_final = np.zeros(len(df))
    
    upper_final[0] = upper_basic.iloc[0]
    lower_final[0] = lower_basic.iloc[0]

    for i in range(1, len(df)):
        if upper_basic.iloc[i] < upper_final[i-1] or close.iloc[i-1] > upper_final[i-1]:
            upper_final[i] = upper_basic.iloc[i]
        else:
            upper_final[i] = upper_final[i-1]

        if lower_basic.iloc[i] > lower_final[i-1] or close.iloc[i-1] < lower_final[i-1]:
            lower_final[i] = lower_basic.iloc[i]
        else:
            lower_final[i] = lower_final[i-1]

        prev_trend = trend[i-1]
        if prev_trend == 1:
            if close.iloc[i] < lower_final[i-1]:
                trend[i] = -1
            else:
                trend[i] = 1
        else:
            if close.iloc[i] > upper_final[i-1]:
                trend[i] = 1
            else:
                trend[i] = -1

    df['Supertrend'] = np.where(trend == 1, lower_final, upper_final)
    df['ST_Trend'] = trend.astype(int)
    df['ATR'] = atr

    # EWO
    df['EWO'] = df['Close'].rolling(5).mean() - df['Close'].rolling(35).mean()

    # Aroon
    period = 14
    def pos_last_high(x): return (len(x) - 1) - int(np.argmax(x))
    def pos_last_low(x): return (len(x) - 1) - int(np.argmin(x))
    
    aroon_up = 100 * (period - df['High'].rolling(period).apply(pos_last_high, raw=True)) / period
    aroon_down = 100 * (period - df['Low'].rolling(period).apply(pos_last_low, raw=True)) / period
    df['AroonOsc'] = aroon_up - aroon_down

    return df

def check_signals(df, htf_trend):
    if 'ST_Trend' not in df.columns: 
        df['Final_Signal'] = 0 
        return df
    
    df['ST_prev'] = df['ST_Trend'].shift(1).fillna(0)
    df['Signal_Raw'] = 0
    df.loc[(df['ST_prev'] == -1) & (df['ST_Trend'] == 1), 'Signal_Raw'] = 1 
    df.loc[(df['ST_prev'] == 1) & (df['ST_Trend'] == -1), 'Signal_Raw'] = -1 

    df['Final_Signal'] = 0
    
    buy_cond = (
        (df['Signal_Raw'] == 1) & 
        (df['EWO'] > 0) & 
        (df['AroonOsc'] > 0) & 
        (htf_trend == 1)
    )
    
    sell_cond = (
        (df['Signal_Raw'] == -1) & 
        (df['EWO'] < 0) & 
        (df['AroonOsc'] < 0) & 
        (htf_trend == -1)
    )
    
    df.loc[buy_cond, 'Final_Signal'] = 1
    df.loc[sell_cond, 'Final_Signal'] = -1
    
    return df

# =====================================================================
# GLOBAL SHARED STATE
# =====================================================================
@st.cache_resource
def get_shared_state():
    return {
        "tick_queue": queue.Queue(),
        "stop_event": threading.Event(),
        "current_instrument": {"name": "NIFTY", "key": "NSE_INDEX|Nifty 50"},
        "status": "Disconnected" # New status tracker
    }

shared = get_shared_state()
tick_queue = shared["tick_queue"]
stop_event = shared["stop_event"]

# =====================================================================
# DATA LOADING
# =====================================================================
INSTRUMENTS_DIR = r"D:\bot\instruments\segment_wise"

@st.cache_data
def get_available_segments():
    if not os.path.exists(INSTRUMENTS_DIR): 
        st.sidebar.error(f"Folder not found: {INSTRUMENTS_DIR}")
        return []
    files = [os.path.splitext(f)[0] for f in os.listdir(INSTRUMENTS_DIR) if f.endswith(".parquet")]
    return files

@st.cache_data
def load_scrip_data(segment_name):
    file_path = os.path.join(INSTRUMENTS_DIR, f"{segment_name}.parquet")
    if not os.path.exists(file_path): 
        st.sidebar.error(f"File missing: {file_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_parquet(file_path)
        col_map = {}
        for col in df.columns:
            c = col.lower().strip()
            if c in ['tradingsymbol', 'trading_symbol', 'symbol', 'ticker', 'scrip']: 
                col_map[col] = 'tradingsymbol'
            if c in ['instrument_key', 'key', 'token', 'instrument_token']: 
                col_map[col] = 'instrument_key'
        
        df = df.rename(columns=col_map)
        return df
    except Exception as e: 
        st.sidebar.error(f"Error reading parquet: {e}")
        return pd.DataFrame()

# =====================================================================
# WEBSOCKET WORKER (WITH AUTO-RECONNECT)
# =====================================================================
async def websocket_worker(instrument_key):
    while not stop_event.is_set(): # Main Reconnect Loop
        token = get_access_token()
        if not token:
            shared["status"] = "Auth Failed"
            await asyncio.sleep(5)
            continue

        shared["status"] = "Connecting..."
        
        # 1. Authorize
        auth_url = "https://api.upstox.com/v3/feed/market-data-feed/authorize"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            resp = requests.get(auth_url, headers=headers, timeout=5)
            resp.raise_for_status()
            ws_uri = resp.json()["data"]["authorized_redirect_uri"]
        except Exception as e:
            shared["status"] = f"Auth Error: {e}"
            await asyncio.sleep(5)
            continue

        # 2. Connect
        try:
            async with websockets.connect(ws_uri) as ws:
                shared["status"] = "🟢 Connected"
                
                sub = {"guid": "live_feed", "method": "sub", "data": {"mode": "full", "instrumentKeys": [instrument_key]}}
                await ws.send(json.dumps(sub).encode('utf-8'))

                while not stop_event.is_set():
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        feed = pb.FeedResponse()
                        feed.ParseFromString(msg)
                        feed_dict = MessageToDict(feed, preserving_proto_field_name=False)

                        if "feeds" in feed_dict:
                            for _, fdata in feed_dict["feeds"].items():
                                full_feed = fdata.get("fullFeed", {})
                                ltpc = {}
                                vol = 0
                                if "indexFF" in full_feed: ltpc = full_feed["indexFF"].get("ltpc", {})
                                elif "marketFF" in full_feed: 
                                    ltpc = full_feed["marketFF"].get("ltpc", {})
                                    vol = int(full_feed["marketFF"].get("eFeedDetails", {}).get("tv", 0))
                                
                                if "ltp" in ltpc:
                                    tick_queue.put({
                                        "timestamp": pd.Timestamp.now(),
                                        "price": float(ltpc["ltp"]),
                                        "volume": vol
                                    })
                    except asyncio.TimeoutError:
                        # Timeout just means no new ticks in 2s, connection is likely still fine
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        shared["status"] = "🔴 Disconnected"
                        break # Break inner loop to trigger reconnect
                    except Exception:
                        break 
        except Exception as e:
            shared["status"] = f"🔴 Connection Error: {e}"
            await asyncio.sleep(5) # Wait before retrying

def start_thread(instrument_key):
    stop_event.clear()
    t = threading.Thread(target=lambda: asyncio.run(websocket_worker(instrument_key)), daemon=True)
    add_script_run_ctx(t)
    t.start()

def stop_thread():
    stop_event.set()
    shared["status"] = "Stopped"

# =====================================================================
# UI LAYOUT
# =====================================================================
with st.sidebar:
    st.header("Instrument Selection")
    
    segments = get_available_segments()
    if segments:
        selected_segment = st.selectbox("Segment", segments)
    else:
        selected_segment = None

    selected_key = None
    selected_name = None

    if selected_segment:
        df_scrips = load_scrip_data(selected_segment)
        if not df_scrips.empty and 'tradingsymbol' in df_scrips.columns:
            selected_name = st.selectbox("Scrip", df_scrips['tradingsymbol'].unique())
            if selected_name:
                row = df_scrips[df_scrips['tradingsymbol'] == selected_name].iloc[0]
                if 'instrument_key' in row:
                    selected_key = row['instrument_key']
                else:
                    st.error("Missing 'instrument_key' column.")
        elif not df_scrips.empty:
            st.error(f"❌ Missing 'tradingsymbol' column.")
            st.info(f"Found: {list(df_scrips.columns)}")

    st.divider()
    
    # STATUS INDICATOR
    status = shared.get("status", "Disconnected")
    if "Connected" in status:
        st.success(status)
    elif "Connecting" in status:
        st.warning(status)
    else:
        st.error(status)
        
    col1, col2 = st.columns(2)
    if col1.button("▶️ START", type="primary", disabled=(selected_key is None)):
        shared["current_instrument"]["name"] = selected_name
        shared["current_instrument"]["key"] = selected_key
        st.session_state["live_df"] = pd.DataFrame(columns=["timestamp", "price", "volume"])
        start_thread(selected_key)
        
    if col2.button("⏹️ STOP"): stop_thread()
    
    if st.button("♻️ Reset Data"):
        st.session_state["live_df"] = pd.DataFrame(columns=["timestamp", "price", "volume"])
        st.rerun()

# =====================================================================
# MAIN PROCESSING
# =====================================================================
if "live_df" not in st.session_state: st.session_state["live_df"] = pd.DataFrame()
while not tick_queue.empty():
    st.session_state["live_df"] = pd.concat([
        st.session_state["live_df"], 
        pd.DataFrame([tick_queue.get()])
    ], ignore_index=True)

if len(st.session_state["live_df"]) > 15000:
    st.session_state["live_df"] = st.session_state["live_df"].iloc[-15000:]

df = st.session_state["live_df"].copy()

if "timestamp" not in df.columns and not df.empty:
    st.session_state["live_df"] = pd.DataFrame(columns=["timestamp", "price", "volume"])
    st.rerun()

st.subheader(f"Analyzing: {shared['current_instrument']['name']}")

if not df.empty and "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Aggregation
    df_1m = df.set_index("timestamp").resample("1min").agg({
        "price": ["first", "max", "min", "last"]
    }).dropna()
    df_1m.columns = ["Open", "High", "Low", "Close"]

    # HTF Simulation
    df_htf = df_1m.resample(HTF_RESAMPLE).agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last"
    }).dropna()
    
    df_htf = calculate_indicators(df_htf)
    current_htf_trend = df_htf['ST_Trend'].iloc[-1] if 'ST_Trend' in df_htf.columns else 0

    # 1-min Indicators & Signals
    df_1m = calculate_indicators(df_1m)
    df_1m = check_signals(df_1m, current_htf_trend)

    # VISUALIZATION
    recent = df_1m.tail(60) 
    
    if not recent.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=recent.index, open=recent['Open'], high=recent['High'], low=recent['Low'], close=recent['Close'],
            increasing_line_color='#26A69A', decreasing_line_color='#EF5350'
        )])

        if 'Supertrend' in recent.columns:
            fig.add_trace(go.Scatter(
                x=recent.index, y=recent['Supertrend'], 
                mode='lines', line=dict(color='blue', width=1), name='Supertrend'
            ))

        if 'Final_Signal' in recent.columns:
            buys = recent[recent['Final_Signal'] == 1]
            if not buys.empty:
                fig.add_trace(go.Scatter(
                    x=buys.index, y=buys['Low'] - 10,
                    mode='markers+text', marker_symbol='triangle-up', marker_size=15, marker_color='green',
                    text="BUY", textposition="bottom center", name='Buy Signal'
                ))
                
            sells = recent[recent['Final_Signal'] == -1]
            if not sells.empty:
                fig.add_trace(go.Scatter(
                    x=sells.index, y=sells['High'] + 10,
                    mode='markers+text', marker_symbol='triangle-down', marker_size=15, marker_color='red',
                    text="SELL", textposition="top center", name='Sell Signal'
                ))

        fig.update_layout(
            template='plotly_white', height=600, xaxis_rangeslider_visible=False,
            title=f"Live Feed | HTF Trend: {'BULLISH 🟢' if current_htf_trend == 1 else 'BEARISH 🔴' if current_htf_trend == -1 else 'NEUTRAL ⚪'}"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display available columns for debugging
        cols_to_show = ['Close']
        if 'Supertrend' in recent.columns: cols_to_show.append('Supertrend')
        if 'Final_Signal' in recent.columns: cols_to_show.append('Final_Signal')
        
        st.dataframe(recent[cols_to_show].sort_index(ascending=False).head(5))
        
        if len(df_1m) < 50:
            st.info(f"⏳ Collecting data... {len(df_1m)}/50 candles ready for signal generation.")

if not stop_event.is_set():
    time.sleep(1)
    st.rerun()