# pages/13_Live_Entry_Monitor.py
"""
Live Entry Signal Monitor
Uses Upstox WebSocket feed to monitor validated stocks for entry signals in REAL-TIME

Flow:
1. Input validated stocks (from Page 10)
2. Connect to live market data feed (WebSocket)
3. Aggregate ticks → 1-min → 15-min candles
4. Calculate indicators in real-time
5. Generate INSTANT alerts when entry conditions met

Based on architecture from 11_Simple_supertrend.py
"""

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
from pathlib import Path
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.config import get_access_token
from core.api.instruments import load_segment_instruments

st.set_page_config(page_title="Live Entry Monitor", layout="wide")
st.title("🎯 Live Entry Signal Monitor")

# =====================================================================
# CONFIGURATION
# =====================================================================

# Entry signal parameters
PULLBACK_EMA_PERIOD = 21
RSI_PERIOD = 14
VOLUME_SPIKE_THRESHOLD = 1.3

# =====================================================================
# INDICATOR CALCULATIONS
# =====================================================================
def calculate_indicators(df):
    """Calculate all indicators needed for entry signals"""
    if len(df) < 50:
        return df
    
    df = df.copy()
    
    # EMAs for pullback detection
    df['EMA_9'] = df['Close'].ewm(span=9).mean()
    df['EMA_21'] = df['Close'].ewm(span=21).mean()
    df['EMA_50'] = df['Close'].ewm(span=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volume analysis
    df['Volume_MA'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # ATR for volatility
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.ewm(alpha=1/14, adjust=False).mean()
    
    return df


def detect_pullback_entry_live(df, regime='Trending Bullish'):
    """
    Detect pullback entry in real-time
    
    For Trending Bullish regime:
    1. Price pulls back to EMA21
    2. RSI cools off (40-60)
    3. Green candle forms (bounce)
    4. Volume spike confirms
    
    Returns: Dict with signal status
    """
    if len(df) < 50:
        return {'signal': False, 'status': 'Collecting data', 'progress': f'{len(df)}/50'}
    
    # Latest values
    current = df.iloc[-1]
    prev = df.iloc[-2]
    
    current_close = current['Close']
    current_rsi = current['RSI']
    current_volume_ratio = current['Volume_Ratio']
    ema_21 = current['EMA_21']
    ema_9 = current['EMA_9']
    
    # Check if in pullback zone
    dist_to_ema21 = abs(current_close - ema_21) / current_close
    near_ema21 = dist_to_ema21 < 0.01  # Within 1%
    
    dist_to_ema9 = abs(current_close - ema_9) / current_close
    near_ema9 = dist_to_ema9 < 0.005  # Within 0.5%
    
    # RSI cooled off
    rsi_cooled = 40 <= current_rsi <= 60
    
    # Bounce confirmation (green candle after red)
    current_green = current['Close'] > current['Open']
    prev_red = prev['Close'] < prev['Open']
    bounced = current_green and prev_red
    
    # Volume spike
    volume_spike = current_volume_ratio > VOLUME_SPIKE_THRESHOLD
    
    # ENTRY SIGNAL
    if (near_ema21 or near_ema9) and rsi_cooled and bounced and volume_spike:
        entry_price = current_close
        stop_loss = min(ema_21, df['Low'].iloc[-5:].min()) * 0.995
        target = entry_price + (entry_price - stop_loss) * 2
        
        return {
            'signal': True,
            'type': 'PULLBACK ENTRY',
            'status': '🚀 BUY SIGNAL',
            'entry_price': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'target': round(target, 2),
            'rsi': round(current_rsi, 1),
            'volume_ratio': round(current_volume_ratio, 2),
            'reason': f'Pullback to EMA{21 if near_ema21 else 9}, RSI {current_rsi:.1f}, Bounce + Volume {current_volume_ratio:.1f}x'
        }
    
    # WAITING STATE
    elif (near_ema21 or near_ema9) and rsi_cooled:
        return {
            'signal': False,
            'status': '⏳ WAITING',
            'reason': f'In pullback zone (RSI {current_rsi:.1f}), waiting for bounce + volume',
            'watch_for': 'Green candle with volume spike'
        }
    
    # NOT READY
    else:
        status_parts = []
        if not (near_ema21 or near_ema9):
            status_parts.append(f'Price {current_close:.2f} not near EMA21 ({ema_21:.2f})')
        if not rsi_cooled:
            status_parts.append(f'RSI {current_rsi:.1f} not in 40-60 range')
        
        return {
            'signal': False,
            'status': '👀 MONITORING',
            'reason': ' | '.join(status_parts) if status_parts else 'Conditions not met'
        }


# =====================================================================
# WEBSOCKET WORKER (Multi-Symbol)
# =====================================================================
@st.cache_resource
def get_shared_state():
    """Global state for WebSocket data"""
    return {
        "tick_queues": {},  # symbol -> queue
        "stop_event": threading.Event(),
        "status": {}  # symbol -> status
    }

shared = get_shared_state()

async def websocket_worker_multi(instrument_keys_dict):
    """
    WebSocket worker for multiple symbols
    
    Args:
        instrument_keys_dict: {symbol: instrument_key}
    """
    stop_event = shared["stop_event"]
    
    while not stop_event.is_set():
        token = get_access_token()
        if not token:
            for symbol in instrument_keys_dict.keys():
                shared["status"][symbol] = "Auth Failed"
            await asyncio.sleep(5)
            continue
        
        # Authorize
        auth_url = "https://api.upstox.com/v3/feed/market-data-feed/authorize"
        headers = {"Authorization": f"Bearer {token}"}
        
        try:
            resp = requests.get(auth_url, headers=headers, timeout=5)
            resp.raise_for_status()
            ws_uri = resp.json()["data"]["authorized_redirect_uri"]
        except Exception as e:
            for symbol in instrument_keys_dict.keys():
                shared["status"][symbol] = f"Auth Error: {e}"
            await asyncio.sleep(5)
            continue
        
        # Connect
        try:
            async with websockets.connect(ws_uri) as ws:
                # Subscribe to all symbols
                instrument_keys_list = list(instrument_keys_dict.values())
                sub = {
                    "guid": "multi_feed",
                    "method": "sub",
                    "data": {
                        "mode": "full",
                        "instrumentKeys": instrument_keys_list
                    }
                }
                await ws.send(json.dumps(sub).encode('utf-8'))
                
                for symbol in instrument_keys_dict.keys():
                    shared["status"][symbol] = "🟢 Connected"
                
                # Receive messages
                while not stop_event.is_set():
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        feed = pb.FeedResponse()
                        feed.ParseFromString(msg)
                        feed_dict = MessageToDict(feed, preserving_proto_field_name=False)
                        
                        if "feeds" in feed_dict:
                            for inst_key, fdata in feed_dict["feeds"].items():
                                # Find which symbol this belongs to
                                symbol = None
                                for sym, key in instrument_keys_dict.items():
                                    if key == inst_key:
                                        symbol = sym
                                        break
                                
                                if symbol is None:
                                    continue
                                
                                full_feed = fdata.get("fullFeed", {})
                                ltpc = {}
                                vol = 0
                                
                                if "indexFF" in full_feed:
                                    ltpc = full_feed["indexFF"].get("ltpc", {})
                                elif "marketFF" in full_feed:
                                    ltpc = full_feed["marketFF"].get("ltpc", {})
                                    vol = int(full_feed["marketFF"].get("eFeedDetails", {}).get("tv", 0))
                                
                                if "ltp" in ltpc:
                                    if symbol not in shared["tick_queues"]:
                                        shared["tick_queues"][symbol] = queue.Queue()
                                    
                                    shared["tick_queues"][symbol].put({
                                        "timestamp": pd.Timestamp.now(),
                                        "price": float(ltpc["ltp"]),
                                        "volume": vol
                                    })
                    
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        for symbol in instrument_keys_dict.keys():
                            shared["status"][symbol] = "🔴 Disconnected"
                        break
                    except Exception:
                        break
        
        except Exception as e:
            for symbol in instrument_keys_dict.keys():
                shared["status"][symbol] = f"🔴 Error: {e}"
            await asyncio.sleep(5)


def start_multi_feed(instrument_keys_dict):
    """Start WebSocket for multiple symbols"""
    shared["stop_event"].clear()
    t = threading.Thread(
        target=lambda: asyncio.run(websocket_worker_multi(instrument_keys_dict)),
        daemon=True
    )
    add_script_run_ctx(t)
    t.start()


def stop_feed():
    """Stop WebSocket"""
    shared["stop_event"].set()
    for symbol in shared["status"].keys():
        shared["status"][symbol] = "Stopped"


# =====================================================================
# UI
# =====================================================================
st.markdown("""
**Connect to live market data and get INSTANT alerts when entry signals trigger.**

**Process:**
1. Enter your validated stocks (from Page 10)
2. Click START to connect
3. System monitors all stocks simultaneously
4. Alerts you when entry conditions are met
""")

# ========== SECTION 1: Watch List Setup ==========
st.header("1️⃣ Watch List Setup")

watch_list_input = st.text_area(
    "Enter Validated Symbols (one per line)",
    value="SIEMENS\nADANIGREEN\nDLF",
    height=100,
    help="Enter symbols that passed validation on Page 10"
)

# Manual instrument key option
st.markdown("---")
manual_mode = st.checkbox("⚙️ Manual Instrument Key Entry", 
                          help="Use this if instrument files are missing but you know the keys")

manual_keys = {}
if manual_mode:
    st.info("💡 Find instrument keys on Upstox: Login → Market → Symbol → Copy instrument_key")
    
    if watch_list_input:
        watch_symbols = [s.strip() for s in watch_list_input.split('\n') if s.strip()]
        
        for symbol in watch_symbols:
            key = st.text_input(f"Instrument Key for {symbol}", 
                              key=f"manual_key_{symbol}",
                              placeholder="NSE_EQ|INE...")
            if key:
                manual_keys[symbol] = key

if watch_list_input:
    watch_symbols = [s.strip() for s in watch_list_input.split('\n') if s.strip()]
    st.success(f"✅ Monitoring {len(watch_symbols)} stocks: {', '.join(watch_symbols)}")
else:
    st.stop()

# Get instrument keys (manual or from file)
instrument_keys_dict = {}
missing_symbols = []

if manual_mode and manual_keys:
    # Use manual keys
    instrument_keys_dict = manual_keys
    st.success(f"✅ Using manual instrument keys for {len(manual_keys)} symbols")
else:
    # Load from instruments file using existing infrastructure
    instruments_df = load_segment_instruments("NSE_EQ")
    
    if instruments_df.empty:
        st.error("❌ Instrument data not found.")
        
        # Diagnostics
        st.subheader("🔍 Diagnostics")
        
        # Show expected location based on your infrastructure
        from core.api.instruments import SEGMENT_DIR
        expected_path = SEGMENT_DIR / "NSE_EQ.parquet"
        
        st.code(f"Looking for: {expected_path}")
        
        if expected_path.exists():
            st.success(f"✅ File exists at: {expected_path}")
            st.warning("⚠️ File exists but failed to load. Check file format.")
            try:
                test_df = pd.read_parquet(expected_path)
                st.write("Columns found:", list(test_df.columns))
                st.write("Sample data:")
                st.dataframe(test_df.head())
            except Exception as e:
                st.error(f"Error reading file: {e}")
        else:
            st.warning(f"❌ File not found at: {expected_path}")
            
            # Check if directory exists
            if SEGMENT_DIR.exists():
                st.info(f"✅ Instruments directory exists: {SEGMENT_DIR}")
                st.write("Files found:")
                for file in SEGMENT_DIR.glob("*.parquet"):
                    st.write(f"  - {file.name}")
            else:
                st.error(f"❌ Instruments directory not found: {SEGMENT_DIR}")
            
            st.markdown("---")
            st.subheader("📋 How to Fix")
            st.markdown("""
            **Option 1: Download Instruments**
            1. Go to **Page 1: Login & Instruments**
            2. Click "Download Instruments" or "Download All Segments"
            3. Wait for download to complete
            4. Come back to this page
            
            **Option 2: Manual Symbol Entry**
            - Check the "Manual Instrument Key Entry" box above
            - Enter instrument keys for each symbol
            """)
        
        st.stop()
    
    # Normalize column names (in case they're different)
    col_map = {}
    for col in instruments_df.columns:
        c = col.lower().strip()
        if c in ['tradingsymbol', 'trading_symbol', 'symbol']:
            col_map[col] = 'tradingsymbol'
        if c in ['instrument_key', 'key', 'token']:
            col_map[col] = 'instrument_key'
    
    if col_map:
        instruments_df = instruments_df.rename(columns=col_map)
    
    # Verify required columns exist
    if 'tradingsymbol' not in instruments_df.columns or 'instrument_key' not in instruments_df.columns:
        st.error("❌ Instrument file is missing required columns")
        st.write("Expected: tradingsymbol, instrument_key")
        st.write("Found:", list(instruments_df.columns))
        st.stop()
    
    # Get keys from instruments file
    for symbol in watch_symbols:
        match = instruments_df[instruments_df['tradingsymbol'] == symbol]
        if not match.empty:
            instrument_keys_dict[symbol] = match.iloc[0]['instrument_key']
        else:
            missing_symbols.append(symbol)

if missing_symbols:
    st.warning(f"⚠️ Could not find instrument keys for: {', '.join(missing_symbols)}")

if not instrument_keys_dict:
    st.error("❌ No valid symbols found")
    st.stop()

# ========== SECTION 2: Live Feed Controls ==========
st.header("2️⃣ Live Feed Control")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("▶️ START MONITORING", type="primary"):
        # Initialize data storage
        for symbol in instrument_keys_dict.keys():
            st.session_state[f"live_df_{symbol}"] = pd.DataFrame(columns=["timestamp", "price", "volume"])
            st.session_state[f"signal_status_{symbol}"] = "Starting..."
        
        # Start WebSocket
        start_multi_feed(instrument_keys_dict)
        st.success("✅ Live feed started")
        time.sleep(1)
        st.rerun()

with col2:
    if st.button("⏹️ STOP"):
        stop_feed()
        st.info("⏹️ Stopped")

with col3:
    if st.button("♻️ RESET"):
        for symbol in instrument_keys_dict.keys():
            st.session_state[f"live_df_{symbol}"] = pd.DataFrame(columns=["timestamp", "price", "volume"])
        st.rerun()

# ========== SECTION 3: Live Monitoring ==========
st.header("3️⃣ Live Signal Status")

# Process ticks for each symbol
for symbol in instrument_keys_dict.keys():
    # Initialize if needed
    if f"live_df_{symbol}" not in st.session_state:
        st.session_state[f"live_df_{symbol}"] = pd.DataFrame(columns=["timestamp", "price", "volume"])
    
    # Get ticks from queue
    if symbol in shared["tick_queues"]:
        while not shared["tick_queues"][symbol].empty():
            tick = shared["tick_queues"][symbol].get()
            st.session_state[f"live_df_{symbol}"] = pd.concat([
                st.session_state[f"live_df_{symbol}"],
                pd.DataFrame([tick])
            ], ignore_index=True)
    
    # Limit size
    if len(st.session_state[f"live_df_{symbol}"]) > 5000:
        st.session_state[f"live_df_{symbol}"] = st.session_state[f"live_df_{symbol}"].iloc[-5000:]

# Display each symbol
for symbol in instrument_keys_dict.keys():
    with st.expander(f"📊 {symbol} - {shared['status'].get(symbol, 'Not Started')}", expanded=True):
        
        df_ticks = st.session_state[f"live_df_{symbol}"].copy()
        
        if df_ticks.empty:
            st.info("⏳ Waiting for data...")
            continue
        
        # Aggregate to 1-minute candles
        df_ticks['timestamp'] = pd.to_datetime(df_ticks['timestamp'])
        df_1m = df_ticks.set_index('timestamp').resample('1min').agg({
            'price': ['first', 'max', 'min', 'last'],
            'volume': 'sum'
        }).dropna()
        
        df_1m.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Calculate indicators
        df_1m = calculate_indicators(df_1m)
        
        # Detect entry signal
        signal_result = detect_pullback_entry_live(df_1m, regime='Trending Bullish')
        
        # Display status
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = df_1m['Close'].iloc[-1] if len(df_1m) > 0 else 0
            st.metric("Current Price", f"₹{current_price:.2f}")
        
        with col2:
            status = signal_result['status']
            if '🚀' in status:
                st.success(status)
            elif '⏳' in status:
                st.warning(status)
            else:
                st.info(status)
        
        with col3:
            st.metric("Candles", f"{len(df_1m)}/50")
        
        with col4:
            if 'rsi' in signal_result:
                st.metric("RSI", signal_result['rsi'])
        
        # If signal triggered
        if signal_result['signal']:
            st.balloons()
            
            st.success(f"### 🚀 BUY SIGNAL TRIGGERED!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Entry Price", f"₹{signal_result['entry_price']}")
            col2.metric("Stop Loss", f"₹{signal_result['stop_loss']}")
            col3.metric("Target", f"₹{signal_result['target']}")
            
            st.info(f"**Reason:** {signal_result['reason']}")
            
            # Execution instructions
            st.markdown("### 📋 Execute Now:")
            st.code(f"""
1. Place LIMIT order: {symbol} @ ₹{signal_result['entry_price']}
2. Set STOP LOSS: ₹{signal_result['stop_loss']}
3. Set TARGET: ₹{signal_result['target']}
4. Calculate position size (1% risk)
            """)
        
        # Show chart
        if len(df_1m) >= 20:
            recent = df_1m.tail(60)
            
            fig = go.Figure(data=[go.Candlestick(
                x=recent.index,
                open=recent['Open'],
                high=recent['High'],
                low=recent['Low'],
                close=recent['Close']
            )])
            
            # Add EMA lines
            if 'EMA_21' in recent.columns:
                fig.add_trace(go.Scatter(
                    x=recent.index, y=recent['EMA_21'],
                    mode='lines', name='EMA 21',
                    line=dict(color='orange', width=1)
                ))
            
            if 'EMA_9' in recent.columns:
                fig.add_trace(go.Scatter(
                    x=recent.index, y=recent['EMA_9'],
                    mode='lines', name='EMA 9',
                    line=dict(color='blue', width=1)
                ))
            
            fig.update_layout(
                height=400,
                xaxis_rangeslider_visible=False,
                title=f"{symbol} - 1min Chart"
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Auto-refresh
if not shared["stop_event"].is_set():
    time.sleep(1)
    st.rerun()