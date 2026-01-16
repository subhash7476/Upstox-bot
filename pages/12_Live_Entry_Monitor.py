# pages/12_Live_Entry_Monitor.py
"""
Live Entry Monitor - FINAL WORKING VERSION
- Loads shortlisted stocks from data/state/shortlisted_stocks.csv
- Uses Upstox v2 API with symbol mapping fix
- Works with TradingDB structure
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import json
import time
import os
from pathlib import Path
from datetime import datetime, date, timedelta
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.config import get_access_token
from core.database import TradingDB
from core.indicators import compute_supertrend

st.set_page_config(page_title="Live Entry Monitor", layout="wide", page_icon="🎯")

# =====================================================================
# CONFIGURATION
# =====================================================================

# Try multiple possible locations for shortlisted stocks
RESULTS_PATHS = [
    "data/state/shortlisted_stocks.csv",  # Primary (from Daily Analyzer)
    "data/daily_analyzer_results.csv",     # Legacy
    "data/shortlisted_stocks.csv"          # Alternative
]

LIVE_CACHE_TABLE = "live_ohlcv_cache"

# =====================================================================
# API CLASS WITH SYMBOL MAPPING FIX
# =====================================================================

class UpstoxMarketData:
    """
    Upstox Market Data API v2
    Handles NSE_EQ|ISIN (request) vs NSE_EQ:SYMBOL (response)
    """
    
    BASE_URL = "https://api.upstox.com/v2"
    
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }
    
    def get_market_quote(self, instrument_keys: list, symbol_map: dict) -> dict:
        """
        Get market quotes
        
        Args:
            instrument_keys: List of NSE_EQ|INE... keys
            symbol_map: Dict mapping instrument_key -> trading_symbol
        
        Returns:
            Dict mapping instrument_key -> quote data
        """
        if not instrument_keys:
            return {}
        
        keys_param = ",".join(instrument_keys)
        url = f"{self.BASE_URL}/market-quote/quotes"
        params = {"instrument_key": keys_param}
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') != 'success':
                raise Exception(f"API Error: {data.get('message')}")
            
            # Map response keys back to instrument keys
            return self._map_response(data.get('data', {}), instrument_keys, symbol_map)
            
        except Exception as e:
            st.error(f"API Error: {e}")
            return {}
    
    def _map_response(self, response_data: dict, inst_keys: list, symbol_map: dict) -> dict:
        """Map response (NSE_EQ:SYMBOL) back to instrument keys (NSE_EQ|ISIN)"""
        quotes = {}
        
        # Build reverse lookup: symbol -> inst_key
        symbol_to_key = {v: k for k, v in symbol_map.items()}
        
        for response_key, quote_data in response_data.items():
            if not isinstance(quote_data, dict):
                continue
            
            # Extract symbol from quote
            symbol = quote_data.get('symbol')
            
            if symbol and symbol in symbol_to_key:
                inst_key = symbol_to_key[symbol]
                quotes[inst_key] = quote_data
        
        return quotes
    
    def build_1min_candle(self, quote_data: dict) -> dict:
        """Convert quote to candle"""
        ohlc = quote_data.get('ohlc', {})
        now = datetime.now()
        timestamp = now.replace(second=0, microsecond=0)
        
        return {
            'timestamp': timestamp,
            'open': float(ohlc.get('open', 0)),
            'high': float(ohlc.get('high', 0)),
            'low': float(ohlc.get('low', 0)),
            'close': float(quote_data.get('last_price', ohlc.get('close', 0))),
            'volume': int(quote_data.get('volume', 0))
        }

# =====================================================================
# DATABASE FUNCTIONS
# =====================================================================

@st.cache_resource
def get_db():
    """Get database connection"""
    return TradingDB()

def init_live_cache_table():
    """Initialize live cache table"""
    db = get_db()
    
    db.con.execute(f"""
        CREATE TABLE IF NOT EXISTS {LIVE_CACHE_TABLE} (
            symbol VARCHAR,
            timestamp TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT,
            PRIMARY KEY (symbol, timestamp)
        )
    """)

def fetch_historical_ohlcv(symbol: str, instrument_key: str, days: int = 5) -> pd.DataFrame:
    """Fetch historical 1-minute OHLCV data using instrument_key"""
    db = get_db()
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    
    query = f"""
        SELECT 
            timestamp,
            open as Open,
            high as High,
            low as Low,
            close as Close,
            volume as Volume
        FROM ohlcv_1m
        WHERE instrument_key = '{instrument_key}'
          AND DATE(timestamp) >= '{start_date}'
          AND DATE(timestamp) < '{end_date}'
        ORDER BY timestamp
    """
    
    try:
        df = db.con.execute(query).df()
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching historical data for {symbol}: {e}")
        return pd.DataFrame()

def get_todays_cached_data(symbol: str) -> pd.DataFrame:
    """
    Get today's cached data from live_ohlcv_cache
    This fills the gap between yesterday's close and current time
    """
    db = get_db()
    
    today = datetime.now().date()
    
    query = f"""
        SELECT 
            timestamp,
            open as Open,
            high as High,
            low as Low,
            close as Close,
            volume as Volume
        FROM {LIVE_CACHE_TABLE}
        WHERE symbol = '{symbol}'
          AND DATE(timestamp) = '{today}'
        ORDER BY timestamp
    """
    
    try:
        df = db.con.execute(query).df()
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        # Silently return empty - error will show in main UI
        return pd.DataFrame()

def save_live_candle(symbol: str, candle_data: dict):
    """Save current candle to cache"""
    db = get_db()
    
    try:
        db.con.execute(f"""
            INSERT OR REPLACE INTO {LIVE_CACHE_TABLE}
            (symbol, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            symbol,
            candle_data['timestamp'],
            candle_data['open'],
            candle_data['high'],
            candle_data['low'],
            candle_data['close'],
            candle_data['volume']
        ])
    except Exception as e:
        st.warning(f"Error caching {symbol}: {e}")

def get_instrument_key(symbol: str) -> str:
    """Get instrument key for symbol"""
    db = get_db()
    
    query = f"""
        SELECT instrument_key
        FROM instruments
        WHERE trading_symbol = '{symbol}'
          AND segment = 'NSE_EQ'
        LIMIT 1
    """
    
    try:
        result = db.con.execute(query).fetchone()
        return result[0] if result else None
    except:
        return None

# =====================================================================
# LOAD SHORTLISTED STOCKS
# =====================================================================

def load_shortlisted_stocks() -> pd.DataFrame:
    """
    Load shortlisted stocks from multiple possible locations
    """
    for results_path in RESULTS_PATHS:
        file_path = Path(results_path)
        
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                
                # Validate required columns
                required_cols = ['Symbol']
                if all(col in df.columns for col in required_cols):
                    st.info(f"✅ Loaded from: `{results_path}`")
                    return df
                    
            except Exception as e:
                st.warning(f"Error reading {results_path}: {e}")
                continue
    
    return pd.DataFrame()

# =====================================================================
# INDICATORS & SIGNALS
# =====================================================================

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators"""
    if len(df) < 50:
        return df
    
    df = df.copy()
    
    # EMAs
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volume MA
    df['Volume_MA'] = df['Volume'].rolling(20).mean()
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    return df

def detect_entry_signal(df: pd.DataFrame, regime: str = "Trending Bullish") -> dict:
    """Detect entry signals"""
    if len(df) < 50:
        return {
            'signal': False,
            'status': f'⏳ Building data ({len(df)}/50 candles)',
            'reason': 'Insufficient data'
        }
    
    current = df.iloc[-1]
    
    # Check for NaN
    if pd.isna(current['Close']) or pd.isna(current['EMA_21']):
        return {
            'signal': False,
            'status': '⚠️ Calculating indicators...',
            'reason': 'NaN values'
        }
    
    # Signal conditions
    near_ema = abs(current['Close'] - current['EMA_21']) / current['Close'] < 0.01
    rsi_ok = 40 <= current['RSI'] <= 60
    bullish_ema = current['EMA_9'] > current['EMA_21']
    
    # Check regime
    if "Bullish" in regime:
        if near_ema and rsi_ok and bullish_ema:
            atr = current['ATR'] if not pd.isna(current['ATR']) else (current['Close'] * 0.02)
            
            entry = current['Close']
            sl = current['EMA_21'] * 0.995
            tp = entry + (2 * atr)
            
            return {
                'signal': True,
                'status': '🚀 BUY SIGNAL',
                'entry_price': entry,
                'stop_loss': sl,
                'take_profit': tp,
                'risk': entry - sl,
                'reward': tp - entry,
                'reason': f'Price near EMA21, RSI={current["RSI"]:.1f}, Bullish EMA'
            }
    
    # No signal
    reasons = []
    if not near_ema:
        pct_from_ema = abs(current['Close'] - current['EMA_21'])/current['Close']*100
        reasons.append(f"{pct_from_ema:.2f}% from EMA21")
    if not rsi_ok:
        reasons.append(f"RSI={current['RSI']:.1f}")
    if not bullish_ema:
        reasons.append("EMAs bearish")
    
    return {
        'signal': False,
        'status': '👀 MONITORING',
        'reason': ' | '.join(reasons)
    }

# =====================================================================
# MAIN UI
# =====================================================================

def main():
    st.title("🎯 Live Entry Monitor")
    st.caption("Real-time monitoring for Daily Analyzer shortlisted stocks")
    
    # Initialize
    init_live_cache_table()
    
    # Load shortlisted stocks
    st.header("1️⃣ Shortlisted Stocks")
    
    results_df = load_shortlisted_stocks()
    
    if results_df.empty:
        st.warning("⚠️ No shortlisted stocks found")
        st.info("**Checked locations:**")
        for path in RESULTS_PATHS:
            exists = "✅" if Path(path).exists() else "❌"
            st.write(f"{exists} `{path}`")
        
        st.info("💡 **Solution:** Run the Daily Regime Analyzer first (Page 9)")
        st.stop()
    
    st.success(f"✅ {len(results_df)} stocks shortlisted")
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    # Select stocks
    selected_symbols = st.multiselect(
        "Select stocks to monitor",
        options=results_df['Symbol'].tolist(),
        default=results_df['Symbol'].tolist()[:3]
    )
    
    if not selected_symbols:
        st.warning("Please select at least one stock")
        st.stop()
    
    # Get instrument keys and build symbol map
    st.header("2️⃣ Instrument Mapping")
    
    instrument_map = {}
    symbol_map = {}
    regime_map = {}
    
    for _, row in results_df.iterrows():
        symbol = row['Symbol']
        if symbol in selected_symbols:
            inst_key = get_instrument_key(symbol)
            if inst_key:
                instrument_map[symbol] = inst_key
                symbol_map[inst_key] = symbol  # For API mapping
                regime_map[symbol] = row.get('Regime', 'Unknown')
            else:
                st.warning(f"⚠️ No instrument key for {symbol}")
    
    if not instrument_map:
        st.error("No valid instruments found")
        st.stop()
    
    st.success(f"✅ Mapped {len(instrument_map)} instruments")
    
    # Controls
    st.header("3️⃣ Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        refresh_interval = st.number_input(
            "Refresh (seconds)",
            min_value=10,
            max_value=300,
            value=60,
            step=10
        )
    
    with col2:
        if st.button("🔄 Refresh Now", type="primary"):
            st.rerun()
    
    with col3:
        auto_refresh = st.checkbox("Auto Refresh", value=True)
    
    # Fetch live data
    st.header("4️⃣ Live Data")
    
    token = get_access_token()
    if not token:
        st.error("No access token")
        st.stop()
    
    api = UpstoxMarketData(token)
    
    with st.spinner("Fetching market quotes..."):
        inst_keys = list(instrument_map.values())
        market_quotes = api.get_market_quote(inst_keys, symbol_map)
    
    if not market_quotes:
        st.error("Failed to fetch quotes")
        st.stop()
    
    # Process each symbol
    for symbol in selected_symbols:
        if symbol not in instrument_map:
            continue
        
        inst_key = instrument_map[symbol]
        regime = regime_map.get(symbol, 'Unknown')
        quote_data = market_quotes.get(inst_key, {})
        
        if not quote_data:
            st.warning(f"No data for {symbol}")
            continue
        
        # DON'T save current quote as a candle - it's day-level OHLC, not 1-minute!
        # The quote API returns:
        #   ohlc.open = today's open (9:15 AM)
        #   ohlc.high = today's high (from any time)
        #   ohlc.low = today's low (from any time)
        #   last_price = current price
        # This creates fake candles with wrong OHLC!
        
        # Instead, rely on backfilled data which has real 1-minute candles
        # current_candle = api.build_1min_candle(quote_data)
        # save_live_candle(symbol, current_candle)  # REMOVED - causes fake candles
        
        # Load data
        hist_df = fetch_historical_ohlcv(symbol, inst_key, days=5)
        live_df = get_todays_cached_data(symbol)
        
        # Debug info
        with st.expander(f"🔍 Data Debug: {symbol}", expanded=False):
            st.write(f"**Historical data:** {len(hist_df)} candles")
            if not hist_df.empty:
                st.write(f"  - From: {hist_df.index.min()}")
                st.write(f"  - To: {hist_df.index.max()}")
            
            st.write(f"**Live cache data:** {len(live_df)} candles")
            if not live_df.empty:
                st.write(f"  - From: {live_df.index.min()}")
                st.write(f"  - To: {live_df.index.max()}")
        
        # Combine
        if not hist_df.empty and not live_df.empty:
            combined_df = pd.concat([hist_df, live_df])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df.sort_index(inplace=True)
            st.info(f"✅ Combined: {len(hist_df)} historical + {len(live_df)} live = {len(combined_df)} total candles")
        elif not hist_df.empty:
            combined_df = hist_df
            st.info(f"📊 Using historical only: {len(combined_df)} candles")
        elif not live_df.empty:
            combined_df = live_df
            st.info(f"📊 Using live cache only: {len(combined_df)} candles")
        else:
            st.warning(f"No data for {symbol}")
            continue
        
        # Calculate indicators
        combined_df = calculate_indicators(combined_df)
        
        # Detect signal
        signal = detect_entry_signal(combined_df, regime)
        
        # Display
        with st.expander(f"📊 {symbol} - {regime}", expanded=True):
            
            # Current price
            ohlc = quote_data.get('ohlc', {})
            last_price = quote_data.get('last_price', 0)
            volume = quote_data.get('volume', 0)
            
            open_price = ohlc.get('open', 0)
            change = ((last_price - open_price) / open_price * 100) if open_price > 0 else 0
            
            # Show live price indicator
            st.info(f"📡 **LIVE:** Current price from API (not on chart) - Use backfill for chart updates")
            
            # Metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("💹 LTP", f"₹{last_price:.2f}", f"{change:+.2f}%")
            
            with col2:
                st.metric("🔺 High", f"₹{ohlc.get('high', 0):.2f}")
            
            with col3:
                st.metric("🔻 Low", f"₹{ohlc.get('low', 0):.2f}")
            
            with col4:
                vol_str = f"{volume/1000000:.2f}M" if volume >= 1000000 else f"{volume/1000:.1f}K"
                st.metric("📊 Volume", vol_str)
            
            with col5:
                st.metric("📈 Candles", len(combined_df))
            
            st.divider()
            
            # Add backfill button
            col_a, col_b = st.columns([3, 1])
            
            with col_a:
                st.caption(f"💡 **Tip:** Use backfill to get today's complete 1-minute candle data")
            
            with col_b:
                if st.button(f"🔄 Backfill", key=f"backfill_{symbol}", help="Fetch today's full intraday data", use_container_width=True):
                    with st.spinner(f"Backfilling {symbol}..."):
                        try:
                            # Fetch intraday data
                            url = f"https://api.upstox.com/v2/historical-candle/intraday/{inst_key}/1minute"
                            
                            headers = {
                                'Accept': 'application/json',
                                'Authorization': f'Bearer {token}'
                            }
                            
                            response = requests.get(url, headers=headers, timeout=30)
                            
                            if response.status_code == 200:
                                data = response.json()
                                
                                if data.get('status') == 'success':
                                    candles = data.get('data', {}).get('candles', [])
                                    
                                    if candles:
                                        db = get_db()
                                        inserted = 0
                                        today = datetime.now().date()
                                        
                                        for candle in candles:
                                            ts = pd.to_datetime(candle[0])
                                            
                                            if ts.date() == today:
                                                try:
                                                    db.con.execute(f"""
                                                        INSERT OR REPLACE INTO {LIVE_CACHE_TABLE}
                                                        (symbol, timestamp, open, high, low, close, volume)
                                                        VALUES (?, ?, ?, ?, ?, ?, ?)
                                                    """, [
                                                        symbol,
                                                        ts,
                                                        float(candle[1]),
                                                        float(candle[2]),
                                                        float(candle[3]),
                                                        float(candle[4]),
                                                        int(candle[5])
                                                    ])
                                                    inserted += 1
                                                except:
                                                    pass
                                        
                                        st.success(f"✅ Inserted {inserted} candles for {symbol}")
                                        st.rerun()
                                    else:
                                        st.warning("No candles received")
                                else:
                                    st.error(f"API Error: {data.get('message')}")
                            else:
                                st.error(f"HTTP Error: {response.status_code}")
                        
                        except Exception as e:
                            st.error(f"Error: {e}")
            
            st.divider()
            
            # Signal
            if signal['signal']:
                st.success(f"### {signal['status']}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Trade Details:**")
                    st.write(f"- Entry: ₹{signal['entry_price']:.2f}")
                    st.write(f"- Stop Loss: ₹{signal['stop_loss']:.2f}")
                    st.write(f"- Take Profit: ₹{signal['take_profit']:.2f}")
                
                with col2:
                    st.write("**Risk/Reward:**")
                    st.write(f"- Risk: ₹{signal['risk']:.2f}")
                    st.write(f"- Reward: ₹{signal['reward']:.2f}")
                    rr = signal['reward'] / signal['risk'] if signal['risk'] > 0 else 0
                    st.write(f"- R:R: 1:{rr:.2f}")
                
                st.info(f"**Reason:** {signal['reason']}")
            else:
                st.info(f"**Status:** {signal['status']}")
                st.caption(f"Reason: {signal['reason']}")
            
            # Chart
            if len(combined_df) >= 30:
                with st.expander("📈 Chart", expanded=False):
                    # Controls row
                    col_range, col_volume = st.columns([3, 1])
                    
                    with col_range:
                        # Show different time ranges based on selection
                        chart_range = st.radio(
                            "Time Range",
                            options=["Last Hour (60min)", "Last 2 Hours (120min)", "Today Only", "Full Data"],
                            index=1,
                            horizontal=True,
                            key=f"chart_range_{symbol}"
                        )
                    
                    with col_volume:
                        show_volume = st.checkbox("Show Volume", value=False, key=f"volume_{symbol}")
                    
                    # Select data based on range
                    if chart_range == "Last Hour (60min)":
                        chart_df = combined_df.tail(60)
                    elif chart_range == "Last 2 Hours (120min)":
                        chart_df = combined_df.tail(120)
                    elif chart_range == "Today Only":
                        # Get only today's data
                        today = datetime.now().date()
                        chart_df = combined_df[combined_df.index.date == today]
                    else:  # Full Data
                        chart_df = combined_df.tail(390)  # Last full trading day
                    
                    if len(chart_df) == 0:
                        st.warning("No data for selected range")
                    else:
                        fig = go.Figure()
                        
                        # Candlestick
                        fig.add_trace(go.Candlestick(
                            x=chart_df.index,
                            open=chart_df['Open'],
                            high=chart_df['High'],
                            low=chart_df['Low'],
                            close=chart_df['Close'],
                            name='Price',
                            increasing_line_color='#26a69a',
                            decreasing_line_color='#ef5350'
                        ))
                        
                        # EMA 21
                        if 'EMA_21' in chart_df.columns:
                            fig.add_trace(go.Scatter(
                                x=chart_df.index,
                                y=chart_df['EMA_21'],
                                mode='lines',
                                name='EMA 21',
                                line=dict(color='orange', width=2),
                                opacity=0.8
                            ))
                        
                        # EMA 9
                        if 'EMA_9' in chart_df.columns:
                            fig.add_trace(go.Scatter(
                                x=chart_df.index,
                                y=chart_df['EMA_9'],
                                mode='lines',
                                name='EMA 9',
                                line=dict(color='blue', width=1.5),
                                opacity=0.7
                            ))
                        
                        # Volume (optional)
                        if show_volume:
                            fig.add_trace(go.Bar(
                                x=chart_df.index,
                                y=chart_df['Volume'],
                                name='Volume',
                                yaxis='y2',
                                opacity=0.2,
                                marker_color='lightgray',
                                showlegend=False,
                                hovertemplate='Volume: %{y:,}<extra></extra>'
                            ))
                        
                        # Calculate proper y-axis ranges
                        price_min = chart_df['Low'].min()
                        price_max = chart_df['High'].max()
                        price_range = price_max - price_min
                        price_padding = price_range * 0.05  # 5% padding
                        
                        # Base layout (without volume)
                        layout_config = {
                            'height': 500,
                            'xaxis_rangeslider_visible': False,
                            'margin': dict(l=10, r=10, t=30, b=0),
                            'hovermode': 'x unified',
                            'template': 'plotly_white',
                            'showlegend': True,
                            'legend': dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ),
                            'xaxis': dict(
                                title="Time",
                                gridcolor='#e8e8e8',
                                showgrid=True
                            ),
                            'yaxis': dict(
                                title="Price (₹)",
                                gridcolor='#e8e8e8',
                                showgrid=True,
                                range=[price_min - price_padding, price_max + price_padding],
                                fixedrange=False
                            )
                        }
                        
                        # Add volume axis if enabled
                        if show_volume:
                            volume_max = chart_df['Volume'].max()
                            layout_config['yaxis2'] = dict(
                                overlaying='y',
                                side='right',
                                showticklabels=False,
                                showgrid=False,
                                range=[0, volume_max * 8],
                                fixedrange=True
                            )
                        
                        fig.update_layout(**layout_config)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Data info
                        st.caption(f"Showing {len(chart_df)} candles | From: {chart_df.index.min()} | To: {chart_df.index.max()}")
    
    # Auto refresh
    if auto_refresh:
        st.caption(f"⏱️ Refreshing in {refresh_interval} seconds...")
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()