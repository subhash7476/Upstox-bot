# pages/11_Entry_Timing_System.py
"""
Entry Timing & Signal Generation System
Takes validated stocks from regime analysis and finds optimal entry points

Flow:
1. Import validated stocks (from page 10)
2. Monitor for entry triggers
3. Multi-timeframe confirmation
4. Generate BUY signals with exact entry price, SL, TP
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import os
from datetime import datetime, timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from core.entry_signals import (
        generate_entry_signal,
        calculate_multi_timeframe_trend,
        detect_pullback_entry,
        detect_extreme_entry,
        detect_momentum_breakout,
        detect_bb_squeeze_breakout
    )
    from core.regime_gmm import MarketRegimeGMM
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

DERIVED_ROOT = Path("data/derived")

st.set_page_config(layout="wide", page_title="Entry Timing System")
st.title("🎯 Entry Timing & Signal Generation")

st.markdown("""
**Purpose:** Find the optimal entry point for validated stocks.

**Process:**
1. Enter your validated stocks (from page 10)
2. System monitors for entry triggers
3. Shows BUY signal with exact entry price, SL, TP
4. Multi-timeframe confirmation
""")

# ========== SECTION 1: Watch List Management ==========
st.header("1️⃣ Watch List Management")

st.markdown("""
**From Page 10 (Regime Validator)**, you identified stocks with:
- Historical Precision > 60%
- Trade Win Rate > 55%
- Current regime: Trending Bullish

**Enter those stocks here to monitor for entry signals.**
""")

# Input validated stocks
col1, col2 = st.columns([3, 1])

with col1:
    watch_list_input = st.text_area(
        "Enter Validated Stocks (one per line)",
        value="SIEMENS\nADANIGREEN",
        height=100,
        help="Enter only stocks that passed validation from page 10"
    )

with col2:
    st.markdown("### Current Regime")
    default_regime = st.selectbox(
        "Regime Type",
        ['Trending Bullish', 'Trending Bearish', 'Quiet Bullish', 'Ranging', 'Volatile'],
        help="From your regime analysis"
    )

# Parse watch list
if watch_list_input:
    watch_symbols = [s.strip() for s in watch_list_input.split('\n') if s.strip()]
    st.success(f"Monitoring {len(watch_symbols)} stocks: {', '.join(watch_symbols)}")
else:
    st.warning("Enter at least one symbol")
    st.stop()

# ========== SECTION 2: Entry Strategy Selection ==========
st.header("2️⃣ Entry Strategy")

st.markdown("""
**Strategy depends on regime type:**
- **Trending**: Wait for pullback to value (9/21 EMA)
- **Ranging**: Enter at extremes (RSI < 30 or > 70)
- **Volatile**: Momentum breakout with volume
""")

col1, col2, col3 = st.columns(3)

with col1:
    entry_strategy = st.selectbox(
        "Entry Strategy",
        ['auto', 'pullback', 'extreme', 'momentum', 'squeeze'],
        help="Auto selects based on regime type"
    )

with col2:
    min_risk_reward = st.number_input(
        "Min Risk/Reward",
        1.0, 5.0, 2.0, 0.5,
        help="Minimum 1:2 R/R ratio"
    )

with col3:
    require_mtf_align = st.checkbox(
        "Require MTF Alignment",
        value=True,
        help="All timeframes must align (safer but fewer signals)"
    )

# ========== SECTION 3: Scan for Entry Signals ==========
st.header("3️⃣ Entry Signals Scanner")

st.markdown("Click to scan all watch list stocks for entry signals (uses latest data).")

if st.button("🔍 Scan for Entry Signals", key="scan"):
    
    signals_found = []
    waiting_stocks = []
    no_signal_stocks = []
    
    progress = st.progress(0)
    status_text = st.empty()
    
    for i, symbol in enumerate(watch_symbols):
        status_text.text(f"Scanning {symbol}... ({i+1}/{len(watch_symbols)})")
        
        # Load data for all timeframes
        try:
            # 5-minute
            path_5m = DERIVED_ROOT / symbol / "5minute"
            files_5m = list(path_5m.glob("*.parquet"))
            if not files_5m:
                progress.progress((i + 1) / len(watch_symbols))
                continue
            df_5m = pd.read_parquet(files_5m[0]).tail(200)
            
            # 15-minute
            path_15m = DERIVED_ROOT / symbol / "15minute"
            files_15m = list(path_15m.glob("*.parquet"))
            if not files_15m:
                progress.progress((i + 1) / len(watch_symbols))
                continue
            df_15m = pd.read_parquet(files_15m[0]).tail(200)
            
            # 60-minute
            path_60m = DERIVED_ROOT / symbol / "60minute"
            files_60m = list(path_60m.glob("*.parquet"))
            if not files_60m:
                progress.progress((i + 1) / len(watch_symbols))
                continue
            df_60m = pd.read_parquet(files_60m[0]).tail(200)
            
            # Daily
            path_daily = DERIVED_ROOT / symbol / "1day"
            files_daily = list(path_daily.glob("*.parquet"))
            if not files_daily:
                progress.progress((i + 1) / len(watch_symbols))
                continue
            df_daily = pd.read_parquet(files_daily[0]).tail(200)
            
            # Convert index to datetime
            for df in [df_5m, df_15m, df_60m, df_daily]:
                df.index = pd.to_datetime(df.index)
            
            # Generate entry signal
            signal = generate_entry_signal(
                symbol=symbol,
                regime=default_regime,
                df_5m=df_5m,
                df_15m=df_15m,
                df_60m=df_60m,
                df_daily=df_daily,
                strategy=entry_strategy
            )
            
            # Validate risk/reward
            if signal.get('signal', False):
                rr = signal.get('risk_reward', 0)
                if rr >= min_risk_reward:
                    # Check MTF alignment if required
                    if require_mtf_align and not signal['multi_timeframe']['aligned']:
                        waiting_stocks.append({
                            'Symbol': symbol,
                            'Status': 'MTF Not Aligned',
                            'Reason': 'Waiting for timeframe alignment'
                        })
                    else:
                        signals_found.append(signal)
                else:
                    waiting_stocks.append({
                        'Symbol': symbol,
                        'Status': 'Poor R/R',
                        'Reason': f'R/R {rr:.1f} < {min_risk_reward}'
                    })
            
            # Check if in waiting state (e.g., in pullback zone)
            elif signal.get('state') in ['WAIT', 'SQUEEZE']:
                waiting_stocks.append({
                    'Symbol': symbol,
                    'Status': signal.get('state'),
                    'Reason': signal.get('reason', ''),
                    'Watch For': signal.get('watch_for', '')
                })
            
            else:
                no_signal_stocks.append({
                    'Symbol': symbol,
                    'Reason': signal.get('reason', 'No signal detected')
                })
        
        except Exception as e:
            no_signal_stocks.append({
                'Symbol': symbol,
                'Reason': f'Error: {str(e)}'
            })
        
        progress.progress((i + 1) / len(watch_symbols))
    
    status_text.empty()
    progress.empty()
    
    # ========== DISPLAY RESULTS ==========
    
    # BUY SIGNALS
    if signals_found:
        st.subheader(f"🚀 BUY SIGNALS ({len(signals_found)} stocks)")
        
        for sig in signals_found:
            with st.expander(f"✅ {sig['symbol']} - {sig['type']}", expanded=True):
                
                # Main metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Entry Price", f"₹{sig['entry_price']}")
                
                with col2:
                    st.metric("Stop Loss", f"₹{sig['stop_loss']}")
                    risk_pct = (sig['entry_price'] - sig['stop_loss']) / sig['entry_price'] * 100
                    st.caption(f"Risk: {risk_pct:.2f}%")
                
                with col3:
                    st.metric("Target", f"₹{sig['target']}")
                    reward_pct = (sig['target'] - sig['entry_price']) / sig['entry_price'] * 100
                    st.caption(f"Reward: {reward_pct:.2f}%")
                
                with col4:
                    st.metric("Risk/Reward", f"1:{sig['risk_reward']}")
                    st.caption(f"Confidence: {sig['confidence']}")
                
                # Reason
                st.info(f"**Entry Reason:** {sig['reason']}")
                
                # Multi-timeframe status
                mtf = sig['multi_timeframe']
                st.markdown("**Multi-Timeframe Analysis:**")
                
                mtf_df = pd.DataFrame({
                    'Timeframe': ['5min', '15min', '60min', 'Daily'],
                    'Trend': [mtf['trends']['5min'], mtf['trends']['15min'], 
                             mtf['trends']['60min'], mtf['trends']['daily']]
                })
                st.dataframe(mtf_df, use_container_width=True)
                
                alignment_color = "green" if mtf['aligned'] else "orange"
                st.markdown(f"**Alignment:** :{alignment_color}[{mtf['alignment']}]")
                
                # Trade execution plan
                st.markdown("---")
                st.markdown("### 📋 Execution Plan")
                st.code(f"""
SYMBOL: {sig['symbol']}
ENTRY: Market/Limit order at ₹{sig['entry_price']} (or better)
STOP LOSS: ₹{sig['stop_loss']} (risk {risk_pct:.2f}%)
TARGET: ₹{sig['target']} (reward {reward_pct:.2f}%)
POSITION SIZE: Calculate based on 1-2% account risk
TIME: {sig['timestamp']} (15min timeframe)
VALIDITY: Intraday / Cancel if not filled by day end
                """, language="text")
                
                # Warning if MTF not aligned but signal exists
                if not mtf['aligned']:
                    st.warning("⚠️ Timeframes not fully aligned - use smaller position size")
    
    else:
        st.info("No BUY signals found. Check waiting list below.")
    
    # WAITING LIST
    if waiting_stocks:
        st.subheader(f"⏳ Waiting for Entry ({len(waiting_stocks)} stocks)")
        
        wait_df = pd.DataFrame(waiting_stocks)
        st.dataframe(wait_df, use_container_width=True)
        
        st.info("💡 These stocks are setting up but haven't triggered yet. Monitor closely.")
    
    # NO SIGNALS
    if no_signal_stocks:
        with st.expander(f"❌ No Signals ({len(no_signal_stocks)} stocks)"):
            no_sig_df = pd.DataFrame(no_signal_stocks)
            st.dataframe(no_sig_df, use_container_width=True)

# ========== SECTION 4: Manual Stock Analysis ==========
st.header("4️⃣ Detailed Analysis (Single Stock)")

st.markdown("Analyze a specific stock in detail across all timeframes.")

analyze_symbol = st.selectbox("Select Stock to Analyze", watch_symbols)

if st.button("📊 Analyze Stock", key="analyze"):
    
    try:
        # Load all timeframes
        df_5m = pd.read_parquet(DERIVED_ROOT / analyze_symbol / "5minute" / f"merged_{analyze_symbol}_5minute.parquet").tail(200)
        df_15m = pd.read_parquet(DERIVED_ROOT / analyze_symbol / "15minute" / f"merged_{analyze_symbol}_15minute.parquet").tail(200)
        df_60m = pd.read_parquet(DERIVED_ROOT / analyze_symbol / "60minute" / f"merged_{analyze_symbol}_60minute.parquet").tail(200)
        df_daily = pd.read_parquet(DERIVED_ROOT / analyze_symbol / "1day" / f"merged_{analyze_symbol}_1day.parquet").tail(200)
        
        for df in [df_5m, df_15m, df_60m, df_daily]:
            df.index = pd.to_datetime(df.index)
        
        # Multi-timeframe analysis
        st.subheader("Multi-Timeframe Trend")
        mtf = calculate_multi_timeframe_trend(df_5m, df_15m, df_60m, df_daily)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("5-minute", mtf['trends']['5min'])
        col2.metric("15-minute", mtf['trends']['15min'])
        col3.metric("60-minute", mtf['trends']['60min'])
        col4.metric("Daily", mtf['trends']['daily'])
        
        st.metric("Overall Alignment", mtf['alignment'])
        
        # Test all strategies
        st.subheader("All Entry Strategies")
        
        strategies = ['pullback', 'extreme', 'momentum', 'squeeze']
        strategy_results = []
        
        for strat in strategies:
            sig = generate_entry_signal(
                symbol=analyze_symbol,
                regime=default_regime,
                df_5m=df_5m,
                df_15m=df_15m,
                df_60m=df_60m,
                df_daily=df_daily,
                strategy=strat
            )
            
            strategy_results.append({
                'Strategy': strat.capitalize(),
                'Signal': '✅' if sig.get('signal', False) else '❌',
                'Type': sig.get('type', 'N/A'),
                'Entry': sig.get('entry_price', 'N/A'),
                'R/R': sig.get('risk_reward', 'N/A'),
                'Reason': sig.get('reason', 'No signal')
            })
        
        result_df = pd.DataFrame(strategy_results)
        st.dataframe(result_df, use_container_width=True)
        
        # Current price info
        st.subheader("Current Price Action (15min)")
        
        latest = df_15m.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Close", f"₹{latest['Close']:.2f}")
        col2.metric("High", f"₹{latest['High']:.2f}")
        col3.metric("Low", f"₹{latest['Low']:.2f}")
        col4.metric("Volume", f"{latest['Volume']:,.0f}")
        
    except Exception as e:
        st.error(f"Error analyzing {analyze_symbol}: {e}")

# ========== SECTION 5: Risk Management Calculator ==========
st.header("5️⃣ Position Size Calculator")

st.markdown("Calculate position size based on your account size and risk tolerance.")

col1, col2, col3 = st.columns(3)

with col1:
    account_size = st.number_input("Account Size (₹)", 10000, 10000000, 100000, 10000)

with col2:
    risk_per_trade = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.5)

with col3:
    entry_price_calc = st.number_input("Entry Price (₹)", 0.0, 100000.0, 1000.0, 10.0)

stop_loss_calc = st.number_input("Stop Loss (₹)", 0.0, 100000.0, 950.0, 10.0)

if entry_price_calc > stop_loss_calc:
    risk_per_share = entry_price_calc - stop_loss_calc
    max_loss = account_size * (risk_per_trade / 100)
    quantity = int(max_loss / risk_per_share)
    total_investment = quantity * entry_price_calc
    
    st.subheader("Position Sizing")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Quantity to Buy", f"{quantity} shares")
    col2.metric("Total Investment", f"₹{total_investment:,.0f}")
    col3.metric("Max Loss (if SL hit)", f"₹{max_loss:,.0f}")
    
    st.info(f"💡 Risk per share: ₹{risk_per_share:.2f} | Account risk: {risk_per_trade}% = ₹{max_loss:,.0f}")
else:
    st.warning("Entry price must be greater than stop loss")

# Footer
st.markdown("---")
st.caption("""
**Pro Tips:**
1. Only enter when MTF aligned (all timeframes agree)
2. Wait for pullback in trending regimes (don't chase)
3. Use 1:2 minimum risk/reward ratio
4. Never risk more than 1-2% of account per trade
5. If no clear signal, wait - patience is a position
""")