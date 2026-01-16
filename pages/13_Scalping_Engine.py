"""
₹500 Option Scalping Engine - Main Control Panel
Page 13 in the trading bot application

Features:
- Real-time stock scanning (60-second cycles)
- Binary filter system (PASS/FAIL)
- "₹500 PROBABLE" signal generation
- Active position monitoring
- Global guardrails enforcement
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time
import sys
from pathlib import Path

# Add root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.guardrails import GlobalGuardrails
from core.scanner import MultiStockScanner
from core.signal_generator import SignalGenerator
from core.indicators import compute_supertrend
from core.config import get_access_token

# Page configuration
st.set_page_config(
    page_title="₹500 Scalping Engine",
    page_icon="🚀",
    layout="wide"
)

# Initialize session state
if 'scanner' not in st.session_state:
    st.session_state.scanner = None
if 'signals' not in st.session_state:
    st.session_state.signals = []
if 'active_positions' not in st.session_state:
    st.session_state.active_positions = []
if 'scanner_running' not in st.session_state:
    st.session_state.scanner_running = False
if 'last_scan_time' not in st.session_state:
    st.session_state.last_scan_time = None

# Title and header
st.title("🚀 ₹500 Option Scalping Engine")
st.markdown("**Profit-First | Retail-Executable | Discipline-Enforced**")
st.divider()

# Initialize components
guardrails = GlobalGuardrails()
signal_gen = SignalGenerator()

# =============================================================================
# SECTION 1: GLOBAL GUARDRAILS STATUS
# =============================================================================

st.header("🛡️ Global Guardrails")

col1, col2, col3 = st.columns(3)

# Check if trading is allowed
trade_status = guardrails.can_trade_now()

with col1:
    if trade_status['allowed']:
        st.success("✅ ALL SYSTEMS GO")
    else:
        st.error(f"❌ {trade_status['reason']}")

with col2:
    stats = trade_status['stats']
    st.metric(
        "Today's Trades",
        f"{stats['trades_count']}/{guardrails.MAX_TRADES_PER_DAY}",
        delta=f"{stats['wins']}W-{stats['losses']}L"
    )

with col3:
    st.metric(
        "Daily P&L",
        f"₹{stats['daily_pnl']:.0f}",
        delta=stats['consecutive_losses'] if stats['consecutive_losses'] > 0 else None,
        delta_color="inverse"
    )

# Show lockout reason if locked
if stats['locked_out']:
    st.error(f"🔒 SYSTEM LOCKED: {stats['lockout_reason']}")
    st.stop()

st.divider()

# =============================================================================
# SECTION 2: SCANNER CONTROLS
# =============================================================================

st.header("📊 Multi-Stock Scanner")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    if not trade_status['allowed']:
        st.warning("Scanner disabled - trading window not active")
        scanner_enabled = False
    else:
        scanner_enabled = st.checkbox(
            "Enable Scanner (60-second auto-refresh)",
            value=st.session_state.scanner_running
        )
        st.session_state.scanner_running = scanner_enabled

with col2:
    manual_scan = st.button("🔍 Manual Scan", disabled=not trade_status['allowed'])

with col3:
    if st.session_state.last_scan_time:
        st.info(f"Last scan: {st.session_state.last_scan_time.strftime('%H:%M:%S')}")
    else:
        st.info("No scans yet")

# =============================================================================
# SECTION 3: LIVE SIGNALS
# =============================================================================

st.divider()
st.header("🎯 Live Signals")

# Mock function to get market data (replace with actual implementation)
def get_market_data():
    """
    TODO: Replace with actual market data fetching
    This should call Upstox API to get:
    - Index data (Nifty/BankNifty 1-minute)
    - Stock data for all 20 symbols
    - Option chain data
    """
    return {
        'index': {
            'name': 'NIFTY',
            'df': pd.DataFrame({
                'Open': [22000] * 30,
                'High': [22050] * 30,
                'Low': [21950] * 30,
                'Close': [22020] * 30,
                'ATR': [50] * 30
            })
        },
        'stocks': {}
    }

# Run scan if enabled or manual button clicked
if scanner_enabled or manual_scan:
    
    with st.spinner("Scanning 20 stocks across 5 filters..."):
        
        # Initialize scanner if not exists
        if st.session_state.scanner is None:
            st.session_state.scanner = MultiStockScanner()
        
        # Get market data
        market_data = get_market_data()
        
        # Run scan
        signals = st.session_state.scanner.scan_all_stocks_parallel(market_data)
        
        # Update state
        st.session_state.signals = signals
        st.session_state.last_scan_time = datetime.now()
        
        # Show scan summary
        st.success(f"✅ Scan complete: {len(signals)} signals generated")

# Display signals
if len(st.session_state.signals) > 0:
    
    for signal in st.session_state.signals:
        
        # Create signal card
        with st.expander(f"🚀 {signal['symbol']} - {signal['strike']} {signal['option_type']} | Expected: ₹{signal['total_expected_profit']:,}", expanded=True):
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Signal details
                st.markdown(signal_gen.format_signal_for_display(signal))
            
            with col2:
                # Action buttons
                st.markdown("### Actions")
                
                if st.button("✅ EXECUTE", key=f"execute_{signal['signal_id']}", type="primary"):
                    # TODO: Implement trade execution
                    st.success("Trade logged - ready to execute")
                    # Move to active positions
                    st.session_state.active_positions.append(signal)
                    # Remove from signals
                    st.session_state.signals.remove(signal)
                    st.rerun()
                
                if st.button("⏭️ SKIP", key=f"skip_{signal['signal_id']}"):
                    st.info("Signal skipped")
                    st.session_state.signals.remove(signal)
                    st.rerun()
                
                st.markdown("---")
                st.markdown(f"**Filters Passed:**")
                st.markdown("✅ ✅ ✅ ✅ ✅")

else:
    st.info("🔍 No signals detected. Scanner will continue monitoring...")

st.divider()

# =============================================================================
# SECTION 4: ACTIVE POSITIONS
# =============================================================================

st.header("📈 Active Positions")

if len(st.session_state.active_positions) > 0:
    
    for pos in st.session_state.active_positions:
        
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            st.markdown(f"**{pos['symbol']} {pos['strike']} {pos['option_type']}**")
            st.caption(f"Entry: ₹{pos['entry_premium']}")
        
        with col2:
            # TODO: Get live LTP and calculate P&L
            current_ltp = pos['entry_premium'] * 1.03  # Mock +3%
            pnl_pct = ((current_ltp - pos['entry_premium']) / pos['entry_premium']) * 100
            pnl_rs = (current_ltp - pos['entry_premium']) * pos['lot_size']
            
            st.metric("Current", f"₹{current_ltp:.2f}", f"{pnl_pct:+.1f}%")
        
        with col3:
            st.metric("P&L", f"₹{pnl_rs:+,.0f}")
        
        with col4:
            # Exit monitor status
            if pnl_pct >= pos['target2_pct']:
                st.success("🎯 Target 2 Hit!")
            elif pnl_pct >= pos['target1_pct']:
                st.info("🎯 Target 1 Hit")
            elif pnl_pct <= -pos['sl_pct']:
                st.error("🛑 SL Hit!")
            else:
                st.caption("Monitoring...")
        
        st.markdown("---")

else:
    st.info("No active positions")

st.divider()

# =============================================================================
# SECTION 5: SCANNER DIAGNOSTICS
# =============================================================================

with st.expander("🔍 Scanner Diagnostics"):
    
    if st.session_state.scanner:
        
        stats = st.session_state.scanner.get_stats()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Universe Size", stats['universe_size'])
        
        with col2:
            st.metric("Total Scans", stats['total_scans'])
        
        with col3:
            st.metric("Signals Generated", stats['signals_generated'])
        
        # Filter stats
        st.markdown("### Filter Performance")
        
        filter_stats = stats.get('filter_stats', {})
        
        for filter_num, filter_data in filter_stats.items():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**{filter_num}. {filter_data['name']}**")
            
            with col2:
                pass_rate = filter_data.get('pass_rate', 0)
                st.metric("Pass Rate", f"{pass_rate:.1f}%")
            
            with col3:
                avg_time = filter_data.get('avg_time_ms', 0)
                st.metric("Avg Time", f"{avg_time:.1f}ms")

# =============================================================================
# AUTO-REFRESH FOR SCANNER
# =============================================================================

if scanner_enabled:
    st.markdown("---")
    st.info("🔄 Scanner will auto-refresh in 60 seconds...")
    st_autorefresh = st.empty()
    
    import time
    time.sleep(60)  # Wait 60 seconds
    st.rerun()  # Trigger refresh

# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("₹500 Scalping Engine v1.0 | Powered by Binary Filters")