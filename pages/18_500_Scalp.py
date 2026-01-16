import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, time
import json
import time as ttime
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="₹500 Scalp Playbook",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #00C853, #FFD600);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .score-box {
        background: rgba(0, 200, 83, 0.1);
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        border: 2px solid #00C853;
        margin: 1rem 0;
    }
    
    .red-box {
        background: rgba(255, 61, 0, 0.1);
        border: 2px solid #FF3D00;
    }
    
    .checklist-item {
        background: rgba(255, 255, 255, 0.03);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        border-left: 4px solid transparent;
    }
    
    .checklist-item.checked {
        background: rgba(0, 200, 83, 0.1);
        border-left-color: #00C853;
    }
    
    .stock-card {
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .stock-card.selected {
        border-color: #00C853;
        background: rgba(0, 200, 83, 0.05);
    }
    
    .stock-card:hover {
        border-color: #FFD600;
    }
    
    .trade-log-entry {
        padding: 0.5rem;
        border-bottom: 1px solid #2d3748;
        font-family: monospace;
    }
    
    .trade-log-entry.win {
        color: #00C853;
    }
    
    .trade-log-entry.loss {
        color: #FF3D00;
    }
    
    .metric-box {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'scalp_state' not in st.session_state:
    st.session_state.scalp_state = {
        'selected_stock': 'PNB',
        'score': 0,
        'trades_today': 0,
        'daily_pnl': 0,
        'consecutive_losses': 0,
        'trade_log': [],
        'checklist_state': {
            'break_reject': False,
            'candle_body': False,
            'volume_spike': False,
            'ltp_reacts': False,
            'no_resistance': False,
            'market_trend': False
        }
    }


def calculate_score():
    """Calculate probability score based on checklist"""
    score = 0
    if st.session_state.scalp_state['checklist_state']['break_reject']:
        score += 2
    if st.session_state.scalp_state['checklist_state']['candle_body']:
        score += 2
    if st.session_state.scalp_state['checklist_state']['volume_spike']:
        score += 2
    if st.session_state.scalp_state['checklist_state']['ltp_reacts']:
        score += 2
    if st.session_state.scalp_state['checklist_state']['no_resistance']:
        score += 1
    if st.session_state.scalp_state['checklist_state']['market_trend']:
        score += 1
    return score


def is_trading_window_open():
    """Check if current time is within trading window (9:20-11:30)"""
    current_time = datetime.now().time()
    start_time = time(9, 20)
    end_time = time(11, 30)
    return start_time <= current_time <= end_time


def execute_trade():
    """Execute a trade based on current setup"""
    state = st.session_state.scalp_state

    # Validate trade conditions
    if state['score'] < 7:
        st.error("❌ Score must be ≥ 7 to execute trade!")
        return False

    if state['trades_today'] >= 6:
        st.error("❌ Daily trade limit reached (6 trades)!")
        return False

    if not is_trading_window_open():
        st.error("❌ Trading window closed (9:20-11:30 only)!")
        return False

    if state['consecutive_losses'] >= 2:
        st.error("❌ Max consecutive losses reached (2)!")
        return False

    # Simulate trade outcome
    # Higher score = better win probability
    win_probability = 0.6 + (state['score'] - 7) * 0.1
    is_win = np.random.random() < win_probability

    # Calculate P&L (400-600 for wins, 100-200 for losses)
    if is_win:
        pnl = np.random.randint(400, 601)
        state['consecutive_losses'] = 0
        trade_type = "WIN"
        color_class = "win"
    else:
        pnl = -np.random.randint(100, 201)
        state['consecutive_losses'] += 1
        trade_type = "LOSS"
        color_class = "loss"

    # Update state
    state['trades_today'] += 1
    state['daily_pnl'] += pnl

    # Log trade
    log_entry = {
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'stock': state['selected_stock'],
        'type': trade_type,
        'pnl': pnl,
        'score': state['score'],
        'color': color_class
    }
    state['trade_log'].insert(0, log_entry)  # Add to beginning

    # Keep only last 20 entries
    if len(state['trade_log']) > 20:
        state['trade_log'] = state['trade_log'][:20]

    return True


def reset_day():
    """Reset daily trading stats"""
    st.session_state.scalp_state.update({
        'trades_today': 0,
        'daily_pnl': 0,
        'consecutive_losses': 0,
        'trade_log': [],
        'checklist_state': {
            'break_reject': False,
            'candle_body': False,
            'volume_spike': False,
            'ltp_reacts': False,
            'no_resistance': False,
            'market_trend': False
        },
        'score': 0
    })


# Main app
st.markdown('<div class="main-header">💰 ₹500 SCALP PLAYBOOK</div>',
            unsafe_allow_html=True)
st.caption("Mechanical Trading System | Score ≥ 7 = ENTRY | Score ≤ 6 = NO TRADE")

# Layout
col1, col2, col3 = st.columns([1, 1.5, 1])

with col1:
    st.subheader("🎯 1. Stock Selection")

    stocks = {
        'PNB': {'lot': 8000, 'desc': 'Massive leverage'},
        'SBIN': {'lot': 1500, 'desc': 'Clean momentum'},
        'ICICIBANK': {'lot': 1375, 'desc': 'Algo friendly'},
        'BANKNIFTY': {'lot': 15, 'desc': 'Premium moves fast'},
        'FINNIFTY': {'lot': 40, 'desc': 'Cleaner intraday'}
    }

    selected_stock = st.session_state.scalp_state['selected_stock']

    for stock, info in stocks.items():
        is_selected = stock == selected_stock
        emoji = "✅" if is_selected else "○"

        if st.button(f"{emoji} {stock} | Lot: {info['lot']}",
                     key=f"stock_{stock}",
                     type="primary" if is_selected else "secondary",
                     use_container_width=True,
                     help=info['desc']):
            st.session_state.scalp_state['selected_stock'] = stock
            st.rerun()

    st.divider()

    st.subheader("⚙️ Option Selection Rule")
    with st.expander("CRITICAL: Buy ONLY these", expanded=True):
        st.markdown("""
        - **ATM or 1 strike ITM**
        - **Delta:** 0.45 – 0.65
        - **Bid–Ask spread:** ≤ 1 tick
        - **Premium range:** ₹0.80 - ₹25
        """)

    with st.expander("❌ NEVER buy these"):
        st.markdown("""
        - Premium < ₹0.80 (spread kills you)
        - Premium > ₹25 (slow % movement)
        - OI with no volume
        - Deep OTM "lottery" strikes
        """)

with col2:
    st.subheader("📊 2. ₹500 Probability Score")
    st.caption("Every trade must score ≥ 7 / 10")

    # Checklist with scoring
    checklist_state = st.session_state.scalp_state['checklist_state']

    col2a, col2b = st.columns(2)

    with col2a:
        checklist_state['break_reject'] = st.checkbox(
            "Underlying breaks/rejects VWAP/High/Low",
            value=checklist_state['break_reject'],
            help="+2 points"
        )

        checklist_state['candle_body'] = st.checkbox(
            "1-min candle body > 0.6× ATR(1m)",
            value=checklist_state['candle_body'],
            help="+2 points"
        )

        checklist_state['volume_spike'] = st.checkbox(
            "Volume spike (> 1.5× last 5 bars)",
            value=checklist_state['volume_spike'],
            help="+2 points"
        )

    with col2b:
        checklist_state['ltp_reacts'] = st.checkbox(
            "Option LTP reacts instantly (no lag)",
            value=checklist_state['ltp_reacts'],
            help="+2 points"
        )

        checklist_state['no_resistance'] = st.checkbox(
            "No nearby resistance/support (≤0.2%)",
            value=checklist_state['no_resistance'],
            help="+1 point"
        )

        checklist_state['market_trend'] = st.checkbox(
            "Market not choppy (trend or impulse)",
            value=checklist_state['market_trend'],
            help="+1 point"
        )

    # Calculate and display score
    score = calculate_score()
    st.session_state.scalp_state['score'] = score

    # Score display
    score_color = "#00C853" if score >= 7 else "#FF3D00"
    st.markdown(f"""
    <div class="score-box {"red-box" if score < 7 else ""}">
        <h1 style="font-size: 4rem; margin: 0; color: {score_color}">{score}</h1>
        <h3 style="margin: 0;">/ 10 POINTS</h3>
        <p style="font-size: 0.9rem; color: #94a3b8;">
            {score} out of 6 conditions checked
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Trade signal
    if score >= 7:
        st.success("✅ **ENTRY ALLOWED** - Score ≥ 7")
    else:
        st.error("❌ **NO TRADE** - Score ≤ 6")

    st.divider()

    # Trade simulator
    st.subheader("🎮 Trade Simulator")

    sim_col1, sim_col2, sim_col3 = st.columns(3)

    with sim_col1:
        if st.button("🎯 Simulate Good Setup", use_container_width=True):
            # Auto-check high probability conditions
            st.session_state.scalp_state['checklist_state'] = {
                'break_reject': True,
                'candle_body': True,
                'volume_spike': True,
                'ltp_reacts': True,
                'no_resistance': True,
                'market_trend': False
            }
            st.rerun()

    with sim_col2:
        if st.button("💀 Simulate Bad Setup", use_container_width=True):
            # Auto-check low probability conditions
            st.session_state.scalp_state['checklist_state'] = {
                'break_reject': False,
                'candle_body': True,
                'volume_spike': False,
                'ltp_reacts': False,
                'no_resistance': True,
                'market_trend': True
            }
            st.rerun()

    with sim_col3:
        if st.button("🎲 Random Setup", use_container_width=True):
            # Random conditions
            st.session_state.scalp_state['checklist_state'] = {
                key: np.random.choice([True, False])
                for key in st.session_state.scalp_state['checklist_state']
            }
            st.rerun()

with col3:
    st.subheader("🛡️ 3. Daily Risk Governor")

    state = st.session_state.scalp_state

    # Metrics
    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        trades_color = "#FF3D00" if state['trades_today'] >= 6 else "#00C853"
        st.metric("Trades Today",
                  f"{state['trades_today']}/6",
                  delta=None,
                  delta_color="off")

        loss_color = "#FF3D00" if state['daily_pnl'] <= -2000 else "#00C853"
        st.metric("Daily P&L",
                  f"₹{state['daily_pnl']}",
                  delta=None,
                  delta_color="off")

    with metric_col2:
        consec_color = "#FF3D00" if state['consecutive_losses'] >= 2 else "#00C853"
        st.metric("Consecutive Losses",
                  f"{state['consecutive_losses']}/2",
                  delta=None,
                  delta_color="off")

        window_open = is_trading_window_open()
        window_color = "#00C853" if window_open else "#FF3D00"
        window_text = "OPEN" if window_open else "CLOSED"
        st.metric("Trading Window",
                  window_text,
                  delta=None,
                  delta_color="off")

    # Execute Trade Button
    execute_disabled = (score < 7 or
                        state['trades_today'] >= 6 or
                        not window_open or
                        state['consecutive_losses'] >= 2)

    if st.button("🚀 EXECUTE TRADE",
                 type="primary",
                 disabled=execute_disabled,
                 use_container_width=True):
        if execute_trade():
            st.success(f"Trade executed on {state['selected_stock']}!")
            ttime.sleep(0.5)
            st.rerun()

    # Reset Day Button
    if st.button("🔄 Reset Day",
                 type="secondary",
                 use_container_width=True):
        reset_day()
        st.success("Day reset!")
        ttime.sleep(0.5)
        st.rerun()

    st.divider()

    # Trade Log
    st.subheader("📝 Trade Log")

    trade_log_container = st.container(height=300)

    with trade_log_container:
        if not state['trade_log']:
            st.info("No trades yet today")
        else:
            for entry in state['trade_log']:
                emoji = "✅" if entry['type'] == "WIN" else "❌"
                pnl_color = "#00C853" if entry['pnl'] > 0 else "#FF3D00"
                pnl_sign = "+" if entry['pnl'] > 0 else ""

                st.markdown(f"""
                **{emoji} {entry['timestamp']}** | {entry['stock']}  
                Score: {entry['score']} | P&L: <span style="color:{pnl_color}">{pnl_sign}₹{entry['pnl']}</span>
                """, unsafe_allow_html=True)
                st.divider()

# Bottom Section
st.divider()
st.subheader("📈 Trade Analytics")

col_anal1, col_anal2, col_anal3, col_anal4 = st.columns(4)

with col_anal1:
    if state['trade_log']:
        win_rate = len([t for t in state['trade_log']
                       if t['type'] == "WIN"]) / len(state['trade_log'])
        st.metric("Win Rate", f"{win_rate*100:.1f}%")

with col_anal2:
    if state['trade_log']:
        avg_win = np.mean([t['pnl']
                          for t in state['trade_log'] if t['pnl'] > 0])
        st.metric("Avg Win", f"₹{avg_win:.0f}")

with col_anal3:
    if state['trade_log']:
        avg_loss = np.mean([abs(t['pnl'])
                           for t in state['trade_log'] if t['pnl'] < 0])
        st.metric("Avg Loss", f"₹{avg_loss:.0f}")

with col_anal4:
    if state['trade_log']:
        expectancy = (win_rate * avg_win) - ((1-win_rate) * avg_loss)
        st.metric("Expectancy", f"₹{expectancy:.0f}")

# System Insights
st.divider()
st.subheader("💡 System Insights")

insight_col1, insight_col2 = st.columns(2)

with insight_col1:
    with st.expander("✅ Why This Works", expanded=True):
        st.markdown("""
        You are exploiting:
        - **Option repricing latency**
        - **Algo-driven micro momentum**
        - **Human hesitation at break levels**
        
        You are NOT:
        - Predicting trend
        - Holding through chop
        - Fighting theta decay
        """)

with insight_col2:
    with st.expander("❌ Common Failure Points", expanded=True):
        st.markdown("""
        - Trading every candle
        - Revenge trades
        - Deep OTM options
        - Holding losers "hoping"
        - Trading during lunch hours
        """)

# Export/Import functionality
st.sidebar.title("⚙️ Settings")

st.sidebar.subheader("Data Management")
if st.sidebar.button("💾 Export Today's Data"):
    # Create export data
    export_data = {
        'date': datetime.now().strftime("%Y-%m-%d"),
        'trades': st.session_state.scalp_state['trade_log'],
        'summary': {
            'total_trades': st.session_state.scalp_state['trades_today'],
            'daily_pnl': st.session_state.scalp_state['daily_pnl'],
            'win_rate': None
        }
    }

    # Convert to JSON
    json_str = json.dumps(export_data, indent=2)

    # Download button
    st.sidebar.download_button(
        label="📥 Download JSON",
        data=json_str,
        file_name=f"scalp_trades_{datetime.now().strftime('%Y%m%d')}.json",
        mime="application/json"
    )

st.sidebar.subheader("Quick Actions")
if st.sidebar.button("🔧 Auto-fill Good Setup"):
    st.session_state.scalp_state['checklist_state'] = {
        'break_reject': True,
        'candle_body': True,
        'volume_spike': True,
        'ltp_reacts': True,
        'no_resistance': True,
        'market_trend': True
    }
    st.rerun()

if st.sidebar.button("🗑️ Clear Checklist"):
    st.session_state.scalp_state['checklist_state'] = {
        'break_reject': False,
        'candle_body': False,
        'volume_spike': False,
        'ltp_reacts': False,
        'no_resistance': False,
        'market_trend': False
    }
    st.rerun()

# Footer
st.sidebar.divider()
st.sidebar.caption("""
**₹500 SCALP SYSTEM**  
Mechanical Execution Only  
No Discretion • No Guessing  
""")
