# pages/14_Option_Analyzer.py
"""
Option Chain Analyzer & Strike Selector

Triggered AFTER a stock signal fires from Live Entry Monitor
Analyzes option chain and recommends best strike with TP/SL

Usage:
1. Enter symbol (e.g., TCS) from signal
2. Select direction (CE for Long, PE for Short)
3. Click "Analyze Options"
4. Get best strike recommendation with TP/SL
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from datetime import datetime, timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from core.option_analyzer import OptionAnalyzer, generate_option_signal
    from core.config import get_access_token
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Make sure option_analyzer.py is in core/ folder")
    st.stop()

st.set_page_config(layout="wide", page_title="Option Analyzer")
st.title("📊 Option Chain Analyzer & Strike Selector")

st.markdown("""
**Use this AFTER a stock signal fires from Live Entry Monitor**

**Process:**
1. Stock signal fires (e.g., TCS Buy at ₹3,300)
2. Enter symbol and direction here
3. System analyzes option chain
4. Recommends best strike with Greeks
5. Calculates conservative TP/SL
6. Shows complete trade plan
""")

# ========== SECTION 1: Input Signal Details ==========
st.header("1️⃣ Signal Details")

col1, col2, col3 = st.columns(3)

with col1:
    symbol = st.text_input("Symbol", value="TCS", help="Stock that triggered signal")

with col2:
    spot_price = st.number_input("Spot Price", 100.0, 100000.0, 3300.0, 10.0, 
                                  help="Current market price")

with col3:
    direction = st.selectbox("Direction", ['LONG (Buy CE)', 'SHORT (Buy PE)'],
                             help="Signal direction")

signal_direction = 'LONG' if 'LONG' in direction else 'SHORT'
option_type = 'CE' if signal_direction == 'LONG' else 'PE'

# Confidence level
col1, col2, col3 = st.columns(3)

with col1:
    confidence = st.selectbox("Signal Confidence", ['HIGH', 'MEDIUM', 'LOW'],
                              help="From your regime/signal analysis")

with col2:
    strategy = st.selectbox("TP/SL Strategy", ['CONSERVATIVE', 'BALANCED', 'AGGRESSIVE'])

with col3:
    # Expiry selection
    expiry_option = st.selectbox("Expiry", ['Auto (Nearest)', 'Select from Available'],
                                 help="Select expiry date")

# Show available expiries if user wants to select
selected_expiry = None
if expiry_option == 'Select from Available':
    # Determine segment
    if symbol.upper() in ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']:
        temp_segment = 'NSE_FO'  # Expiries are in NSE_FO even for indices
    else:
        temp_segment = 'NSE_FO'
    
    # Get available expiries
    try:
        from core.option_analyzer import get_expiries_for_underlying
        available_expiries = get_expiries_for_underlying(symbol, temp_segment)
        
        if available_expiries:
            st.info(f"📅 Found {len(available_expiries)} available expiries")
            selected_expiry = st.selectbox("Select Expiry Date", available_expiries,
                                          help="Available expiry dates from instruments")
        else:
            st.warning("No expiries found. Using auto mode.")
            selected_expiry = None
    except Exception as e:
        st.error(f"Could not load expiries: {e}")
        selected_expiry = None

# ========== SECTION 2: Fetch Option Chain ==========
st.header("2️⃣ Option Chain Analysis")

# Lot size lookup
@st.cache_data
def load_lot_sizes():
    """Load F&O lot sizes from CSV"""
    lot_file = Path(ROOT) / "data" / "fno_lot_sizes.csv"
    
    if lot_file.exists():
        try:
            df = pd.read_csv(lot_file)
            return dict(zip(df['symbol'], df['lot_size']))
        except Exception as e:
            st.warning(f"⚠️ Could not load lot sizes CSV: {e}")
            st.info("Using fallback lot sizes")
            # Return fallback
            return {
                'TCS': 175,
                'INFY': 300,
                'RELIANCE': 250,
                'HDFCBANK': 550,
                'ICICIBANK': 1375,
                'SBIN': 1500,
                'WIPRO': 1200,
                'KOTAKBANK': 400,
                'AXISBANK': 650,
                'ITC': 1600,
            }
    else:
        st.info(f"💡 Lot sizes CSV not found at: {lot_file}")
        # Return fallback
        return {
            'TCS': 175,
            'INFY': 300,
            'RELIANCE': 250,
            'HDFCBANK': 550,
            'ICICIBANK': 1375,
            'SBIN': 1500,
            'WIPRO': 1200,
            'KOTAKBANK': 400,
            'AXISBANK': 650,
            'ITC': 1600,
        }

LOT_SIZES = load_lot_sizes()

lot_size = LOT_SIZES.get(symbol.upper(), 100)  # Default 100 if not found

if symbol.upper() not in LOT_SIZES:
    st.warning(f"⚠️ Lot size not found for {symbol}. Using default: 100")
    st.info("💡 Add to data/fno_lot_sizes.csv or use custom lot size below")

st.info(f"📦 Lot Size for {symbol}: {lot_size}")

# Manual lot size override
custom_lot = st.checkbox("Override Lot Size")
if custom_lot:
    lot_size = st.number_input("Custom Lot Size", 1, 10000, lot_size, 1)

if st.button("🔍 Analyze Options", type="primary"):
    
    with st.spinner("Fetching option chain and analyzing strikes..."):
        
        try:
            # Get access token
            access_token = get_access_token()
            
            if not access_token:
                st.error("❌ Access token not found. Go to Page 1 to login.")
                st.stop()
            
            # Determine segment (Index vs Stock)
            if symbol.upper() in ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']:
                segment = 'NSE_INDEX'
            else:
                segment = 'NSE_FO'
            
            # Initialize analyzer
            analyzer = OptionAnalyzer(access_token)
            
            # Resolve expiry date FIRST (before calling API)
            if selected_expiry:
                # User selected specific expiry
                expiry_to_use = selected_expiry
                st.info(f"🔍 Using selected expiry: {expiry_to_use}")
            else:
                # Auto mode - resolve nearest expiry
                st.info(f"🔍 Fetching nearest expiry for {symbol}...")
                
                from core.option_analyzer import get_expiries_for_underlying
                available_expiries = get_expiries_for_underlying(symbol, segment)
                
                if not available_expiries:
                    st.error(f"❌ No expiries found for {symbol}")
                    st.info("💡 This symbol may not have options available")
                    st.stop()
                
                expiry_to_use = available_expiries[0]  # First (nearest) expiry
                st.success(f"✅ Resolved nearest expiry: **{expiry_to_use}**")
                
                # Show other available expiries
                if len(available_expiries) > 1:
                    st.caption(f"Other expiries available: {', '.join(available_expiries[1:4])}" + 
                              (f" ... and {len(available_expiries)-4} more" if len(available_expiries) > 4 else ""))
            
            # Now fetch with the resolved expiry date
            st.info(f"📡 Calling Upstox API with expiry: {expiry_to_use}")
            
            option_chain = analyzer.fetch_option_chain(
                symbol=symbol,
                expiry=expiry_to_use,  # Now this is always a real date
                segment=segment
            )
            
            if option_chain.empty:
                st.warning("⚠️ Could not fetch option chain from Upstox API")
                
                # Show debug info
                with st.expander("🔍 Debug Information", expanded=True):
                    st.write(f"**Symbol:** {symbol}")
                    st.write(f"**Segment:** {segment}")
                    st.write(f"**Expiry Used:** {expiry_to_use}")  # Show actual date used
                    
                    # Try to show what instrument key was used
                    from core.option_analyzer import get_instrument_key, get_expiries_for_underlying
                    inst_key = get_instrument_key(symbol, segment)
                    st.write(f"**Instrument Key:** {inst_key}")
                    
                    st.markdown(f"**API Call Made:**")
                    st.code(f"""
GET /v2/option/chain
Params:
  instrument_key: {inst_key}
  expiry_date: {expiry_to_use}
                    """)
                    
                    # Show available expiries
                    try:
                        avail_exp = get_expiries_for_underlying(symbol, segment)
                        if avail_exp:
                            st.write(f"**Available Expiries ({len(avail_exp)}):**")
                            for i, exp in enumerate(avail_exp[:5]):  # Show first 5
                                marker = " ← USED" if exp == expiry_to_use else ""
                                st.write(f"  {i+1}. {exp}{marker}")
                            if len(avail_exp) > 5:
                                st.write(f"  ... and {len(avail_exp)-5} more")
                        else:
                            st.write("**Available Expiries:** None found")
                    except Exception as e:
                        st.write(f"**Available Expiries:** Error fetching ({e})")
                    
                    st.markdown("""
                    **Possible reasons:**
                    1. **No contracts for this expiry** - Market may not have opened contracts yet
                    2. **Market closed** - After hours, some expiries may not have data
                    3. **Symbol not in F&O** - Only F&O stocks have options
                    4. **API rate limit** - Too many requests, wait a moment
                    5. **Expiry too far** - Try nearest expiry (first in list)
                    
                    **Try:**
                    - Wait 30 seconds and try again (rate limit)
                    - Use 'Select from Available' and pick first expiry
                    - Check if TCS options are actually trading today
                    - Verify market hours (9:15 AM - 3:30 PM)
                    """)
                
                st.info("💡 Using DEMO mode with sample data for testing")
                
                # Create sample option chain for demonstration
                strikes = np.arange(spot_price - 200, spot_price + 200, 50)
                
                chain_data = []
                for strike in strikes:
                    # Sample CE data
                    moneyness = (spot_price - strike) / spot_price
                    delta_ce = max(0.1, min(0.9, 0.5 + moneyness))
                    premium_ce = max(5, abs(spot_price - strike) * 0.5 + np.random.uniform(10, 50))
                    
                    chain_data.append({
                        'strike': strike,
                        'option_type': 'CE',
                        'premium': round(premium_ce, 2),
                        'delta': round(delta_ce, 2),
                        'gamma': round(np.random.uniform(0.001, 0.01), 4),
                        'theta': round(-np.random.uniform(0.5, 3), 2),
                        'vega': round(np.random.uniform(5, 20), 2),
                        'iv': round(np.random.uniform(0.12, 0.25), 2),
                        'oi': int(np.random.uniform(1000, 10000)),
                        'volume': int(np.random.uniform(100, 5000))
                    })
                    
                    # Sample PE data
                    delta_pe = max(-0.9, min(-0.1, -0.5 + moneyness))
                    premium_pe = max(5, abs(spot_price - strike) * 0.5 + np.random.uniform(10, 50))
                    
                    chain_data.append({
                        'strike': strike,
                        'option_type': 'PE',
                        'premium': round(premium_pe, 2),
                        'delta': round(delta_pe, 2),
                        'gamma': round(np.random.uniform(0.001, 0.01), 4),
                        'theta': round(-np.random.uniform(0.5, 3), 2),
                        'vega': round(np.random.uniform(5, 20), 2),
                        'iv': round(np.random.uniform(0.12, 0.25), 2),
                        'oi': int(np.random.uniform(1000, 10000)),
                        'volume': int(np.random.uniform(100, 5000))
                    })
                
                option_chain = pd.DataFrame(chain_data)
                
                st.warning("🧪 DEMO MODE: Using sample Greeks data")
            
            else:
                # SUCCESS! Real data fetched
                st.success(f"✅ Successfully fetched {len(option_chain)} option contracts from Upstox!")
                
                # Show a sample of the data
                with st.expander("📋 Option Chain Preview (Click to expand)"):
                    st.write(f"Total contracts: {len(option_chain)}")
                    st.write(f"CE contracts: {len(option_chain[option_chain['option_type']=='CE'])}")
                    st.write(f"PE contracts: {len(option_chain[option_chain['option_type']=='PE'])}")
                    st.dataframe(option_chain.head(10), use_container_width=True)
            
            # Display full option chain
            with st.expander("📋 Full Option Chain Data"):
                st.dataframe(option_chain, use_container_width=True)
            
            # Generate option signal
            st.info("Analyzing strikes and selecting best option...")
            
            signal = generate_option_signal(
                symbol=symbol,
                spot_price=spot_price,
                signal_direction=signal_direction,
                option_chain=option_chain,
                lot_size=lot_size,
                confidence=confidence,
                account_capital=100000,  # Default, can be made configurable
                strategy=strategy
            )
            
            if 'error' in signal:
                st.error(f"❌ {signal['error']}")
                st.stop()
            
            # ========== SECTION 3: Display Results ==========
            st.success("✅ Analysis Complete!")
            
            st.markdown("---")
            st.header("3️⃣ Recommended Trade")
            
            # Main recommendation box
            st.markdown(f"""
            <div style='background-color: #1e3a5f; padding: 20px; border-radius: 10px; border: 2px solid #4a90e2;'>
                <h2 style='color: #4a90e2; margin-top: 0;'>🎯 {symbol} {signal['option_type']} Option</h2>
                <h3 style='color: white;'>Strike: {signal['strike']} | Premium: ₹{signal['entry_premium']}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Entry Premium", f"₹{signal['entry_premium']}")
                st.caption(f"Delta: {signal['delta']:.2f}")
            
            with col2:
                st.metric("Take Profit", f"₹{signal['tp_premium']}", 
                         delta=f"+{signal['tp_profit']:.0f}")
                st.caption(f"TP: {signal['tp_premium']}")
            
            with col3:
                st.metric("Stop Loss", f"₹{signal['sl_premium']}", 
                         delta=f"-{signal['sl_loss']:.0f}", delta_color="inverse")
                st.caption(f"SL: {signal['sl_premium']}")
            
            with col4:
                st.metric("Risk/Reward", signal['risk_reward'])
                st.caption(f"Score: {signal['score']:.0f}/100")
            
            # Greeks and metrics
            st.subheader("📈 Option Greeks & Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Greeks:**")
                st.write(f"• Delta: {signal['delta']:.3f}")
                st.write(f"• IV: {signal['iv']*100:.1f}%")
                st.write(f"• OI: {signal['oi']:,}")
            
            with col2:
                st.markdown("**Position:**")
                st.write(f"• Lots: {signal['lots']}")
                st.write(f"• Quantity: {signal['quantity']}")
                st.write(f"• Investment: ₹{signal['investment']:,.2f}")
            
            with col3:
                st.markdown("**P&L Projection:**")
                st.write(f"• If TP: +₹{signal['tp_profit']:,.2f} 🟢")
                st.write(f"• If SL: -₹{signal['sl_loss']:,.2f} 🔴")
                st.write(f"• Risk: {(signal['sl_loss']/signal['investment']*100):.1f}%")
            
            # Execution plan
            st.markdown("---")
            st.subheader("📋 Execution Plan")
            
            execution_plan = f"""
**STEP 1: PLACE OPTION ORDER**
```
BUY {signal['lots']} LOT {symbol} {signal['strike']} {signal['option_type']}
Entry: ₹{signal['entry_premium']} (or better)
Total Investment: ₹{signal['investment']:,.2f}
```

**STEP 2: SET TARGET ORDER**
```
SELL @ ₹{signal['tp_premium']} (Take Profit)
Expected Profit: ₹{signal['tp_profit']:,.2f}
```

**STEP 3: SET STOP LOSS**
```
SELL @ ₹{signal['sl_premium']} (Stop Loss)
Max Loss: ₹{signal['sl_loss']:,.2f}
```

**STEP 4: TIME STOP**
```
Exit by: 3:15 PM (if intraday)
OR
Exit by: 2 days / Thursday (if swing)
```

**RISK MANAGEMENT:**
- Account Risk: {(signal['sl_loss']/100000)*100:.2f}% (assuming ₹1L capital)
- Max Concurrent: 3 positions
- Daily Loss Limit: 3% (₹3,000)
"""
            
            st.markdown(execution_plan)
            
            # Download trade plan
            st.download_button(
                label="📥 Download Trade Plan",
                data=execution_plan,
                file_name=f"{symbol}_{signal['strike']}{signal['option_type']}_trade_plan.txt",
                mime="text/plain"
            )
            
            # ========== SECTION 4: Alternative Strikes ==========
            st.markdown("---")
            st.subheader("🔄 Alternative Strike Options")
            
            # Filter and score other strikes
            filtered_chain = option_chain[
                (option_chain['option_type'] == signal['option_type']) &
                (option_chain['premium'] > 0)
            ].copy()
            
            # Calculate simple score for each
            filtered_chain['simple_score'] = filtered_chain.apply(
                lambda row: (
                    (40 if 0.45 <= abs(row['delta']) <= 0.65 else 20) +
                    (30 if row['iv'] < 0.20 else 10) +
                    (20 if row['oi'] > 1000 else 5)
                ),
                axis=1
            )
            
            # Sort by score and show top 5
            top_alternatives = filtered_chain.nlargest(5, 'simple_score')[
                ['strike', 'premium', 'delta', 'iv', 'oi', 'simple_score']
            ]
            
            # Highlight selected strike
            def highlight_selected(row):
                if row['strike'] == signal['strike']:
                    return ['background-color: #1e3a5f; color: white'] * len(row)
                return [''] * len(row)
            
            styled_df = top_alternatives.style.apply(highlight_selected, axis=1)
            st.dataframe(styled_df, use_container_width=True)
            
            st.caption(f"✅ Selected strike (highlighted): {signal['strike']} {signal['option_type']}")
        
        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            st.code(traceback.format_exc())

# ========== SECTION 5: Position Sizing Calculator ==========
st.markdown("---")
st.header("4️⃣ Position Sizing Calculator")

st.markdown("Calculate optimal position size based on your account and risk tolerance.")

col1, col2 = st.columns(2)

with col1:
    account_capital = st.number_input("Account Capital (₹)", 10000, 10000000, 100000, 5000)

with col2:
    risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.5)

if 'signal' in locals() and 'error' not in signal:
    # Recalculate position size with custom parameters
    from core.option_analyzer import OptionAnalyzer
    
    analyzer = OptionAnalyzer("")
    
    position = analyzer.calculate_position_size(
        entry_premium=signal['entry_premium'],
        lot_size=lot_size,
        account_capital=account_capital,
        risk_pct=risk_pct/100,
        sl_pct=0.30  # 30% SL
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Recommended Lots", position['lots'])
        st.caption(f"Total Quantity: {position['quantity']}")
    
    with col2:
        st.metric("Investment Required", f"₹{position['investment']:,.0f}")
        st.caption(f"{position['deployment_pct']:.1f}% of capital")
    
    with col3:
        st.metric("Maximum Risk", f"₹{position['max_risk']:,.0f}")
        st.caption(f"{position['risk_pct_of_capital']:.2f}% of capital")
    
    # Warning if over-leverage
    if position['deployment_pct'] > 30:
        st.warning("⚠️ Investment exceeds 30% of capital. Consider reducing position size.")
    
    if position['risk_pct_of_capital'] > 2:
        st.error("❌ Risk exceeds 2% of capital. REDUCE position size!")

# Footer tips
st.markdown("---")
st.caption("""
**💡 Pro Tips:**
- Always verify Greeks before entry (Delta, IV)
- Never risk more than 1-2% per trade
- Exit by 3:15 PM for intraday
- Never hold through expiry week (exit Thursday)
- Trail SL to breakeven at +15% profit
""")