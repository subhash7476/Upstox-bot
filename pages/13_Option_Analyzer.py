# pages/13_Options_Trading.py
"""
🎯 OPTIONS TRADING & PAPER TRADING
===================================
Complete options chain analysis and realistic paper trading simulation

Features:
- Option chain viewer with Greeks
- Strike selection based on technical levels
- Paper trading with realistic fills
- Position tracking and P&L
- Trade journal
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import requests
import json
from pathlib import Path
import sys
import os
from scipy.stats import norm
import math

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.config import get_access_token
from core.database import TradingDB
from core.indicators import compute_supertrend

st.set_page_config(page_title="Options Trading", layout="wide", page_icon="📈")

# =====================================================================
# CONFIGURATION
# =====================================================================

PAPER_TRADES_TABLE = "paper_trades"
POSITIONS_TABLE = "paper_positions"
ALERTS_TABLE = "option_alerts"

# Risk-free rate (RBI repo rate - approximate)
RISK_FREE_RATE = 0.065  # 6.5%

# Default volatility if IV not available
DEFAULT_IV = 0.25  # 25%

# =====================================================================
# DATABASE SETUP
# =====================================================================

@st.cache_resource
def get_db():
    """Get database connection"""
    return TradingDB()

def init_tables():
    """Initialize paper trading tables"""
    db = get_db()
    
    # Paper trades table with SL/TP
    db.con.execute(f"""
        CREATE TABLE IF NOT EXISTS {PAPER_TRADES_TABLE} (
            trade_id INTEGER PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            symbol VARCHAR,
            instrument_key VARCHAR,
            option_type VARCHAR,  -- CE or PE
            strike DOUBLE,
            expiry DATE,
            action VARCHAR,  -- BUY or SELL
            quantity INTEGER,
            entry_price DOUBLE,
            stop_loss DOUBLE,  -- Auto SL price
            take_profit DOUBLE,  -- Auto TP price
            exit_price DOUBLE,
            exit_timestamp TIMESTAMP,
            exit_reason VARCHAR,  -- MANUAL, SL_HIT, TP_HIT
            status VARCHAR,  -- OPEN or CLOSED
            pnl DOUBLE,
            pnl_pct DOUBLE,
            notes TEXT
        )
    """)
    
    # Active positions table with SL/TP
    db.con.execute(f"""
        CREATE TABLE IF NOT EXISTS {POSITIONS_TABLE} (
            position_id INTEGER PRIMARY KEY,
            symbol VARCHAR,
            instrument_key VARCHAR,
            option_type VARCHAR,
            strike DOUBLE,
            expiry DATE,
            quantity INTEGER,
            entry_price DOUBLE,
            stop_loss DOUBLE,
            take_profit DOUBLE,
            entry_timestamp TIMESTAMP,
            current_price DOUBLE,
            last_updated TIMESTAMP,
            unrealized_pnl DOUBLE,
            unrealized_pnl_pct DOUBLE
        )
    """)
    
    # Alerts table
    db.con.execute(f"""
        CREATE TABLE IF NOT EXISTS {ALERTS_TABLE} (
            alert_id INTEGER PRIMARY KEY,
            position_id INTEGER,
            alert_type VARCHAR,  -- SL_HIT, TP_HIT, CUSTOM
            trigger_price DOUBLE,
            triggered_at TIMESTAMP,
            message TEXT,
            acknowledged BOOLEAN DEFAULT FALSE
        )
    """)

# =====================================================================
# GREEKS CALCULATOR (BLACK-SCHOLES)
# =====================================================================

def calculate_greeks(
    spot_price: float,
    strike_price: float,
    time_to_expiry: float,  # in years
    volatility: float,  # IV as decimal
    risk_free_rate: float,
    option_type: str  # 'CE' or 'PE'
) -> dict:
    """
    Calculate option Greeks using Black-Scholes model
    
    Returns:
        dict with Delta, Gamma, Theta, Vega, Rho
    """
    
    if time_to_expiry <= 0:
        return {
            'delta': 0, 'gamma': 0, 'theta': 0, 
            'vega': 0, 'rho': 0, 'iv': volatility
        }
    
    # d1 and d2
    d1 = (np.log(spot_price / strike_price) + 
          (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / \
         (volatility * np.sqrt(time_to_expiry))
    
    d2 = d1 - volatility * np.sqrt(time_to_expiry)
    
    # Greeks
    if option_type == 'CE':
        # Call option
        delta = norm.cdf(d1)
        theta = (-spot_price * norm.pdf(d1) * volatility / (2 * np.sqrt(time_to_expiry)) -
                 risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2))
        rho = strike_price * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
    else:
        # Put option
        delta = -norm.cdf(-d1)
        theta = (-spot_price * norm.pdf(d1) * volatility / (2 * np.sqrt(time_to_expiry)) +
                 risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2))
        rho = -strike_price * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)
    
    # Gamma and Vega (same for both call and put)
    gamma = norm.pdf(d1) / (spot_price * volatility * np.sqrt(time_to_expiry))
    vega = spot_price * norm.pdf(d1) * np.sqrt(time_to_expiry)
    
    # Convert theta to per-day (divide by 365)
    theta_per_day = theta / 365
    
    # Convert vega to per 1% change (divide by 100)
    vega_per_pct = vega / 100
    
    return {
        'delta': round(delta, 4),
        'gamma': round(gamma, 4),
        'theta': round(theta_per_day, 4),  # Per day
        'vega': round(vega_per_pct, 4),    # Per 1% IV change
        'rho': round(rho, 4),
        'iv': round(volatility, 4)
    }

def calculate_time_to_expiry(expiry_date: date) -> float:
    """Calculate time to expiry in years"""
    today = date.today()
    days_to_expiry = (expiry_date - today).days
    
    # Consider only trading days (approximately)
    trading_days = days_to_expiry * (5/7)  # Rough approximation
    
    # Convert to years
    return max(trading_days / 252, 0.001)  # Min 0.001 to avoid division by zero

# =====================================================================
# RISK CALCULATOR
# =====================================================================

def calculate_risk_metrics(
    entry_price: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    quantity: int,
    lot_size: int = 1
) -> dict:
    """
    Calculate risk metrics for a trade
    
    Returns:
        dict with max_loss, max_profit, risk_reward, breakeven, etc.
    """
    
    total_quantity = quantity * lot_size
    
    # Calculate prices
    stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
    take_profit_price = entry_price * (1 + take_profit_pct / 100)
    
    # Calculate P&L
    max_loss = (entry_price - stop_loss_price) * total_quantity
    max_profit = (take_profit_price - entry_price) * total_quantity
    
    # Risk-reward ratio
    risk_reward = max_profit / max_loss if max_loss > 0 else 0
    
    # Breakeven (for options, this is just entry price for buyer)
    breakeven_price = entry_price
    
    return {
        'stop_loss_price': round(stop_loss_price, 2),
        'take_profit_price': round(take_profit_price, 2),
        'max_loss': round(max_loss, 2),
        'max_profit': round(max_profit, 2),
        'risk_reward': round(risk_reward, 2),
        'breakeven': round(breakeven_price, 2),
        'total_quantity': total_quantity
    }

# =====================================================================
# ALERTS & AUTO SL/TP
# =====================================================================

def check_and_execute_alerts(api):
    """
    Check all open positions for SL/TP hits and execute automatically
    
    Returns:
        list of triggered alerts
    """
    db = get_db()
    triggered_alerts = []
    
    try:
        # Get all open positions
        positions = db.con.execute(f"""
            SELECT * FROM {POSITIONS_TABLE}
        """).fetchall()
        
        if not positions:
            return triggered_alerts
        
        # Get current prices
        inst_keys = [pos[2] for pos in positions]  # instrument_key column
        ltp_data = api.get_ltp(inst_keys)
        
        for pos in positions:
            position_id = pos[0]
            inst_key = pos[2]
            option_type = pos[3]
            strike = pos[4]
            entry_price = pos[7]
            stop_loss = pos[8]
            take_profit = pos[9]
            quantity = pos[6]
            
            # Get current price
            if inst_key not in ltp_data:
                continue
            
            current_price = ltp_data[inst_key].get('last_price', 0)
            
            if current_price == 0:
                continue
            
            # Check SL hit
            if stop_loss and current_price <= stop_loss:
                # Execute SL
                success, msg = close_position(
                    position_id, 
                    current_price, 
                    exit_reason='SL_HIT'
                )
                
                if success:
                    alert_msg = f"🛑 STOP LOSS HIT | {option_type} {strike} | Exit: ₹{current_price:.2f}"
                    triggered_alerts.append(alert_msg)
                    
                    # Log alert
                    db.con.execute(f"""
                        INSERT INTO {ALERTS_TABLE}
                        (position_id, alert_type, trigger_price, triggered_at, message)
                        VALUES (?, 'SL_HIT', ?, CURRENT_TIMESTAMP, ?)
                    """, [position_id, current_price, alert_msg])
            
            # Check TP hit
            elif take_profit and current_price >= take_profit:
                # Execute TP
                success, msg = close_position(
                    position_id, 
                    current_price, 
                    exit_reason='TP_HIT'
                )
                
                if success:
                    alert_msg = f"🎯 TARGET HIT | {option_type} {strike} | Exit: ₹{current_price:.2f}"
                    triggered_alerts.append(alert_msg)
                    
                    # Log alert
                    db.con.execute(f"""
                        INSERT INTO {ALERTS_TABLE}
                        (position_id, alert_type, trigger_price, triggered_at, message)
                        VALUES (?, 'TP_HIT', ?, CURRENT_TIMESTAMP, ?)
                    """, [position_id, current_price, alert_msg])
            
            # Update current price in position
            else:
                unrealized_pnl = (current_price - entry_price) * quantity
                unrealized_pnl_pct = ((current_price - entry_price) / entry_price) * 100
                
                db.con.execute(f"""
                    UPDATE {POSITIONS_TABLE}
                    SET current_price = ?,
                        last_updated = CURRENT_TIMESTAMP,
                        unrealized_pnl = ?,
                        unrealized_pnl_pct = ?
                    WHERE position_id = ?
                """, [current_price, unrealized_pnl, unrealized_pnl_pct, position_id])
        
        return triggered_alerts
        
    except Exception as e:
        st.error(f"Error checking alerts: {e}")
        return triggered_alerts

def get_recent_alerts(limit: int = 10) -> pd.DataFrame:
    """Get recent alerts"""
    db = get_db()
    
    try:
        query = f"""
            SELECT * FROM {ALERTS_TABLE}
            ORDER BY triggered_at DESC
            LIMIT {limit}
        """
        
        return db.con.execute(query).df()
        
    except:
        return pd.DataFrame()

# =====================================================================
# UPSTOX API - OPTIONS
# =====================================================================

class UpstoxOptions:
    """Upstox Options API wrapper"""
    
    BASE_URL = "https://api.upstox.com/v2"
    
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }
    
    def get_option_chain(self, symbol: str, expiry_date: str = None):
        """
        Get option chain for a symbol
        
        Args:
            symbol: Underlying symbol (e.g., NIFTY, BANKNIFTY, RELIANCE)
            expiry_date: Optional expiry date (YYYY-MM-DD)
        
        Returns:
            Dict with CE and PE data
        """
        # For now, we'll fetch from instruments and build chain
        # Real implementation would use Upstox option chain API
        pass
    
    def get_option_greeks(self, instrument_key: str):
        """Get Greeks for an option"""
        # Greeks: Delta, Gamma, Theta, Vega, IV
        # Upstox doesn't provide Greeks directly, would need Black-Scholes calculation
        pass
    
    def get_ltp(self, instrument_keys: list):
        """Get Last Traded Price for options"""
        if not instrument_keys:
            return {}
        
        keys_param = ",".join(instrument_keys)
        url = f"{self.BASE_URL}/market-quote/ltp"
        params = {"instrument_key": keys_param}
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') == 'success':
                return data.get('data', {})
            
            return {}
            
        except:
            return {}

# =====================================================================
# OPTION CHAIN BUILDER
# =====================================================================

def get_fo_expiries(symbol: str, db: TradingDB) -> list:
    """Get available F&O expiry dates for a symbol"""
    
    # Query using trading_symbol pattern match (more reliable)
    # Trading symbols are like: "RELIANCE 1280 CE 30 DEC 25"
    query = f"""
        SELECT DISTINCT expiry
        FROM instruments
        WHERE trading_symbol LIKE '{symbol} %'
          AND segment = 'NSE_FO'
          AND (instrument_type = 'CE' OR instrument_type = 'PE')
          AND expiry IS NOT NULL
        ORDER BY expiry
    """
    
    try:
        result = db.con.execute(query).fetchall()
        
        if not result:
            return []
        
        # Convert to date objects
        expiries = []
        for row in result:
            if row[0]:
                if isinstance(row[0], str):
                    # Parse string dates
                    try:
                        exp_date = pd.to_datetime(row[0]).date()
                        expiries.append(exp_date)
                    except:
                        pass
                elif isinstance(row[0], datetime):
                    expiries.append(row[0].date())
                elif isinstance(row[0], date):
                    expiries.append(row[0])
        
        return sorted(list(set(expiries)))
        
    except Exception as e:
        st.error(f"Error fetching expiries: {e}")
        import traceback
        st.code(traceback.format_exc())
        return []

def get_recommended_expiry(expiries: list, trading_horizon: str = "auto") -> tuple:
    """
    Get recommended expiry based on trading strategy and current date
    
    Args:
        expiries: List of available expiry dates
        trading_horizon: 'current_month', 'next_month', or 'auto'
    
    Returns:
        (recommended_expiry, reason)
    """
    if not expiries:
        return None, "No expiries available"
    
    today = date.today()
    current_month = today.month
    current_year = today.year
    
    # Get current day of month to determine trading week
    day_of_month = today.day
    
    # Categorize expiries
    current_month_expiries = []
    next_month_expiries = []
    later_expiries = []
    
    for exp in expiries:
        if exp < today:
            continue  # Skip expired
        
        if exp.year == current_year and exp.month == current_month:
            current_month_expiries.append(exp)
        elif (exp.year == current_year and exp.month == current_month + 1) or \
             (exp.year == current_year + 1 and current_month == 12 and exp.month == 1):
            next_month_expiries.append(exp)
        else:
            later_expiries.append(exp)
    
    # Trading logic based on week of month
    if trading_horizon == "auto":
        # Last week of month (day >= 24)
        if day_of_month >= 24:
            # Check if current month expiry is within 7 days
            if current_month_expiries:
                nearest_current = min(current_month_expiries)
                days_to_expiry = (nearest_current - today).days
                
                if days_to_expiry <= 7:
                    # Too close to expiry - prefer next month
                    if next_month_expiries:
                        recommended = min(next_month_expiries)
                        reason = f"🎯 NEXT MONTH (Current expiry in {days_to_expiry} days - too close)"
                    elif current_month_expiries:
                        recommended = nearest_current
                        reason = f"⚠️ CURRENT MONTH (No next month available, {days_to_expiry} days left)"
                    else:
                        recommended = expiries[0]
                        reason = "📅 NEAREST AVAILABLE"
                else:
                    # Still safe to trade current month
                    recommended = nearest_current
                    reason = f"✅ CURRENT MONTH ({days_to_expiry} days to expiry)"
            elif next_month_expiries:
                recommended = min(next_month_expiries)
                reason = "🎯 NEXT MONTH (Current month expired)"
            else:
                recommended = expiries[0]
                reason = "📅 NEAREST AVAILABLE"
        
        # First 2-3 weeks (day < 24)
        else:
            if current_month_expiries:
                recommended = min(current_month_expiries)
                days_to_expiry = (recommended - today).days
                reason = f"✅ CURRENT MONTH ({days_to_expiry} days to expiry)"
            elif next_month_expiries:
                recommended = min(next_month_expiries)
                reason = "🎯 NEXT MONTH"
            else:
                recommended = expiries[0]
                reason = "📅 NEAREST AVAILABLE"
    
    elif trading_horizon == "current_month":
        if current_month_expiries:
            recommended = min(current_month_expiries)
            days_to_expiry = (recommended - today).days
            reason = f"✅ CURRENT MONTH (Manual selection, {days_to_expiry} days left)"
        else:
            recommended = expiries[0]
            reason = "⚠️ Current month not available - using nearest"
    
    elif trading_horizon == "next_month":
        if next_month_expiries:
            recommended = min(next_month_expiries)
            reason = "🎯 NEXT MONTH (Manual selection)"
        elif current_month_expiries:
            recommended = min(current_month_expiries)
            reason = "⚠️ Next month not available - using current"
        else:
            recommended = expiries[0]
            reason = "📅 Using nearest available"
    
    else:
        recommended = expiries[0]
        reason = "📅 NEAREST AVAILABLE"
    
    return recommended, reason

def build_option_chain(symbol: str, expiry: date, db: TradingDB, token: str) -> pd.DataFrame:
    """
    Build option chain for a symbol and expiry
    
    Returns DataFrame with columns:
    - Strike
    - CE_Symbol, CE_InstrumentKey, CE_LTP
    - PE_Symbol, PE_InstrumentKey, PE_LTP
    """
    
    # Query based on trading_symbol pattern
    # Format: "RELIANCE 1280 CE 30 DEC 25"
    query = f"""
        SELECT 
            trading_symbol,
            instrument_key,
            strike_price as strike,
            instrument_type as option_type,
            lot_size,
            tick_size
        FROM instruments
        WHERE trading_symbol LIKE '{symbol} %'
          AND segment = 'NSE_FO'
          AND expiry = '{expiry}'
          AND (instrument_type = 'CE' OR instrument_type = 'PE')
          AND strike_price > 0
        ORDER BY strike_price, instrument_type
    """
    
    try:
        options_df = db.con.execute(query).df()
        
        if options_df.empty:
            st.warning(f"⚠️ No options found for {symbol} with expiry {expiry}")
            st.info("💡 **Tip:** Try a different expiry date or check if this symbol has F&O contracts")
            
            # Debug: Show what we're searching for
            with st.expander("🔍 Debug Info"):
                st.code(f"Query: trading_symbol LIKE '{symbol} %' AND expiry = '{expiry}'")
                
                # Check if symbol exists at all
                test_query = f"""
                    SELECT COUNT(*) as count
                    FROM instruments  
                    WHERE trading_symbol LIKE '{symbol} %'
                      AND segment = 'NSE_FO'
                """
                
                count = db.con.execute(test_query).fetchone()[0]
                st.write(f"Found {count} total F&O contracts for {symbol}")
            
            return pd.DataFrame()
        
        # Get unique strikes
        strikes = sorted(options_df['strike'].unique())
        
        st.caption(f"✅ Found {len(strikes)} strikes with {len(options_df)} total contracts")
        
        # Build chain structure
        chain_data = []
        
        for strike in strikes:
            row_data = {'Strike': strike}
            
            # CE data
            ce = options_df[(options_df['strike'] == strike) & (options_df['option_type'] == 'CE')]
            if not ce.empty:
                row_data['CE_Symbol'] = ce.iloc[0]['trading_symbol']
                row_data['CE_InstrumentKey'] = ce.iloc[0]['instrument_key']
                row_data['CE_LotSize'] = ce.iloc[0]['lot_size']
            
            # PE data
            pe = options_df[(options_df['strike'] == strike) & (options_df['option_type'] == 'PE')]
            if not pe.empty:
                row_data['PE_Symbol'] = pe.iloc[0]['trading_symbol']
                row_data['PE_InstrumentKey'] = pe.iloc[0]['instrument_key']
                row_data['PE_LotSize'] = pe.iloc[0]['lot_size']
            
            chain_data.append(row_data)
        
        chain_df = pd.DataFrame(chain_data)
        
        # Fetch LTPs for all options
        api = UpstoxOptions(token)
        
        all_keys = []
        if 'CE_InstrumentKey' in chain_df.columns:
            all_keys.extend(chain_df['CE_InstrumentKey'].dropna().tolist())
        if 'PE_InstrumentKey' in chain_df.columns:
            all_keys.extend(chain_df['PE_InstrumentKey'].dropna().tolist())
        
        if all_keys:
            with st.spinner(f"Fetching live prices for {len(all_keys)} options..."):
                try:
                    ltp_data = api.get_ltp(all_keys)
                    
                    # Debug
                    st.caption(f"API returned data for {len(ltp_data)} instruments")
                    
                    if not ltp_data:
                        st.warning("⚠️ No LTP data returned from API. Using manual entry option.")
                        st.info("💡 **Tip:** Market might be closed or API rate limit hit. You can still use Strike Selector with manual prices.")
                    
                except Exception as e:
                    st.error(f"Error fetching LTP: {e}")
                    ltp_data = {}
            
            # Add LTP to chain
            ltps_added = 0
            for idx, row in chain_df.iterrows():
                if 'CE_InstrumentKey' in row and pd.notna(row['CE_InstrumentKey']):
                    ce_key = row['CE_InstrumentKey']
                    if ce_key in ltp_data:
                        ce_ltp = ltp_data[ce_key].get('last_price', 0)
                        chain_df.at[idx, 'CE_LTP'] = ce_ltp
                        if ce_ltp > 0:
                            ltps_added += 1
                    else:
                        chain_df.at[idx, 'CE_LTP'] = 0.0
                
                if 'PE_InstrumentKey' in row and pd.notna(row['PE_InstrumentKey']):
                    pe_key = row['PE_InstrumentKey']
                    if pe_key in ltp_data:
                        pe_ltp = ltp_data[pe_key].get('last_price', 0)
                        chain_df.at[idx, 'PE_LTP'] = pe_ltp
                        if pe_ltp > 0:
                            ltps_added += 1
                    else:
                        chain_df.at[idx, 'PE_LTP'] = 0.0
            
            if ltps_added == 0:
                st.warning(f"⚠️ Could not fetch any live prices. Requested {len(all_keys)} instruments but got 0 valid prices.")
                
                # Offer manual entry
                with st.expander("✏️ Manual Price Entry (For Testing)", expanded=True):
                    st.info("💡 **Market Closed or API Issue?** You can manually enter approximate prices to test the Strike Selector.")
                    
                    # Estimate prices based on moneyness
                    spot_estimate = chain_df['Strike'].median()
                    st.write(f"Estimated spot price: ₹{spot_estimate:.2f}")
                    
                    use_estimated = st.checkbox("Use estimated prices based on moneyness", value=True)
                    
                    if use_estimated:
                        # Simple estimation: OTM options cheaper, ITM more expensive
                        for idx, row in chain_df.iterrows():
                            strike = row['Strike']
                            
                            # CE estimation
                            if 'CE_InstrumentKey' in row and pd.notna(row['CE_InstrumentKey']):
                                moneyness = (strike - spot_estimate) / spot_estimate * 100
                                if moneyness < -5:  # Deep ITM
                                    estimated_ce = spot_estimate - strike + 20
                                elif moneyness < 0:  # ITM
                                    estimated_ce = spot_estimate - strike + 10
                                elif moneyness < 2:  # ATM
                                    estimated_ce = spot_estimate * 0.015  # 1.5% of spot
                                elif moneyness < 5:  # OTM
                                    estimated_ce = spot_estimate * 0.008  # 0.8% of spot
                                else:  # Far OTM
                                    estimated_ce = spot_estimate * 0.003  # 0.3% of spot
                                
                                chain_df.at[idx, 'CE_LTP'] = max(estimated_ce, 0.5)
                            
                            # PE estimation  
                            if 'PE_InstrumentKey' in row and pd.notna(row['PE_InstrumentKey']):
                                moneyness = (strike - spot_estimate) / spot_estimate * 100
                                if moneyness > 5:  # Deep ITM
                                    estimated_pe = strike - spot_estimate + 20
                                elif moneyness > 0:  # ITM
                                    estimated_pe = strike - spot_estimate + 10
                                elif moneyness > -2:  # ATM
                                    estimated_pe = spot_estimate * 0.015  # 1.5% of spot
                                elif moneyness > -5:  # OTM
                                    estimated_pe = spot_estimate * 0.008  # 0.8% of spot
                                else:  # Far OTM
                                    estimated_pe = spot_estimate * 0.003  # 0.3% of spot
                                
                                chain_df.at[idx, 'PE_LTP'] = max(estimated_pe, 0.5)
                        
                        st.success("✅ Estimated prices added! You can now use Strike Selector.")
                        st.caption("⚠️ Note: These are approximations for testing only. Use real market prices for actual trading.")
            else:
                st.success(f"✅ Fetched {ltps_added} live prices")
        
        return chain_df
        
    except Exception as e:
        st.error(f"❌ Error building option chain: {e}")
        import traceback
        st.code(traceback.format_exc())
        return pd.DataFrame()

# =====================================================================
# STRIKE SELECTION HELPERS
# =====================================================================

def suggest_strikes(underlying_price: float, chain_df: pd.DataFrame, strategy: str = "ATM") -> dict:
    """
    Suggest optimal strikes based on strategy
    
    Strategies:
    - ATM: At-the-money
    - OTM1: 1 strike out-of-the-money
    - OTM2: 2 strikes out-of-the-money
    - ITM1: 1 strike in-the-money
    """
    
    if chain_df.empty:
        return {}
    
    strikes = chain_df['Strike'].tolist()
    
    # Find ATM
    atm_strike = min(strikes, key=lambda x: abs(x - underlying_price))
    atm_idx = strikes.index(atm_strike)
    
    suggestions = {
        'ATM': atm_strike,
        'ATM_CE': atm_strike,
        'ATM_PE': atm_strike
    }
    
    # OTM for calls (higher strikes)
    if atm_idx + 1 < len(strikes):
        suggestions['OTM1_CE'] = strikes[atm_idx + 1]
    if atm_idx + 2 < len(strikes):
        suggestions['OTM2_CE'] = strikes[atm_idx + 2]
    
    # OTM for puts (lower strikes)
    if atm_idx - 1 >= 0:
        suggestions['OTM1_PE'] = strikes[atm_idx - 1]
    if atm_idx - 2 >= 0:
        suggestions['OTM2_PE'] = strikes[atm_idx - 2]
    
    # ITM
    if atm_idx - 1 >= 0:
        suggestions['ITM1_CE'] = strikes[atm_idx - 1]
    if atm_idx + 1 < len(strikes):
        suggestions['ITM1_PE'] = strikes[atm_idx + 1]
    
    return suggestions

# =====================================================================
# PAPER TRADING
# =====================================================================

def place_paper_trade(
    symbol: str,
    instrument_key: str,
    option_type: str,
    strike: float,
    expiry: date,
    action: str,
    quantity: int,
    price: float,
    notes: str = ""
):
    """Place a paper trade"""
    db = get_db()
    
    try:
        # Insert trade
        db.con.execute(f"""
            INSERT INTO {PAPER_TRADES_TABLE}
            (symbol, instrument_key, option_type, strike, expiry, action, 
             quantity, entry_price, status, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)
        """, [symbol, instrument_key, option_type, strike, expiry, action, 
              quantity, price, notes])
        
        # Update/create position
        existing = db.con.execute(f"""
            SELECT position_id, quantity FROM {POSITIONS_TABLE}
            WHERE instrument_key = ?
        """, [instrument_key]).fetchone()
        
        if existing:
            # Update existing position
            new_qty = existing[1] + (quantity if action == 'BUY' else -quantity)
            
            if new_qty == 0:
                # Close position
                db.con.execute(f"""
                    DELETE FROM {POSITIONS_TABLE}
                    WHERE position_id = ?
                """, [existing[0]])
            else:
                db.con.execute(f"""
                    UPDATE {POSITIONS_TABLE}
                    SET quantity = ?,
                        current_price = ?,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE position_id = ?
                """, [new_qty, price, existing[0]])
        else:
            # New position
            if action == 'BUY':
                db.con.execute(f"""
                    INSERT INTO {POSITIONS_TABLE}
                    (symbol, instrument_key, option_type, strike, expiry,
                     quantity, entry_price, entry_timestamp, current_price, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, CURRENT_TIMESTAMP)
                """, [symbol, instrument_key, option_type, strike, expiry,
                      quantity, price, price])
        
        return True, "Trade placed successfully"
        
    except Exception as e:
        return False, f"Error: {e}"

def get_open_positions() -> pd.DataFrame:
    """Get all open positions"""
    db = get_db()
    
    try:
        query = f"""
            SELECT * FROM {POSITIONS_TABLE}
            ORDER BY entry_timestamp DESC
        """
        
        return db.con.execute(query).df()
        
    except:
        return pd.DataFrame()

def close_position(position_id: int, exit_price: float):
    """Close a position"""
    db = get_db()
    
    try:
        # Get position details
        pos = db.con.execute(f"""
            SELECT * FROM {POSITIONS_TABLE}
            WHERE position_id = ?
        """, [position_id]).fetchone()
        
        if not pos:
            return False, "Position not found"
        
        # Calculate P&L
        entry_price = pos[7]  # entry_price column
        quantity = pos[6]  # quantity column
        
        pnl = (exit_price - entry_price) * quantity
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        
        # Update trade record
        db.con.execute(f"""
            UPDATE {PAPER_TRADES_TABLE}
            SET exit_price = ?,
                exit_timestamp = CURRENT_TIMESTAMP,
                status = 'CLOSED',
                pnl = ?,
                pnl_pct = ?
            WHERE instrument_key = ?
              AND status = 'OPEN'
        """, [exit_price, pnl, pnl_pct, pos[2]])  # instrument_key
        
        # Delete position
        db.con.execute(f"""
            DELETE FROM {POSITIONS_TABLE}
            WHERE position_id = ?
        """, [position_id])
        
        return True, f"Position closed | P&L: ₹{pnl:.2f} ({pnl_pct:+.2f}%)"
        
    except Exception as e:
        return False, f"Error: {e}"

# =====================================================================
# MAIN UI
# =====================================================================

def main():
    st.title("📈 Options Trading & Paper Trading")
    st.caption("Analyze option chains and practice with paper trading")
    
    # Initialize tables
    init_tables()
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Option Chain",
        "🎯 Strike Selector", 
        "💼 Paper Trading",
        "📒 Trade Journal"
    ])
    
    # ========== TAB 1: OPTION CHAIN ==========
    with tab1:
        st.header("Option Chain Viewer")
        
        # Quick select popular stocks
        with st.expander("🚀 Quick Select F&O Stocks", expanded=False):
            popular_stocks = [
                "RELIANCE", "TATAMOTORS", "HDFCBANK", "ICICIBANK", "INFY",
                "TCS", "WIPRO", "SBIN", "AXISBANK", "KOTAKBANK",
                "BAJFINANCE", "MARUTI", "HINDUNILVR", "ITC", "LT",
                "TATASTEEL", "ADANIPORTS", "BHARTIARTL", "HINDALCO", "IDEA"
            ]
            
            cols = st.columns(5)
            for idx, stock in enumerate(popular_stocks):
                with cols[idx % 5]:
                    if st.button(stock, key=f"quick_{stock}", use_container_width=True):
                        st.session_state['option_symbol'] = stock
                        st.rerun()
        
        # Expiry strategy selector
        with st.expander("⚙️ Expiry Selection Strategy", expanded=False):
            col_a, col_b = st.columns(2)
            
            with col_a:
                expiry_strategy = st.radio(
                    "Selection Mode",
                    options=["Auto (Smart)", "Current Month", "Next Month", "Manual"],
                    index=0,
                    help="Auto mode selects based on days to expiry"
                )
            
            with col_b:
                st.markdown("**Rules:**")
                st.caption("• Days 1-23: Current month expiry")
                st.caption("• Days 24-31 (< 7 days to expiry): Next month")
                st.caption("• < 7 days warning: High theta decay")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Symbol selection - use session state to preserve value
            if 'option_symbol' not in st.session_state:
                st.session_state['option_symbol'] = 'RELIANCE'
            
            symbol_input = st.text_input(
                "Underlying Symbol", 
                value=st.session_state['option_symbol'],
                key="symbol_input",
                help="Enter NSE symbol (e.g., RELIANCE, SRF, TATAMOTORS)"
            )
            
            # Convert to uppercase for database matching
            symbol = symbol_input.strip().upper()
            
            # Update session state
            st.session_state['option_symbol'] = symbol
            
            # Validate symbol exists in F&O
            if symbol:
                validation_query = f"""
                    SELECT COUNT(*) as count
                    FROM instruments
                    WHERE trading_symbol LIKE '{symbol} %'
                      AND segment = 'NSE_FO'
                    LIMIT 1
                """
                
                try:
                    result = db.con.execute(validation_query).fetchone()
                    if result and result[0] > 0:
                        st.success(f"✓ {symbol} found ({result[0]} contracts)")
                    else:
                        st.error(f"✗ {symbol} not found in F&O")
                        st.caption("💡 Try: RELIANCE, TATAMOTORS, WIPRO, IDEA")
                except:
                    pass
        
        db = get_db()
        token = get_access_token()
        
        if not token:
            st.error("Please login first (Page 1)")
            st.stop()
        
        with col2:
            # Get expiries
            expiries = get_fo_expiries(symbol, db)
            
            if not expiries:
                st.warning(f"No F&O contracts found for {symbol}")
                st.stop()
            
            # Map strategy selection
            strategy_map = {
                "Auto (Smart)": "auto",
                "Current Month": "current_month",
                "Next Month": "next_month",
                "Manual": "manual"
            }
            
            selected_strategy = strategy_map.get(expiry_strategy, "auto")
            
            # Get recommended expiry (if not manual)
            if selected_strategy != "manual":
                recommended_expiry, reason = get_recommended_expiry(expiries, trading_horizon=selected_strategy)
                
                # Show recommendation
                if recommended_expiry:
                    st.info(f"💡 {reason}")
                
                # Set default
                default_idx = 0
                if recommended_expiry and recommended_expiry in expiries:
                    default_idx = expiries.index(recommended_expiry)
            else:
                default_idx = 0
                st.caption("📝 Manual mode - select any expiry")
            
            expiry = st.selectbox(
                "Expiry Date",
                options=expiries,
                index=default_idx,
                format_func=lambda x: f"{x.strftime('%d %b %Y')} ({(x - date.today()).days} days)",
                help="Select expiry based on your trading horizon"
            )
            
            # Show days to expiry warning/info
            if expiry:
                days_left = (expiry - date.today()).days
                
                col_warn1, col_warn2 = st.columns([1, 3])
                
                with col_warn1:
                    if days_left <= 7:
                        st.error(f"⚠️ {days_left}d")
                    elif days_left <= 14:
                        st.warning(f"📅 {days_left}d")
                    else:
                        st.success(f"✅ {days_left}d")
                
                with col_warn2:
                    if days_left <= 7:
                        st.caption("High theta decay")
                    elif days_left <= 14:
                        st.caption("Moderate time decay")
                    else:
                        st.caption("Good time value")
        
        with col3:
            if st.button("🔄 Load Chain", type="primary", use_container_width=True):
                st.rerun()
        
        # Build and display chain
        if expiry:
            chain_df = build_option_chain(symbol, expiry, db, token)
            
            if not chain_df.empty:
                st.success(f"✅ Loaded {len(chain_df)} strikes")
                
                # Display chain
                st.dataframe(
                    chain_df.style.format({
                        'Strike': '{:.2f}',
                        'CE_LTP': '{:.2f}',
                        'PE_LTP': '{:.2f}'
                    }).background_gradient(
                        subset=['CE_LTP', 'PE_LTP'],
                        cmap='RdYlGn'
                    ),
                    use_container_width=True,
                    height=500
                )
                
                # Store in session for other tabs
                st.session_state['chain_df'] = chain_df
                st.session_state['symbol'] = symbol
                st.session_state['expiry'] = expiry
            else:
                st.warning("No option chain data available")
    
    # ========== TAB 2: STRIKE SELECTOR ==========
    with tab2:
        st.header("🎯 Strike Selector & Trade Analyzer")
        
        if 'chain_df' not in st.session_state:
            st.info("👈 Please load an option chain first (Tab 1)")
        else:
            chain_df = st.session_state['chain_df']
            symbol = st.session_state['symbol']
            expiry = st.session_state['expiry']
            
            st.success(f"📊 Analyzing {symbol} | Expiry: {expiry.strftime('%d %b %Y')}")
            
            # Input section
            st.subheader("📍 Market Context")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                spot_price = st.number_input(
                    "Current Spot Price",
                    value=float(chain_df['Strike'].median()),
                    step=10.0,
                    help="Current price of the underlying stock"
                )
            
            with col2:
                direction = st.radio(
                    "Trade Direction",
                    options=["Bullish (Buy CE)", "Bearish (Buy PE)"],
                    horizontal=True
                )
                is_bullish = "Bullish" in direction
            
            with col3:
                target_move_pct = st.number_input(
                    "Expected Move %",
                    value=3.0,
                    step=0.5,
                    help="How much % do you expect the stock to move?"
                )
            
            st.divider()
            
            # Calculate target price
            if is_bullish:
                target_price = spot_price * (1 + target_move_pct/100)
                st.info(f"🎯 **Target Price:** ₹{target_price:.2f} ({target_move_pct:+.1f}% move)")
            else:
                target_price = spot_price * (1 - target_move_pct/100)
                st.info(f"🎯 **Target Price:** ₹{target_price:.2f} ({target_move_pct:.1f}% move)")
            
            # Get time to expiry
            days_to_expiry = (expiry - date.today()).days
            time_to_expiry = calculate_time_to_expiry(expiry)
            
            # IV assumption
            iv = st.slider(
                "Implied Volatility (IV) %",
                min_value=10,
                max_value=100,
                value=30,
                help="Higher IV = More expensive options. Use 25-40% for most stocks"
            ) / 100
            
            st.divider()
            
            # Analyze all strikes
            st.subheader("📊 Strike Analysis & Comparison")
            
            # Debug: Check chain data
            with st.expander("🔍 Debug: Chain Data", expanded=False):
                st.write(f"Total strikes in chain: {len(chain_df)}")
                st.write(f"Columns: {chain_df.columns.tolist()}")
                
                if is_bullish:
                    ce_with_ltp = chain_df[chain_df.get('CE_LTP', pd.Series([0])) > 0]
                    st.write(f"CE options with LTP > 0: {len(ce_with_ltp)}")
                    if len(ce_with_ltp) > 0:
                        st.dataframe(ce_with_ltp[['Strike', 'CE_Symbol', 'CE_LTP']].head())
                else:
                    pe_with_ltp = chain_df[chain_df.get('PE_LTP', pd.Series([0])) > 0]
                    st.write(f"PE options with LTP > 0: {len(pe_with_ltp)}")
                    if len(pe_with_ltp) > 0:
                        st.dataframe(pe_with_ltp[['Strike', 'PE_Symbol', 'PE_LTP']].head())
            
            analysis_data = []
            
            option_col = 'CE' if is_bullish else 'PE'
            
            for idx, row in chain_df.iterrows():
                strike = row['Strike']
                
                # Get option details
                symbol_key = f"{option_col}_Symbol"
                inst_key_col = f"{option_col}_InstrumentKey"
                ltp_col = f"{option_col}_LTP"
                lot_col = f"{option_col}_LotSize"
                
                if symbol_key not in row or pd.isna(row[symbol_key]):
                    continue
                
                option_symbol = row[symbol_key]
                inst_key = row[inst_key_col] if pd.notna(row.get(inst_key_col)) else None
                ltp = row.get(ltp_col, 0)
                lot_size = row.get(lot_col, 1)
                
                # Skip if no valid LTP (check for both 0 and NaN)
                if pd.isna(ltp) or ltp <= 0:
                    continue
                
                # Classify strike
                moneyness_pct = ((strike - spot_price) / spot_price) * 100
                
                if abs(moneyness_pct) <= 1:
                    moneyness = "ATM"
                elif is_bullish:
                    if moneyness_pct > 0:
                        if moneyness_pct <= 2:
                            moneyness = "OTM1"
                        elif moneyness_pct <= 4:
                            moneyness = "OTM2"
                        else:
                            moneyness = "Far OTM"
                    else:
                        moneyness = "ITM"
                else:  # Bearish
                    if moneyness_pct < 0:
                        if moneyness_pct >= -2:
                            moneyness = "OTM1"
                        elif moneyness_pct >= -4:
                            moneyness = "OTM2"
                        else:
                            moneyness = "Far OTM"
                    else:
                        moneyness = "ITM"
                
                # Calculate Greeks
                greeks = calculate_greeks(
                    spot_price, strike, time_to_expiry,
                    iv, RISK_FREE_RATE, option_col
                )
                
                # Profit potential at target
                if is_bullish:
                    intrinsic_at_target = max(0, target_price - strike)
                else:
                    intrinsic_at_target = max(0, strike - target_price)
                
                # Estimate option price at target (simplified)
                # Real option will have time value too
                time_value_decay = 0.3  # Assume 30% time value decay
                estimated_price_at_target = intrinsic_at_target + (ltp * time_value_decay)
                
                profit_per_lot = (estimated_price_at_target - ltp) * lot_size
                profit_pct = ((estimated_price_at_target - ltp) / ltp) * 100 if ltp > 0 else 0
                
                # ROI calculation
                investment = ltp * lot_size
                roi_pct = (profit_per_lot / investment) * 100 if investment > 0 else 0
                
                # Breakeven
                if is_bullish:
                    breakeven = strike + ltp
                    breakeven_move_pct = ((breakeven - spot_price) / spot_price) * 100
                else:
                    breakeven = strike - ltp
                    breakeven_move_pct = ((spot_price - breakeven) / spot_price) * 100
                
                # Risk score (lower is better)
                # Based on: premium cost, delta, distance from spot
                risk_score = (
                    (abs(moneyness_pct) / 10) +  # Distance penalty
                    ((1 - abs(greeks['delta'])) * 5) +  # Delta penalty
                    (ltp / spot_price * 100)  # Premium penalty
                )
                
                # Reward score (higher is better)
                reward_score = (
                    profit_pct +
                    (abs(greeks['delta']) * 20) +
                    (roi_pct / 10)
                )
                
                # Risk-Reward ratio
                risk_reward = reward_score / risk_score if risk_score > 0 else 0
                
                analysis_data.append({
                    'Strike': strike,
                    'Moneyness': moneyness,
                    'LTP': ltp,
                    'Distance%': moneyness_pct,
                    'Delta': greeks['delta'],
                    'Gamma': greeks['gamma'],
                    'Theta': greeks['theta'],
                    'Vega': greeks['vega'],
                    'Premium': ltp * lot_size,
                    'Breakeven': breakeven,
                    'BE_Move%': breakeven_move_pct,
                    'Target_Price': estimated_price_at_target,
                    'Profit_₹': profit_per_lot,
                    'Profit%': profit_pct,
                    'ROI%': roi_pct,
                    'Risk_Score': risk_score,
                    'Reward_Score': reward_score,
                    'R:R': risk_reward,
                    'Symbol': option_symbol,
                    'InstKey': inst_key,
                    'LotSize': lot_size
                })
            
            if not analysis_data:
                st.warning("No options with valid prices found")
            else:
                analysis_df = pd.DataFrame(analysis_data)
                
                # Sort by Risk-Reward ratio
                analysis_df = analysis_df.sort_values('R:R', ascending=False)
                
                # Highlight best options
                st.success(f"✅ Analyzed {len(analysis_df)} strikes")
                
                # Top 3 recommendations
                st.subheader("🏆 Top 3 Recommended Strikes")
                
                top_3 = analysis_df.head(3)
                
                for i, (idx, row) in enumerate(top_3.iterrows(), 1):
                    with st.expander(f"#{i} - Strike {row['Strike']} ({row['Moneyness']}) - R:R {row['R:R']:.2f}", expanded=(i==1)):
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.markdown("**📊 Option Details**")
                            st.metric("Strike", f"₹{row['Strike']:.0f}")
                            st.metric("Premium (LTP)", f"₹{row['LTP']:.2f}")
                            st.metric("Lot Size", f"{row['LotSize']:.0f}")
                            st.metric("Investment", f"₹{row['Premium']:.0f}")
                        
                        with col_b:
                            st.markdown("**🎯 Profit Potential**")
                            st.metric("Expected Profit", f"₹{row['Profit_₹']:.0f}", f"{row['Profit%']:.1f}%")
                            st.metric("ROI", f"{row['ROI%']:.1f}%")
                            st.metric("Target Price", f"₹{row['Target_Price']:.2f}")
                            st.metric("Breakeven", f"₹{row['Breakeven']:.2f}", f"{row['BE_Move%']:.2f}%")
                        
                        with col_c:
                            st.markdown("**📈 Greeks & Risk**")
                            st.metric("Delta", f"{row['Delta']:.3f}")
                            st.metric("Theta (per day)", f"{row['Theta']:.3f}")
                            st.metric("Risk Score", f"{row['Risk_Score']:.2f}")
                            st.metric("R:R Ratio", f"{row['R:R']:.2f}")
                        
                        # Trade button
                        if st.button(f"🎯 Trade This Strike", key=f"trade_{row['Strike']}", type="primary"):
                            # Save to session for Tab 3
                            st.session_state['selected_strike'] = row['Strike']
                            st.session_state['selected_option_type'] = option_col
                            st.session_state['selected_inst_key'] = row['InstKey']
                            st.session_state['selected_ltp'] = row['LTP']
                            st.session_state['selected_lot_size'] = row['LotSize']
                            
                            st.success("✅ Strike selected! Go to 💼 Paper Trading tab to place trade")
                            st.balloons()
                
                st.divider()
                
                # Full comparison table
                st.subheader("📋 Full Strike Comparison")
                
                # Select columns to display
                display_cols = [
                    'Strike', 'Moneyness', 'LTP', 'Distance%', 'Delta',
                    'Breakeven', 'BE_Move%', 'Profit_₹', 'Profit%', 'ROI%', 'R:R'
                ]
                
                st.dataframe(
                    analysis_df[display_cols].style.format({
                        'Strike': '{:.0f}',
                        'LTP': '₹{:.2f}',
                        'Distance%': '{:+.2f}%',
                        'Delta': '{:.3f}',
                        'Breakeven': '₹{:.2f}',
                        'BE_Move%': '{:+.2f}%',
                        'Profit_₹': '₹{:.0f}',
                        'Profit%': '{:+.1f}%',
                        'ROI%': '{:.1f}%',
                        'R:R': '{:.2f}'
                    }).background_gradient(
                        subset=['R:R', 'ROI%'],
                        cmap='RdYlGn'
                    ),
                    use_container_width=True,
                    height=400
                )
                
                # Download analysis
                if st.button("📥 Download Analysis as CSV"):
                    csv = analysis_df.to_csv(index=False)
                    st.download_button(
                        "Download",
                        csv,
                        f"{symbol}_strike_analysis_{date.today()}.csv",
                        "text/csv"
                    )
    
    # ========== TAB 3: PAPER TRADING ==========
    with tab3:
        st.header("Paper Trading")
        
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.subheader("📊 Open Positions")
            
            positions_df = get_open_positions()
            
            if not positions_df.empty:
                st.dataframe(positions_df, use_container_width=True)
                
                # Close position
                pos_to_close = st.selectbox(
                    "Select position to close",
                    options=positions_df['position_id'].tolist(),
                    format_func=lambda x: f"ID {x}"
                )
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    exit_price = st.number_input("Exit Price", value=0.0, step=0.05)
                
                with col_b:
                    if st.button("Close Position", type="primary"):
                        success, msg = close_position(pos_to_close, exit_price)
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
            else:
                st.info("No open positions")
        
        with col_right:
            st.subheader("🎯 Place Trade")
            
            if 'chain_df' not in st.session_state:
                st.warning("Load option chain first")
            else:
                chain_df = st.session_state['chain_df']
                symbol = st.session_state['symbol']
                expiry = st.session_state['expiry']
                
                with st.form("place_trade_form"):
                    option_type = st.radio("Type", options=["CE", "PE"], horizontal=True)
                    
                    strike = st.selectbox(
                        "Strike",
                        options=chain_df['Strike'].tolist()
                    )
                    
                    action = st.radio("Action", options=["BUY", "SELL"], horizontal=True)
                    
                    quantity = st.number_input("Quantity (Lots)", value=1, min_value=1, step=1)
                    
                    # Get LTP for selected option
                    strike_row = chain_df[chain_df['Strike'] == strike].iloc[0]
                    
                    if option_type == 'CE':
                        default_price = strike_row.get('CE_LTP', 0)
                        inst_key = strike_row.get('CE_InstrumentKey', '')
                    else:
                        default_price = strike_row.get('PE_LTP', 0)
                        inst_key = strike_row.get('PE_InstrumentKey', '')
                    
                    price = st.number_input("Price", value=float(default_price), step=0.05)
                    
                    notes = st.text_area("Notes", placeholder="Entry reason...")
                    
                    submitted = st.form_submit_button("Place Trade", type="primary", use_container_width=True)
                    
                    if submitted:
                        success, msg = place_paper_trade(
                            symbol, inst_key, option_type, strike, expiry,
                            action, quantity, price, notes
                        )
                        
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
    
    # ========== TAB 4: TRADE JOURNAL ==========
    with tab4:
        st.header("Trade Journal")
        
        db = get_db()
        
        try:
            trades_df = db.con.execute(f"""
                SELECT * FROM {PAPER_TRADES_TABLE}
                ORDER BY timestamp DESC
                LIMIT 100
            """).df()
            
            if not trades_df.empty:
                # Summary stats
                closed = trades_df[trades_df['status'] == 'CLOSED']
                
                if not closed.empty:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_pnl = closed['pnl'].sum()
                        st.metric("Total P&L", f"₹{total_pnl:,.2f}")
                    
                    with col2:
                        win_rate = (closed['pnl'] > 0).mean() * 100
                        st.metric("Win Rate", f"{win_rate:.1f}%")
                    
                    with col3:
                        avg_profit = closed[closed['pnl'] > 0]['pnl'].mean()
                        st.metric("Avg Win", f"₹{avg_profit:,.2f}" if not pd.isna(avg_profit) else "₹0.00")
                    
                    with col4:
                        avg_loss = closed[closed['pnl'] < 0]['pnl'].mean()
                        st.metric("Avg Loss", f"₹{avg_loss:,.2f}" if not pd.isna(avg_loss) else "₹0.00")
                
                st.divider()
                
                # All trades
                st.dataframe(trades_df, use_container_width=True, height=400)
                
                # Export
                if st.button("📥 Export to CSV"):
                    csv = trades_df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        "paper_trades.csv",
                        "text/csv"
                    )
            else:
                st.info("No trades yet. Start trading in Tab 3!")
                
        except Exception as e:
            st.error(f"Error loading trades: {e}")

if __name__ == "__main__":
    main()