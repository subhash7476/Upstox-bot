# pages/10_Regime_Backtest_Validator.py
"""
Regime Persistence Backtester
Validates whether high persistence % actually predicts regime continuation

This answers the question: "If today shows 85% persistence, 
does the regime actually continue 85% of the time?"
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import os
from core.indicators import compute_supertrend
from core.config import get_access_token

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from core.regime_backtest import (
        backtest_regime_persistence,
        backtest_trade_signals,
        validate_current_signals
    )
    from core.regime_gmm import MarketRegimeGMM
except ImportError as e:
    st.error(f"Import error: {e}. Make sure regime_backtest.py is in core/ folder.")
    st.stop()

DERIVED_ROOT = Path("data/derived")

st.set_page_config(layout="wide", page_title="Regime Backtest Validator")
st.title("🧪 Regime Persistence Validator")

st.markdown("""
**Purpose:** Test if predicted persistence % actually translates to real results.

For example, if today shows **85% persistence**, does the regime actually continue 85% of the time in historical data?
""")

# Helper function
@st.cache_data(ttl=300)
def get_symbols_with_daily():
    symbols = []
    for sym_dir in DERIVED_ROOT.iterdir():
        if sym_dir.is_dir():
            daily_dir = sym_dir / "1day"
            if daily_dir.exists() and list(daily_dir.glob("*.parquet")):
                symbols.append(sym_dir.name)
    return sorted(symbols)

symbols = get_symbols_with_daily()
if not symbols:
    st.warning("No daily data found. Run resampler first.")
    st.stop()

# ========== SECTION 1: Single Symbol Persistence Backtest ==========
st.header("1️⃣ Backtest Persistence Predictions")

col1, col2, col3 = st.columns(3)

with col1:
    test_symbol = st.selectbox("Select Symbol", symbols)

with col2:
    n_regimes = st.slider("Regimes", 2, 6, 4, key="regimes_backtest")

with col3:
    persist_thresh = st.slider("Persistence Threshold", 0.5, 0.9, 0.7, 0.05, key="persist_backtest")

# Load data
sym_path = DERIVED_ROOT / test_symbol / "1day"
files = list(sym_path.glob("*.parquet"))

if not files:
    st.error(f"No data for {test_symbol}")
    st.stop()

df = pd.read_parquet(files[0])
df.index = pd.to_datetime(df.index)
df = df.sort_index()

st.info(f"Loaded {len(df)} days for {test_symbol} ({df.index[0].date()} to {df.index[-1].date()})")

# Test parameters
col1, col2 = st.columns(2)
with col1:
    test_days = st.number_input("Test on Recent N Days", 20, 100, 30, 
                                 help="How many recent days to test predictions on")
with col2:
    lookback = st.number_input("Min Training Days", 50, 200, 100,
                                help="Minimum historical data needed before making predictions")

if st.button("🧪 Run Persistence Backtest", key="run_persist"):
    with st.spinner("Backtesting persistence predictions..."):
        results = backtest_regime_persistence(
            df,
            n_regimes=n_regimes,
            persistence_threshold=persist_thresh,
            lookback_days=lookback,
            test_days=test_days
        )
    
    if 'Error' in results:
        st.error(results['Error'])
    else:
        st.session_state.persist_results = results
        
        # Display metrics
        st.subheader("📊 Backtest Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Overall Accuracy", f"{results['overall_accuracy']}%",
                     help="How often did the prediction match reality?")
        
        with col2:
            st.metric("Precision", f"{results['precision']}%",
                     help="When we predicted HIGH persistence, how often was it correct?")
        
        with col3:
            st.metric("Recall", f"{results['recall']}%",
                     help="Of all times regime persisted, how many did we catch?")
        
        with col4:
            st.metric("Test Days", results['test_days'])
        
        # Interpretation
        st.markdown("---")
        st.subheader("💡 What This Means")
        
        precision = results['precision']
        if precision >= 70:
            st.success(f"""
            ✅ **HIGH RELIABILITY** ({precision}% precision)
            
            When this model predicts high persistence (>{persist_thresh*100}%), 
            the regime actually continued {precision}% of the time.
            
            **Recommendation:** Trust these signals for {test_symbol}.
            """)
        elif precision >= 55:
            st.warning(f"""
            ⚠️ **MODERATE RELIABILITY** ({precision}% precision)
            
            Slightly better than a coin flip. Use with other confirmations.
            
            **Recommendation:** Combine with other indicators (Supertrend, RSI).
            """)
        else:
            st.error(f"""
            ❌ **LOW RELIABILITY** ({precision}% precision)
            
            The model is not accurately predicting persistence for {test_symbol}.
            
            **Recommendation:** Do NOT rely on this for {test_symbol}.
            """)
        
        # Regime breakdown
        st.markdown("---")
        st.subheader("🔍 Regime-Specific Accuracy")
        
        regime_data = []
        for regime, stats in results['regime_breakdown'].items():
            regime_data.append({
                'Regime': regime,
                'Test Count': stats['count'],
                'Accuracy %': round(stats['accuracy'] * 100, 1),
                'Avg Persistence %': round(stats['avg_persistence'] * 100, 1)
            })
        
        regime_df = pd.DataFrame(regime_data)
        regime_df = regime_df.sort_values('Accuracy %', ascending=False)
        st.dataframe(regime_df, use_container_width=True)
        
        # Show sample predictions
        st.markdown("---")
        st.subheader("📅 Sample Predictions (Last 10 Days)")
        
        results_df = results['results_df'].tail(10).copy()
        display_df = results_df[[
            'date', 'current_regime', 'persistence_prob', 
            'predicted_persist', 'actual_persist', 'correct'
        ]].copy()
        
        display_df['persistence_prob'] = (display_df['persistence_prob'] * 100).round(1)
        display_df.columns = ['Date', 'Regime', 'Persist %', 'Predicted', 'Actual', 'Correct']
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download
        csv = results['results_df'].to_csv(index=False)
        st.download_button("📥 Download Full Results", csv, f"{test_symbol}_persistence_backtest.csv")

# ========== SECTION 2: Trade Signal Backtest ==========
st.header("2️⃣ Backtest Trade Signals (Like Nifty 100 Scanner)")

st.markdown("""
This simulates your actual trading scenario:
1. Each day, check if stock is "tradeable" (Bullish + high persistence + confidence)
2. If yes, enter trade at close
3. Exit after 5 days or when regime changes
4. Calculate win rate and returns
""")

col1, col2, col3 = st.columns(3)

with col1:
    trade_confidence = st.slider("Min Confidence", 0.5, 0.9, 0.6, 0.05, key="conf_trade")

with col2:
    trade_min_duration = st.number_input("Min Regime Duration", 1, 10, 2, key="dur_trade")

with col3:
    trade_test_days = st.number_input("Test Period (Days)", 30, 120, 60, key="days_trade")

if st.button("🎯 Run Trade Signal Backtest", key="run_trade"):
    with st.spinner("Simulating trades..."):
        trade_results = backtest_trade_signals(
            df,
            n_regimes=n_regimes,
            persistence_threshold=persist_thresh,
            confidence_threshold=trade_confidence,
            min_duration=trade_min_duration,
            test_days=trade_test_days
        )
    
    if 'Error' in trade_results:
        st.error(trade_results['Error'])
    else:
        st.session_state.trade_results = trade_results
        
        # Display metrics
        st.subheader("📈 Trade Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", trade_results['total_trades'])
            st.metric("Win Rate", f"{trade_results['win_rate']}%")
        
        with col2:
            st.metric("Avg Return", f"{trade_results['avg_return']}%")
            st.metric("Avg Hold", f"{trade_results['avg_hold_days']} days")
        
        with col3:
            st.metric("Avg Win", f"{trade_results['avg_win']}%")
            st.metric("Avg Loss", f"{trade_results['avg_loss']}%")
        
        with col4:
            st.metric("Best Trade", f"{trade_results['best_trade']}%")
            st.metric("Worst Trade", f"{trade_results['worst_trade']}%")
        
        # Interpretation
        st.markdown("---")
        st.subheader("💡 Trade Signal Verdict")
        
        win_rate = trade_results['win_rate']
        avg_return = trade_results['avg_return']
        
        if win_rate >= 60 and avg_return > 0:
            st.success(f"""
            ✅ **STRONG SIGNALS** ({win_rate}% win rate, {avg_return}% avg return)
            
            The regime-based entry logic is working well for {test_symbol}.
            
            **Action:** These are tradeable signals!
            """)
        elif win_rate >= 50 and avg_return > 0:
            st.warning(f"""
            ⚠️ **MARGINAL EDGE** ({win_rate}% win rate, {avg_return}% avg return)
            
            Slight positive expectancy, but requires good risk management.
            
            **Action:** Use tight stops and position sizing.
            """)
        else:
            st.error(f"""
            ❌ **WEAK SIGNALS** ({win_rate}% win rate, {avg_return}% avg return)
            
            Regime signals are not profitable for {test_symbol} with current settings.
            
            **Action:** Adjust parameters or skip this stock.
            """)
        
        # Show trades
        st.markdown("---")
        st.subheader("📋 Trade Log")
        
        trades_df = trade_results['trades_df'].copy()
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date']).dt.date
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date']).dt.date
        trades_df = trades_df.round(2)
        
        st.dataframe(trades_df, use_container_width=True)
        
        # Download
        csv = trades_df.to_csv(index=False)
        st.download_button("📥 Download Trades", csv, f"{test_symbol}_trades.csv")

# ========== SECTION 3: Validate Current Trade Zone Stocks ==========
st.header("3️⃣ Validate Your Current Trade Zone")

st.markdown("""
Paste the symbols from your Nifty 100 scan (e.g., SIEMENS, JINDALSTEL, ADANIGREEN).  
This will backtest each one and tell you which signals are historically reliable.
""")

trade_zone_input = st.text_area(
    "Enter Symbols (one per line or comma-separated)",
    value="SIEMENS\nJINDALSTEL\nADANIGREEN\nADANIENT\nBAJAJHFL\nGODREJCP",
    height=150
)

if st.button("🔍 Validate Trade Zone Stocks", key="validate_zone"):
    # Parse symbols
    if ',' in trade_zone_input:
        trade_symbols = [s.strip() for s in trade_zone_input.split(',')]
    else:
        trade_symbols = [s.strip() for s in trade_zone_input.split('\n') if s.strip()]
    
    st.info(f"Validating {len(trade_symbols)} symbols...")
    
    validation_results = []
    progress = st.progress(0)
    
    for i, sym in enumerate(trade_symbols):
        sym_path = DERIVED_ROOT / sym / "1day"
        files = list(sym_path.glob("*.parquet"))
        
        if not files:
            progress.progress((i + 1) / len(trade_symbols))
            continue
        
        try:
            sym_df = pd.read_parquet(files[0])
            sym_df.index = pd.to_datetime(sym_df.index)
            sym_df = sym_df.sort_index()
            
            # Backtest persistence
            persist_backtest = backtest_regime_persistence(
                sym_df,
                n_regimes=n_regimes,
                persistence_threshold=persist_thresh,
                test_days=30
            )
            
            # Backtest trades
            trade_backtest = backtest_trade_signals(
                sym_df,
                n_regimes=n_regimes,
                persistence_threshold=persist_thresh,
                confidence_threshold=trade_confidence,
                min_duration=trade_min_duration,
                test_days=60
            )
            
            if 'Error' not in persist_backtest and 'Error' not in trade_backtest:
                # Current prediction
                gmm = MarketRegimeGMM(n_regimes=n_regimes)
                df_regimes = gmm.detect_regimes(sym_df)
                current = gmm.predict_next_regime(df_regimes, threshold=persist_thresh)
                
                validation_results.append({
                    'Symbol': sym,
                    'Current Persist %': current.get('Persistence Prob %', 0),
                    'Historical Accuracy %': persist_backtest['overall_accuracy'],
                    'Historical Precision %': persist_backtest['precision'],
                    'Trade Win Rate %': trade_backtest['win_rate'],
                    'Trade Avg Return %': trade_backtest['avg_return'],
                    'Verdict': 'TRADE' if (persist_backtest['precision'] > 60 and trade_backtest['win_rate'] > 55) else 'SKIP'
                })
        
        except Exception:
            pass
        
        progress.progress((i + 1) / len(trade_symbols))
    
    if validation_results:
        val_df = pd.DataFrame(validation_results)
        val_df = val_df.sort_values('Trade Win Rate %', ascending=False)
        
        st.subheader(f"✅ Validation Results ({len(val_df)} stocks)")
        st.dataframe(val_df, use_container_width=True)
        
        # Summary
        tradeable = val_df[val_df['Verdict'] == 'TRADE']
        st.metric("Historically Reliable Stocks", f"{len(tradeable)} / {len(val_df)}")
        
        if len(tradeable) > 0:
            st.success(f"**TRADE THESE:** {', '.join(tradeable['Symbol'].tolist())}")
        
        # Download
        csv = val_df.to_csv(index=False)
        st.download_button("📥 Download Validation", csv, "trade_zone_validation.csv")
    else:
        st.warning("No valid results. Check if symbols have enough data.")

# Footer
st.markdown("---")
st.caption("💡 Pro tip: Only trade stocks with >60% historical precision AND >55% win rate.")