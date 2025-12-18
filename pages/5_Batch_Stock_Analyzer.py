# pages/9_Batch_Stock_Analyzer.py
"""
Batch Stock Analyzer - Run Multiple Symbols & Strategies
----------------------------------------------------------
Systematically test strategies across Nifty 100 or custom symbol lists
Compare performance, export results, and identify best performers
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import strategies
try:
    from core.strategies.mean_reversion import mean_reversion_basic, STRATEGY_INFO as MR_INFO
    from core.strategies.opening_range import opening_range_breakout, STRATEGY_INFO as ORB_INFO
    from core.strategies.vwap_strategy import vwap_mean_reversion, STRATEGY_INFO as VWAP_INFO
    from core.strategies.simple_momentum import simple_momentum_strategy, STRATEGY_INFO as MOM_INFO
except ImportError:
    st.error("Strategy modules not found. Please ensure core/strategies/ exists.")
    st.stop()

from core.metrics import compute_metrics

st.set_page_config(layout="wide", page_title="Batch Stock Analyzer")

# ==============================================================================
# BACKTESTER (Same as Strategy Lab)
# ==============================================================================
def backtest_strategy(df: pd.DataFrame,
                     initial_capital: float = 100000,
                     risk_per_trade_pct: float = 1.0,
                     stop_loss_pct: float = 0.5,
                     take_profit_pct: float = 1.5,
                     max_holding_bars: int = 100,
                     min_holding_bars: int = 5,
                     cooldown_bars: int = 2,
                     max_trades_per_day: int = 5) -> tuple:
    """
    Clean backtester with trade frequency controls
    
    New Features:
    - Cooldown period after each trade (prevents whipsaw)
    - Max trades per day limit (reduces overtrading)
    - Minimum holding period enforcement
    
    Returns: (trades_df, equity_curve, daily_returns)
    """
    balance = initial_capital
    position = None
    trades = []
    equity = []
    
    cooldown_until = 0  # Bar index until which we're in cooldown
    trades_today = 0
    current_date = None
    
    for i in range(len(df)):
        bar = df.iloc[i]
        
        # Reset daily trade counter
        bar_date = bar.name.date() if hasattr(bar.name, 'date') else None
        if bar_date != current_date:
            current_date = bar_date
            trades_today = 0
        
        # Check for exit if in position
        if position is not None:
            side = position['side']
            entry_price = position['entry_price']
            entry_bar = position['entry_bar']
            bars_held = i - entry_bar
            
            # Calculate P&L
            if side == 'LONG':
                current_pnl_pct = (bar['Close'] - entry_price) / entry_price * 100
            else:  # SHORT
                current_pnl_pct = (entry_price - bar['Close']) / entry_price * 100
            
            # Exit conditions (PRIORITY ORDER)
            exit_reason = None
            
            # 1. Stop loss hit (IMMEDIATE EXIT)
            if current_pnl_pct <= -stop_loss_pct:
                exit_reason = 'SL'
            
            # 2. Take profit hit (IMMEDIATE EXIT)
            elif current_pnl_pct >= take_profit_pct:
                exit_reason = 'TP'
            
            # 3. Max holding period exceeded
            elif bars_held >= max_holding_bars:
                exit_reason = 'TIME'
            
            # 4. Signal exit (only after min holding time)
            elif bars_held >= min_holding_bars:
                signal = bar.get('Signal', 0)
                if (side == 'LONG' and signal == -1) or (side == 'SHORT' and signal == 1):
                    exit_reason = 'SIGNAL'
            
            # Execute exit
            if exit_reason:
                exit_price = bar['Close']
                pnl = (exit_price - entry_price) * position['qty'] if side == 'LONG' else (entry_price - exit_price) * position['qty']
                pnl_pct = current_pnl_pct
                
                balance += pnl
                
                trades.append({
                    'Entry': df.index[entry_bar],
                    'Exit': df.index[i],
                    'Side': side,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'qty': position['qty'],
                    'pnl': pnl,  # lowercase to match compute_metrics
                    'pnl_pct': pnl_pct,
                    'bars_held': bars_held,
                    'exit_reason': exit_reason
                })
                
                position = None
                cooldown_until = i + cooldown_bars  # Set cooldown period
                trades_today += 1
        
        # Check for new entry (with filters)
        if position is None:
            # Filter 1: Cooldown period
            if i < cooldown_until:
                equity.append(balance)
                continue
            
            # Filter 2: Max trades per day
            if max_trades_per_day > 0 and trades_today >= max_trades_per_day:
                equity.append(balance)
                continue
            
            signal = bar.get('Signal', 0)
            
            if signal == 1:  # LONG
                qty = int((balance * risk_per_trade_pct / 100) / (bar['Close'] * stop_loss_pct / 100))
                qty = max(1, qty)
                
                position = {
                    'side': 'LONG',
                    'entry_price': bar['Close'],
                    'entry_bar': i,
                    'qty': qty
                }
            
            elif signal == -1:  # SHORT
                qty = int((balance * risk_per_trade_pct / 100) / (bar['Close'] * stop_loss_pct / 100))
                qty = max(1, qty)
                
                position = {
                    'side': 'SHORT',
                    'entry_price': bar['Close'],
                    'entry_bar': i,
                    'qty': qty
                }
        
        equity.append(balance)
    
    trades_df = pd.DataFrame(trades)
    
    # Calculate daily returns
    equity_series = pd.Series(equity, index=df.index)
    daily_returns = equity_series.resample('D').last().pct_change().dropna()
    
    return trades_df, equity, daily_returns


# ==============================================================================
# BATCH RUNNER
# ==============================================================================
def run_batch_analysis(symbols: list, 
                       strategies: dict,
                       timeframe: str,
                       params: dict,
                       progress_callback=None) -> pd.DataFrame:
    """
    Run backtests across multiple symbols and strategies
    
    Args:
        symbols: List of symbol names
        strategies: Dict of {strategy_name: (strategy_func, strategy_params)}
        timeframe: Timeframe string (5minute, 15minute, etc.)
        params: Backtest parameters (capital, risk, sl, tp)
        progress_callback: Function to report progress
    
    Returns:
        DataFrame with results for each symbol-strategy combination
    """
    results = []
    total_runs = len(symbols) * len(strategies)
    current_run = 0
    
    DATA_DIR = Path("data/derived")
    
    for symbol in symbols:
        # Find data file for this symbol
        symbol_path = DATA_DIR / symbol / timeframe
        
        if not symbol_path.exists():
            if progress_callback:
                progress_callback(f"⚠️ No data found for {symbol} at {timeframe}")
            continue
        
        # Get latest merged file
        parquet_files = list(symbol_path.glob("*.parquet"))
        if not parquet_files:
            if progress_callback:
                progress_callback(f"⚠️ No parquet files for {symbol}")
            continue
        
        data_file = sorted(parquet_files)[-1]  # Use most recent
        
        try:
            # Load data
            df = pd.read_parquet(data_file)
            
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
            
            # Standardize columns
            df.columns = [c.title() if c.lower() in ['open', 'high', 'low', 'close', 'volume'] else c 
                          for c in df.columns]
            
            # Test each strategy
            for strategy_name, (strategy_func, strategy_params) in strategies.items():
                current_run += 1
                
                if progress_callback:
                    progress_callback(f"[{current_run}/{total_runs}] Testing {symbol} with {strategy_name}...")
                
                try:
                    # Apply strategy
                    df_strategy = strategy_func(df.copy(), **strategy_params)
                    
                    # Run backtest
                    trades_df, equity, daily_returns = backtest_strategy(
                        df_strategy,
                        initial_capital=params['initial_capital'],
                        risk_per_trade_pct=params['risk_per_trade'],
                        stop_loss_pct=params['stop_loss_pct'],
                        take_profit_pct=params['take_profit_pct'],
                        min_holding_bars=params.get('min_holding_bars', 5),
                        cooldown_bars=params.get('cooldown_bars', 2),
                        max_trades_per_day=params.get('max_trades_per_day', 5)
                    )
                    
                    # Calculate metrics
                    if not trades_df.empty:
                        metrics = compute_metrics(trades_df, initial_capital=params['initial_capital'])
                        
                        # Calculate additional metrics not in compute_metrics
                        final_capital = equity[-1]
                        total_return = ((final_capital - params['initial_capital']) / params['initial_capital']) * 100
                        
                        # Calculate Sharpe from daily returns
                        sharpe = 0
                        if len(daily_returns) > 0 and daily_returns.std() > 0:
                            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
                        
                        # Safely get wins and losses
                        wins = trades_df[trades_df['pnl'] > 0]
                        losses = trades_df[trades_df['pnl'] <= 0]
                        
                        avg_win_pct = (wins['pnl_pct'].mean()) if not wins.empty else 0
                        avg_loss_pct = abs(losses['pnl_pct'].mean()) if not losses.empty else 0
                        
                        # Calculate expectancy
                        win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0
                        expectancy = (win_rate * avg_win_pct) - ((1 - win_rate) * avg_loss_pct)
                        
                        # Compile result
                        results.append({
                            'Symbol': symbol,
                            'Strategy': strategy_name,
                            'Timeframe': timeframe,
                            'Final Capital': final_capital,
                            'Total Return %': total_return,
                            'Total Trades': metrics.get('Trades', len(trades_df)),
                            'Win Rate %': metrics.get('Win Rate %', win_rate * 100),
                            'Profit Factor': metrics.get('Profit Factor', 0),
                            'Sharpe Ratio': round(sharpe, 2),
                            'Max DD %': abs(metrics.get('Max Drawdown %', 0)),
                            'Avg Win %': round(avg_win_pct, 2),
                            'Avg Loss %': round(avg_loss_pct, 2),
                            'Expectancy': round(expectancy, 2),
                            'Status': '✅ Success'
                        })
                    else:
                        results.append({
                            'Symbol': symbol,
                            'Strategy': strategy_name,
                            'Timeframe': timeframe,
                            'Status': '⚠️ No Trades'
                        })
                
                except Exception as e:
                    import traceback
                    error_detail = str(e)
                    # Get first line of traceback for more context
                    tb_lines = traceback.format_exc().split('\n')
                    error_location = next((line for line in tb_lines if 'File' in line and '.py' in line), '')
                    
                    results.append({
                        'Symbol': symbol,
                        'Strategy': strategy_name,
                        'Timeframe': timeframe,
                        'Status': f'❌ Error: {error_detail}'
                    })
                    
                    if progress_callback:
                        progress_callback(f"❌ {symbol} + {strategy_name} failed: {error_detail}")
        
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Failed to load {symbol}: {e}")
    
    return pd.DataFrame(results)


# ==============================================================================
# UI
# ==============================================================================
st.title("🔬 Batch Stock Analyzer")
st.markdown("**Systematically test strategies across multiple symbols**")

# ==============================================================================
# SIDEBAR - Configuration
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Symbol Selection
    st.subheader("📊 Symbols")
    
    # Load Nifty 100 list
    LIST_FILE = Path("data/Nifty100list.csv")
    symbol_list = []
    
    if LIST_FILE.exists():
        try:
            df_list = pd.read_csv(LIST_FILE)
            if "Symbol" in df_list.columns:
                symbol_list = df_list["Symbol"].dropna().astype(str).unique().tolist()
            elif "symbol" in df_list.columns:
                symbol_list = df_list["symbol"].dropna().astype(str).unique().tolist()
            else:
                symbol_list = df_list.iloc[:, 0].dropna().astype(str).unique().tolist()
        except:
            pass
    
    symbol_mode = st.radio("Selection Mode", ["Quick Select", "Full List", "Custom"])
    
    if symbol_mode == "Quick Select":
        quick_options = {
            "Top 10 Nifty": symbol_list[:10] if symbol_list else [],
            "Top 20 Nifty": symbol_list[:20] if symbol_list else [],
            "Top 50 Nifty": symbol_list[:50] if symbol_list else [],
            "All Nifty 100": symbol_list if symbol_list else []
        }
        quick_choice = st.selectbox("Quick Presets", list(quick_options.keys()))
        selected_symbols = quick_options[quick_choice]
    
    elif symbol_mode == "Full List":
        if symbol_list:
            selected_symbols = st.multiselect(
                "Select Symbols", 
                symbol_list, 
                default=symbol_list[:5]
            )
        else:
            st.warning("No Nifty100list.csv found")
            selected_symbols = []
    
    else:  # Custom
        custom_input = st.text_area(
            "Enter symbols (one per line)",
            value="RELIANCE\nTCS\nHDFCBANK"
        )
        selected_symbols = [s.strip().upper() for s in custom_input.split('\n') if s.strip()]
    
    st.info(f"Selected: {len(selected_symbols)} symbols")
    
    st.divider()
    
    # Strategy Selection
    st.subheader("🎯 Strategies")
    
    strategy_options = {
        "Simple Momentum": (simple_momentum_strategy, {'fast_ema': 5, 'slow_ema': 20, 'rsi_period': 14}),
        "Mean Reversion": (mean_reversion_basic, {'bb_period': 20, 'rsi_period': 14}),
        "Opening Range Breakout": (opening_range_breakout, {'or_minutes': 15}),
        "VWAP Mean Reversion": (vwap_mean_reversion, {'deviation_threshold': 1.0})
    }
    
    selected_strategies = st.multiselect(
        "Select Strategies",
        list(strategy_options.keys()),
        default=["Simple Momentum", "Mean Reversion"]
    )
    
    strategies_to_run = {name: strategy_options[name] for name in selected_strategies}
    
    st.divider()
    st.subheader("📅 Timeframe & Risk")
    
    timeframe = st.selectbox(
        "Timeframe",
        ["5minute", "15minute", "30minute", "60minute"],
        index=1
    )
    
    initial_capital = st.number_input("Initial Capital", value=100000, step=10000)
    risk_per_trade = st.slider("Risk per Trade %", 0.5, 4.0, 1.0, 0.1)
    stop_loss_pct = st.slider("Stop Loss %", 0.3, 2.0, 0.5, 0.1)
    take_profit_pct = st.slider("Take Profit %", 0.5, 5.0, 1.5, 0.1)
    
    st.divider()
    st.subheader("🎯 Trade Frequency Control")
    st.caption("Reduce overtrading and whipsaw")
    
    min_holding_bars = st.slider(
        "Min Holding Bars", 
        1, 20, 5, 1,
        help="Minimum bars to hold position (prevents whipsaw). 5 bars = 75 min for 15min data"
    )
    
    cooldown_bars = st.slider(
        "Cooldown Between Trades",
        0, 10, 2, 1,
        help="Bars to wait after exit before new entry (reduces overtrading)"
    )
    
    max_trades_per_day = st.number_input(
        "Max Trades Per Day",
        1, 20, 5, 1,
        help="Limit daily trades (0 = unlimited)"
    )
    
    # Info box
    st.info(f"""
    **💡 Expected Trade Reduction:**
    - Without filters: ~400-600 trades per symbol
    - With current settings: ~50-150 trades per symbol
    
    **Recommended for 15min data:**
    - Min Holding: 3-5 bars (45-75 min)
    - Cooldown: 2-3 bars (30-45 min)
    - Max/Day: 3-5 trades
    """)
    
    backtest_params = {
        'initial_capital': initial_capital,
        'risk_per_trade': risk_per_trade,
        'stop_loss_pct': stop_loss_pct,
        'take_profit_pct': take_profit_pct,
        'min_holding_bars': min_holding_bars,
        'cooldown_bars': cooldown_bars,
        'max_trades_per_day': max_trades_per_day
    }


# ==============================================================================
# MAIN AREA
# ==============================================================================

# Validation
if not selected_symbols:
    st.warning("⚠️ Please select at least one symbol from the sidebar")
    st.stop()

if not selected_strategies:
    st.warning("⚠️ Please select at least one strategy from the sidebar")
    st.stop()

# Display configuration
st.subheader("📋 Batch Configuration")
col1, col2, col3 = st.columns(3)
col1.metric("Symbols", len(selected_symbols))
col2.metric("Strategies", len(selected_strategies))
col3.metric("Total Runs", len(selected_symbols) * len(selected_strategies))

with st.expander("📝 Show Configuration Details"):
    st.write("**Selected Symbols:**", ", ".join(selected_symbols[:10]) + ("..." if len(selected_symbols) > 10 else ""))
    st.write("**Selected Strategies:**", ", ".join(selected_strategies))
    st.write("**Timeframe:**", timeframe)
    st.write("**Risk Parameters:**", backtest_params)

st.divider()

# Run button
run_batch = st.button("▶️ Run Batch Analysis", type="primary", use_container_width=True)

# Debug mode toggle
debug_mode = st.checkbox("🔍 Debug Mode (show detailed error information)", value=False)

# Progress area
progress_bar = st.progress(0)
status_text = st.empty()

# Results container
results_container = st.container()

# ==============================================================================
# RUN BATCH ANALYSIS
# ==============================================================================
if run_batch:
    with st.spinner("Running batch analysis..."):
        
        # Progress callback
        def update_progress(message):
            status_text.text(message)
        
        # Run batch
        results_df = run_batch_analysis(
            symbols=selected_symbols,
            strategies=strategies_to_run,
            timeframe=timeframe,
            params=backtest_params,
            progress_callback=update_progress
        )
        
        progress_bar.progress(100)
        status_text.success(f"✅ Batch analysis complete! Processed {len(results_df)} runs")
        
        # Store in session state
        st.session_state['batch_results'] = results_df
        st.session_state['batch_timestamp'] = datetime.now()

# ==============================================================================
# DISPLAY RESULTS
# ==============================================================================
if 'batch_results' in st.session_state:
    results_df = st.session_state['batch_results']
    timestamp = st.session_state.get('batch_timestamp', datetime.now())
    
    with results_container:
        st.divider()
        st.header("📊 Batch Results")
        st.caption(f"Analysis completed at: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Filter successful runs
        successful_runs = results_df[results_df['Status'] == '✅ Success'].copy()
        failed_runs = results_df[results_df['Status'] != '✅ Success'].copy()
        
        # Show failed runs in debug mode
        if not failed_runs.empty and debug_mode:
            with st.expander(f"🔍 Debug: Failed Runs ({len(failed_runs)})", expanded=True):
                st.error("The following symbol-strategy combinations failed:")
                st.dataframe(failed_runs[['Symbol', 'Strategy', 'Timeframe', 'Status']], use_container_width=True)
                
                st.info("**Common Solutions:**")
                st.markdown("""
                1. **Check data exists**: Use Data Inspector (Page 8) to verify data files
                2. **Run Data Organizer**: Go to Page 7 and resample to target timeframe  
                3. **Check column names**: Ensure OHLCV columns are Title Case
                4. **Strategy requirements**: Some strategies need specific data (e.g., ORB needs intraday data with time info)
                """)
        
        if successful_runs.empty:
            st.error("❌ No successful runs. Check that data exists for selected symbols at the chosen timeframe.")
            st.dataframe(results_df)
            
            st.info("**Troubleshooting Steps:**")
            st.markdown("""
            1. Go to **Page 8: Data Inspector**
            2. Check if `data/derived/{SYMBOL}/15minute/` exists for your symbols
            3. If not, go to **Page 2** to download raw data
            4. Then go to **Page 7** to create derived data
            """)
            st.stop()
        
        # Summary metrics
        st.subheader("📈 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Successful Runs", len(successful_runs))
        col2.metric("Avg Return", f"{successful_runs['Total Return %'].mean():.2f}%")
        col3.metric("Avg Win Rate", f"{successful_runs['Win Rate %'].mean():.1f}%")
        col4.metric("Avg Sharpe", f"{successful_runs['Sharpe Ratio'].mean():.2f}")
        
        # Top Performers
        st.subheader("🏆 Top Performers (by Total Return)")
        top_n = st.slider("Show top N", 5, 20, 10)
        
        top_performers = successful_runs.nlargest(top_n, 'Total Return %')
        
        # Add visual indicators without matplotlib
        top_performers_display = top_performers.copy()
        top_performers_display['📊'] = top_performers_display['Total Return %'].apply(
            lambda x: '🟢' if x > 10 else ('🟡' if x > 0 else '🔴')
        )
        
        st.dataframe(
            top_performers_display[[
                '📊', 'Symbol', 'Strategy', 'Total Return %', 'Win Rate %', 
                'Profit Factor', 'Sharpe Ratio', 'Total Trades'
            ]],
            use_container_width=True,
            column_config={
                '📊': st.column_config.TextColumn('', width='small'),
                'Total Return %': st.column_config.NumberColumn('Total Return %', format='%.2f%%'),
                'Win Rate %': st.column_config.NumberColumn('Win Rate %', format='%.1f%%'),
                'Profit Factor': st.column_config.NumberColumn('Profit Factor', format='%.2f'),
                'Sharpe Ratio': st.column_config.NumberColumn('Sharpe Ratio', format='%.2f'),
            }
        )
        
        # Strategy Comparison
        st.subheader("📊 Strategy Performance Comparison")
        
        strategy_summary = successful_runs.groupby('Strategy').agg({
            'Total Return %': 'mean',
            'Win Rate %': 'mean',
            'Profit Factor': 'mean',
            'Sharpe Ratio': 'mean',
            'Total Trades': 'sum'
        }).round(2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(strategy_summary, use_container_width=True)
        
        with col2:
            # Bar chart of strategy returns
            fig = go.Figure(data=[
                go.Bar(
                    x=strategy_summary.index,
                    y=strategy_summary['Total Return %'],
                    marker_color=['green' if x > 0 else 'red' for x in strategy_summary['Total Return %']]
                )
            ])
            fig.update_layout(
                title="Average Return by Strategy",
                xaxis_title="Strategy",
                yaxis_title="Avg Return %",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Symbol Performance
        st.subheader("📈 Symbol Performance Breakdown")
        
        symbol_summary = successful_runs.groupby('Symbol').agg({
            'Total Return %': 'mean',
            'Win Rate %': 'mean',
            'Total Trades': 'sum'
        }).round(2).sort_values('Total Return %', ascending=False)
        
        # Add visual indicators
        symbol_summary_display = symbol_summary.reset_index()
        symbol_summary_display['📊'] = symbol_summary_display['Total Return %'].apply(
            lambda x: '🟢' if x > 10 else ('🟡' if x > 0 else '🔴')
        )
        
        st.dataframe(
            symbol_summary_display[['📊', 'Symbol', 'Total Return %', 'Win Rate %', 'Total Trades']],
            use_container_width=True,
            column_config={
                '📊': st.column_config.TextColumn('', width='small'),
                'Total Return %': st.column_config.NumberColumn('Avg Return %', format='%.2f%%'),
                'Win Rate %': st.column_config.NumberColumn('Avg Win Rate %', format='%.1f%%'),
            }
        )
        
        # Full Results Table
        st.subheader("📋 Complete Results")
        
        # Add filters
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_symbol = st.multiselect("Filter by Symbol", successful_runs['Symbol'].unique())
        with col2:
            filter_strategy = st.multiselect("Filter by Strategy", successful_runs['Strategy'].unique())
        with col3:
            min_return = st.number_input("Min Return %", value=-100.0, step=5.0)
        
        # Apply filters
        filtered_df = successful_runs.copy()
        if filter_symbol:
            filtered_df = filtered_df[filtered_df['Symbol'].isin(filter_symbol)]
        if filter_strategy:
            filtered_df = filtered_df[filtered_df['Strategy'].isin(filter_strategy)]
        filtered_df = filtered_df[filtered_df['Total Return %'] >= min_return]
        
        st.dataframe(
            filtered_df.sort_values('Total Return %', ascending=False),
            use_container_width=True
        )
        
        # Export Options
        st.subheader("💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV Export
            csv = successful_runs.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"batch_analysis_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Summary report
            summary_text = f"""
BATCH ANALYSIS SUMMARY
=====================
Timestamp: {timestamp}
Symbols Analyzed: {len(selected_symbols)}
Strategies Tested: {len(selected_strategies)}
Successful Runs: {len(successful_runs)}

OVERALL PERFORMANCE
------------------
Average Return: {successful_runs['Total Return %'].mean():.2f}%
Average Win Rate: {successful_runs['Win Rate %'].mean():.1f}%
Average Sharpe: {successful_runs['Sharpe Ratio'].mean():.2f}
Average Profit Factor: {successful_runs['Profit Factor'].mean():.2f}

TOP 5 PERFORMERS
----------------
{top_performers[['Symbol', 'Strategy', 'Total Return %']].head().to_string(index=False)}

STRATEGY RANKINGS
-----------------
{strategy_summary['Total Return %'].to_string()}
"""
            st.download_button(
                label="📄 Download Summary Report",
                data=summary_text,
                file_name=f"batch_summary_{timestamp.strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

# ==============================================================================
# DOCUMENTATION
# ==============================================================================
with st.expander("📚 How to Use Batch Analyzer", expanded=False):
    st.markdown("""
    ## 🎯 Purpose
    Test multiple trading strategies across many symbols simultaneously to identify:
    - Which strategies work best for different stocks
    - Which stocks are most profitable with specific strategies
    - Overall strategy robustness across market conditions
    
    ## 📋 Workflow
    
    ### 1. Select Symbols
    - **Quick Select**: Use presets (Top 10, Top 20, etc.)
    - **Full List**: Pick individual symbols from Nifty 100
    - **Custom**: Enter your own symbol list
    
    ### 2. Choose Strategies
    Select one or more strategies to test:
    - **Simple Momentum**: EMA crossover with RSI filter
    - **Mean Reversion**: Bollinger Bands + RSI
    - **Opening Range Breakout**: First 15-min range breakout
    - **VWAP Mean Reversion**: Institutional order flow
    
    ### 3. Configure Risk
    - **Initial Capital**: Starting balance
    - **Risk per Trade**: % of capital to risk
    - **Stop Loss**: Maximum loss per trade
    - **Take Profit**: Target profit per trade
    
    ### 4. Run Analysis
    - Click "Run Batch Analysis"
    - Monitor progress
    - Review results
    
    ## 📊 Understanding Results
    
    ### Top Performers Table
    - Shows best symbol-strategy combinations
    - Sorted by total return
    - Highlights win rate and profit factor
    
    ### Strategy Comparison
    - Average performance across all symbols
    - Identifies most consistent strategies
    - Use this to select strategies for live trading
    
    ### Symbol Breakdown
    - Shows which stocks are most profitable
    - Helps identify best candidates for trading
    
    ## 💡 Best Practices
    
    1. **Start Small**: Test 5-10 symbols first
    2. **Multiple Timeframes**: Run analysis on 15min and 60min
    3. **Look for Consistency**: Prefer strategies with 55%+ win rate AND >1.5 profit factor
    4. **Avoid Overfitting**: Don't cherry-pick best results; test forward
    5. **Export Results**: Keep historical records to track strategy evolution
    
    ## ⚠️ Requirements
    
    - Data must exist in `data/derived/{SYMBOL}/{TIMEFRAME}/`
    - Run Page 7 (Data Organizer) first to create derived data
    - Ensure at least 1000+ candles for reliable results
    
    ## 🚀 Next Steps
    
    After finding winning combinations:
    1. Deep-dive individual symbols on Page 6 (Strategy Lab)
    2. Test with different parameters (grid search)
    3. Forward-test on recent data
    4. Deploy best strategies to live trading (Page 5)
    """)