# pages/6_New_Strategy_Lab.py
"""
Professional Strategy Testing Lab
- Mean Reversion, ORB, VWAP, ML strategies
- Clean backtesting engine
- Walk-forward optimization
- ML model training interface
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import strategies
try:
    from core.strategies.mean_reversion import (
        mean_reversion_basic, mean_reversion_advanced, STRATEGY_INFO as MR_INFO
    )
    from core.strategies.opening_range import (
        opening_range_breakout, opening_range_advanced, STRATEGY_INFO as ORB_INFO
    )
    from core.strategies.vwap_strategy import (
        vwap_mean_reversion, vwap_advanced, STRATEGY_INFO as VWAP_INFO
    )
    from core.strategies.simple_momentum import (
        simple_momentum_strategy, simple_momentum_with_atr, STRATEGY_INFO as MOM_INFO
    )
except ImportError as e:
    st.error(f"Strategy modules not found. Error: {e}")
    st.info("Make sure you've created the strategy files in core/strategies/")
    st.stop()

# Import ML modules (optional)
try:
    from core.ml.features import engineer_all_features, get_feature_columns
    from core.ml.trainer import TradingMLTrainer, quick_train_pipeline
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from core.metrics import compute_metrics

st.set_page_config(layout="wide", page_title="Strategy Lab")

# ==============================================================================
# HELPER: Clean Backtester
# ==============================================================================
def backtest_strategy(df: pd.DataFrame,
                     initial_capital: float = 100000,
                     risk_per_trade_pct: float = 1.0,
                     stop_loss_pct: float = 0.5,
                     take_profit_pct: float = 1.5,
                     max_holding_bars: int = 100,
                     min_holding_bars: int = 3) -> tuple:
    """
    Simple, clean backtester with minimum holding time
    
    Key Features:
    - Enforces minimum holding period (prevents whipsaw)
    - Exit priority: SL → TP → TIME → SIGNAL
    - Signal exits only allowed after min holding time
    
    Returns:
        (trades_df, equity_curve, daily_returns)
    """
    balance = initial_capital
    position = None  # {'side': 'LONG'/'SHORT', 'entry_price': float, 'entry_bar': int}
    trades = []
    equity = []
    
    for i in range(len(df)):
        bar = df.iloc[i]
        
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
            
            # 4. Signal exit - ONLY after minimum holding time AND if not in drawdown
            #elif bars_held >= min_holding_bars and bar.get('Exit_Signal', 0) != 0:
            #    # Only exit on signal if we're near breakeven or profitable
            #    if current_pnl_pct > -0.3:  # Allow small drawdown
            #        exit_reason = 'SIGNAL'
            
            # Execute exit
            if exit_reason:
                exit_price = bar['Close']
                
                if side == 'LONG':
                    pnl = (exit_price - entry_price) / entry_price * balance * (risk_per_trade_pct / 100)
                else:
                    pnl = (entry_price - exit_price) / entry_price * balance * (risk_per_trade_pct / 100)
                
                balance += pnl
                
                trades.append({
                    'entry_time': df.index[entry_bar],
                    'exit_time': df.index[i],
                    'side': side,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': current_pnl_pct,
                    'bars_held': i - entry_bar,
                    'exit_reason': exit_reason
                })
                
                position = None
        
        # Check for entry if not in position
        if position is None and bar.get('Signal', 0) != 0:
            signal = bar['Signal']
            
            position = {
                'side': 'LONG' if signal == 1 else 'SHORT',
                'entry_price': bar['Close'],
                'entry_bar': i
            }
        
        equity.append(balance)
    
    trades_df = pd.DataFrame(trades)
    
    # Calculate daily returns for Sharpe ratio
    equity_series = pd.Series(equity, index=df.index)
    daily_returns = equity_series.resample('D').last().pct_change().dropna()
    
    return trades_df, equity, daily_returns


# ==============================================================================
# UI: SIDEBAR
# ==============================================================================
st.title("🧪 Strategy Lab - Professional Backtester")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Strategy Selection
    strategy_type = st.selectbox(
        "Strategy Type",
        ["Simple Momentum (Recommended)", "Mean Reversion", "Opening Range Breakout", "VWAP Mean Reversion", "ML Strategy (Advanced)"]
    )
    
    st.divider()
    
    # Risk Settings
    st.subheader("💰 Risk Management")
    initial_capital = st.number_input("Initial Capital", value=100000, step=10000)
    risk_per_trade = st.slider("Risk per Trade %", 0.5, 4.0, 1.0, 0.1)
    stop_loss_pct = st.slider("Stop Loss %", 0.3, 2.0, 0.5, 0.1)
    take_profit_pct = st.slider("Take Profit %", 0.5, 5.0, 1.5, 0.1)
    
    st.divider()
    
    # Strategy-specific parameters
    st.subheader("🎯 Strategy Parameters")
    
    if strategy_type == "Simple Momentum (Recommended)":
        fast_ema = st.slider("Fast EMA", 3, 10, 5)
        slow_ema = st.slider("Slow EMA", 15, 30, 20)
        rsi_period = st.slider("RSI Period", 10, 20, 14)
        use_advanced = st.checkbox("Use ATR Filters", value=True)
        
    elif strategy_type == "Mean Reversion":
        bb_period = st.slider("BB Period", 10, 30, 20)
        rsi_period = st.slider("RSI Period", 10, 20, 14)
        rsi_oversold = st.slider("RSI Oversold", 20, 35, 30)
        rsi_overbought = st.slider("RSI Overbought", 65, 80, 70)
        use_advanced = st.checkbox("Use Advanced Filters", value=True)
        
    elif strategy_type == "Opening Range Breakout":
        or_minutes = st.slider("Opening Range Minutes", 5, 30, 15, 5)
        buffer_pct = st.slider("Breakout Buffer %", 0.1, 1.0, 0.3, 0.1)
        use_advanced = st.checkbox("Use Advanced Filters", value=True)
        
    elif strategy_type == "VWAP Mean Reversion":
        deviation_threshold = st.slider("Deviation Threshold %", 0.5, 2.0, 1.0, 0.1)
        use_bands = st.checkbox("Use VWAP Bands", value=True)
        use_advanced = st.checkbox("Use Advanced Filters", value=True)
    
    st.divider()
    run_backtest = st.button("▶️ Run Backtest", type="primary", use_container_width=True)


# ==============================================================================
# DATA LOADING
# ==============================================================================
st.subheader("📁 Data Selection")

DATA_DIR = Path("data/derived")
if not DATA_DIR.exists():
    DATA_DIR = Path("data/processed")

files = sorted(DATA_DIR.rglob("*.parquet"))

if not files:
    st.error("No data files found. Please run Page 2 (Data Fetcher) first.")
    st.stop()

selected_file = st.selectbox("Select Data File", files, format_func=lambda x: x.name)

# Load data
try:
    df = pd.read_parquet(selected_file)
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
    
    # Standardize column names
    df.columns = [c.title() if c.lower() in ['open', 'high', 'low', 'close', 'volume'] else c 
                  for c in df.columns]
    
    st.success(f"✅ Loaded {len(df):,} candles from {df.index[0].date()} to {df.index[-1].date()}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Bars", f"{len(df):,}")
    col2.metric("Date Range", f"{(df.index[-1] - df.index[0]).days} days")
    col3.metric("Avg Volume", f"{df['Volume'].mean():.0f}" if 'Volume' in df.columns else "N/A")
    col4.metric("Price Range", f"₹{df['Low'].min():.1f} - ₹{df['High'].max():.1f}")
    
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


# ==============================================================================
# RUN BACKTEST
# ==============================================================================
if run_backtest:
    with st.spinner(f"Running {strategy_type} backtest..."):
        
        # Apply selected strategy
        if strategy_type == "Simple Momentum (Recommended)":
            if use_advanced:
                df_strategy = simple_momentum_with_atr(
                    df.copy(),
                    fast_ema=fast_ema,
                    slow_ema=slow_ema,
                    rsi_period=rsi_period
                )
            else:
                df_strategy = simple_momentum_strategy(
                    df.copy(),
                    fast_ema=fast_ema,
                    slow_ema=slow_ema,
                    rsi_period=rsi_period
                )
            strategy_info = MOM_INFO
        
        elif strategy_type == "Mean Reversion":
            if use_advanced:
                df_strategy = mean_reversion_advanced(
                    df.copy(), 
                    bb_period=bb_period,
                    rsi_period=rsi_period
                )
            else:
                df_strategy = mean_reversion_basic(
                    df.copy(),
                    bb_period=bb_period,
                    rsi_period=rsi_period,
                    rsi_oversold=rsi_oversold,
                    rsi_overbought=rsi_overbought
                )
            strategy_info = MR_INFO
        
        elif strategy_type == "Opening Range Breakout":
            if use_advanced:
                df_strategy = opening_range_advanced(
                    df.copy(),
                    or_minutes=or_minutes,
                    volume_confirmation=True,
                    gap_filter=True
                )
            else:
                df_strategy = opening_range_breakout(
                    df.copy(),
                    or_minutes=or_minutes,
                    buffer_pct=buffer_pct / 100
                )
            strategy_info = ORB_INFO
        
        elif strategy_type == "VWAP Mean Reversion":
            if use_advanced:
                df_strategy = vwap_advanced(
                    df.copy(),
                    deviation_threshold=deviation_threshold,
                    use_bands=use_bands
                )
            else:
                df_strategy = vwap_mean_reversion(
                    df.copy(),
                    deviation_threshold=deviation_threshold
                )
            strategy_info = VWAP_INFO
        
        else:  # ML Strategy
            st.warning("ML Strategy requires trained model. See ML Training section below.")
            st.stop()
        
        # Run backtest
        trades_df, equity_curve, daily_returns = backtest_strategy(
            df_strategy,
            initial_capital=initial_capital,
            risk_per_trade_pct=risk_per_trade,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            min_holding_bars=3  # Minimum 3 bars (45 minutes for 15min data)
        )
        
        # Calculate metrics
        if not trades_df.empty:
            metrics = compute_metrics(trades_df, initial_capital=initial_capital)
            
            # Add Sharpe ratio
            if len(daily_returns) > 0:
                sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
                metrics['Sharpe Ratio'] = round(sharpe, 2)
        else:
            metrics = {}
            st.error("❌ No trades generated! Strategy produced zero signals.")
            st.info("Try adjusting parameters or check if data timeframe matches strategy requirements.")
            st.stop()
    
    # ==============================================================================
    # RESULTS DISPLAY
    # ==============================================================================
    st.divider()
    st.header("📊 Backtest Results")
    
    # Strategy info card
    with st.expander("ℹ️ Strategy Information", expanded=False):
        st.markdown(f"""
        **{strategy_info['name']}**
        
        {strategy_info['description']}
        
        - **Best Timeframe**: {strategy_info['best_timeframe']}
        - **Best Markets**: {', '.join(strategy_info['best_markets'])}
        - **Expected Win Rate**: {strategy_info['expected_win_rate']}
        - **Expected Profit Factor**: {strategy_info['expected_profit_factor']}
        - **Max Holding Time**: {strategy_info['max_holding_time']}
        """)
    
    # Performance Metrics
    final_capital = equity_curve[-1] if equity_curve else initial_capital
    total_return = ((final_capital - initial_capital) / initial_capital) * 100
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Final Capital", f"₹{final_capital:,.0f}", f"{total_return:+.2f}%")
    col2.metric("Total Trades", metrics.get('Trades', 0))
    col3.metric("Win Rate", f"{metrics.get('Win Rate %', 0):.1f}%")
    col4.metric("Profit Factor", f"{metrics.get('Profit Factor', 0):.2f}")
    col5.metric("Sharpe Ratio", f"{metrics.get('Sharpe Ratio', 0):.2f}")
    
    # Advanced Metrics
    st.subheader("📈 Advanced Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Max Drawdown", f"{metrics.get('Max Drawdown %', 0):.2f}%")
    col2.metric("Avg Win/Loss", f"{metrics.get('Avg Win/Loss', 0):.2f}")
    col3.metric("Total PnL", f"₹{metrics.get('Total PnL', 0):,.0f}")
    
    # Holding time analysis
    if not trades_df.empty:
        avg_holding = trades_df['bars_held'].mean()
        col4.metric("Avg Holding (bars)", f"{avg_holding:.1f}")
    
    # Equity Curve Chart
    st.subheader("💹 Equity Curve")
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=list(range(len(equity_curve))),
        y=equity_curve,
        mode='lines',
        name='Equity',
        line=dict(color='#00ff88', width=2)
    ))
    fig_equity.add_hline(y=initial_capital, line_dash="dash", line_color="gray", annotation_text="Initial Capital")
    fig_equity.update_layout(
        title="Portfolio Equity Over Time",
        xaxis_title="Trade Number",
        yaxis_title="Capital (₹)",
        height=400,
        template="plotly_dark"
    )
    st.plotly_chart(fig_equity, use_container_width=True)
    
    # Trade Distribution
    st.subheader("📊 Trade Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # PnL distribution
        fig_pnl = go.Figure(data=[go.Histogram(
            x=trades_df['pnl'],
            nbinsx=30,
            marker_color='#00ff88'
        )])
        fig_pnl.update_layout(
            title="P&L Distribution",
            xaxis_title="Profit/Loss (₹)",
            yaxis_title="Frequency",
            height=300,
            template="plotly_dark"
        )
        st.plotly_chart(fig_pnl, use_container_width=True)
    
    with col2:
        # Exit reasons pie chart
        exit_counts = trades_df['exit_reason'].value_counts()
        fig_exits = go.Figure(data=[go.Pie(
            labels=exit_counts.index,
            values=exit_counts.values,
            hole=0.3
        )])
        fig_exits.update_layout(
            title="Exit Reasons",
            height=300,
            template="plotly_dark"
        )
        st.plotly_chart(fig_exits, use_container_width=True)
    
    # Trade Log
    with st.expander("📋 Trade Log (Last 50 trades)", expanded=False):
        display_df = trades_df.tail(50).copy()
        display_df['entry_time'] = pd.to_datetime(display_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
        display_df['exit_time'] = pd.to_datetime(display_df['exit_time']).dt.strftime('%Y-%m-%d %H:%M')
        
        # Style the dataframe
        def color_pnl(val):
            color = 'green' if val > 0 else 'red'
            return f'color: {color}'
        
        styled_df = display_df.style.applymap(color_pnl, subset=['pnl', 'pnl_pct'])
        st.dataframe(styled_df, use_container_width=True, height=400)


# ==============================================================================
# ML TRAINING SECTION
# ==============================================================================
st.divider()
st.header("🤖 Machine Learning Strategy (Advanced)")

if not ML_AVAILABLE:
    st.warning("⚠️ ML libraries not installed. Install with: `pip install xgboost scikit-learn`")
else:
    with st.expander("🧠 Train ML Model", expanded=False):
        st.markdown("""
        Train a machine learning model to predict future price movements.
        
        **Process:**
        1. Feature engineering (50+ technical indicators)
        2. Train/test split (80/20, time-series aware)
        3. Model training (XGBoost or Random Forest)
        4. Walk-forward validation
        5. Feature importance analysis
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            ml_model_type = st.selectbox("Model Type", ["xgboost", "random_forest"])
            ml_task = st.selectbox("Task", ["classification", "regression"])
        
        with col2:
            target_horizon = st.slider("Prediction Horizon (bars)", 5, 30, 15)
            test_size = st.slider("Test Size %", 10, 30, 20) / 100
        
        if st.button("🚀 Train ML Model"):
            with st.spinner("Engineering features..."):
                # Feature engineering
                df_ml = engineer_all_features(
                    df.copy(), 
                    for_training=True, 
                    target_horizon=target_horizon
                )
                
                feature_cols = get_feature_columns()
                
                # Filter to available columns
                feature_cols = [col for col in feature_cols if col in df_ml.columns]
                
                st.success(f"✅ Created {len(feature_cols)} features from {len(df_ml)} samples")
            
            with st.spinner("Training model..."):
                # Train model
                trainer = TradingMLTrainer(
                    model_type=ml_model_type,
                    task=ml_task,
                    test_size=test_size
                )
                
                X_train, X_test, y_train, y_test = trainer.prepare_data(
                    df_ml, 
                    feature_cols
                )
                
                trainer.train(X_train, y_train)
                metrics_ml = trainer.evaluate(X_test, y_test)
                
                # Feature importance
                importance_df = trainer.get_feature_importance(top_n=15)
                
                # Save model
                model_path = trainer.save_model()
            
            st.success("✅ Model training complete!")
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            if ml_task == 'classification':
                col1.metric("Test Accuracy", f"{metrics_ml['accuracy']:.2%}")
                col2.metric("Precision", f"{metrics_ml['precision']:.2%}")
                col3.metric("F1 Score", f"{metrics_ml['f1_score']:.2%}")
            else:
                col1.metric("RMSE", f"{metrics_ml['rmse']:.6f}")
                col2.metric("MAE", f"{metrics_ml['mae']:.6f}")
                col3.metric("R²", f"{metrics_ml['r2']:.4f}")
            
            # Feature importance chart
            if not importance_df.empty:
                fig_importance = go.Figure(data=[
                    go.Bar(
                        x=importance_df['importance'].head(15),
                        y=importance_df['feature'].head(15),
                        orientation='h',
                        marker_color='#00ff88'
                    )
                ])
                fig_importance.update_layout(
                    title="Top 15 Most Important Features",
                    xaxis_title="Importance",
                    yaxis_title="Feature",
                    height=500,
                    template="plotly_dark"
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            
            st.info(f"💾 Model saved to: {model_path}")
            
            st.markdown("""
            ### Next Steps:
            1. Use saved model for live predictions (Page 5 - Live Trading)
            2. Perform walk-forward validation to test robustness
            3. Monitor performance on new data
            """)


# ==============================================================================
# DOCUMENTATION
# ==============================================================================
with st.expander("📚 Strategy Guide", expanded=False):
    st.markdown("""
    ## Strategy Selection Guide
    
    ### Mean Reversion (Best for beginners)
    ✅ **When to use:**
    - Market is ranging (not trending)
    - Low to normal volatility
    - Liquid stocks (high volume)
    
    ⚙️ **Key Parameters:**
    - BB Period: 15-25 (lower = more sensitive)
    - RSI Oversold: 25-35 (lower = more extreme)
    - Stop Loss: 0.5-0.8%
    - Take Profit: 1.2-2.0%
    
    ---
    
    ### Opening Range Breakout (Best for trending days)
    ✅ **When to use:**
    - Gap up/down days
    - High momentum stocks
    - First 2 hours of trading
    
    ⚙️ **Key Parameters:**
    - OR Minutes: 15 (standard), 5 (aggressive)
    - Buffer: 0.3% (reduces false breakouts)
    - Stop Loss: 0.5-1.0%
    - Take Profit: 1.5-3.0%
    
    ---
    
    ### VWAP Mean Reversion (Best for professionals)
    ✅ **When to use:**
    - Institutional trading hours (10 AM - 2 PM)
    - High volume periods
    - Large-cap stocks
    
    ⚙️ **Key Parameters:**
    - Deviation: 1.0-1.5% (distance from VWAP)
    - Stop Loss: 0.5-0.8%
    - Take Profit: 0.8-1.5%
    
    ---
    
    ### ML Strategy (Advanced)
    ✅ **When to use:**
    - After mastering basic strategies
    - Large datasets (50,000+ candles)
    - Multiple stocks (portfolio approach)
    
    ⚠️ **Warnings:**
    - Requires careful validation
    - Risk of overfitting
    - Needs regular retraining
    
    ---
    
    ## Risk Management Tips
    
    1. **Never risk more than 2% per trade**
    2. **Use stop losses religiously**
    3. **Limit daily loss to 5% of capital**
    4. **Start with paper trading for 1 month**
    5. **Track every trade in a journal**
    """)