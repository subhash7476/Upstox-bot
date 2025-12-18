# pages/5_Batch_Stock_Analyzer_Enhanced.py
"""
Enhanced Batch Stock Analyzer with Regime Awareness
---------------------------------------------------
Key Improvements:
1. Trade Attribution: WHY each trade happened
2. Regime Detection: Trend/Chop/Vol classification
3. Setup Classification: Quality scoring per entry
4. Expectation Modeling: E[R] per setup type
5. Strategy Decomposition: Separate entry/exit/filter logic
6. Decision Traceability: Full audit trail per trade
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import strategies
try:
    from core.strategies.mean_reversion import mean_reversion_basic
    from core.strategies.opening_range import opening_range_breakout
    from core.strategies.vwap_strategy import vwap_mean_reversion
    from core.strategies.simple_momentum import simple_momentum_strategy
except ImportError:
    st.error("Strategy modules not found. Please ensure core/strategies/ exists.")
    st.stop()

from core.metrics import compute_metrics

st.set_page_config(layout="wide", page_title="Enhanced Batch Analyzer")

# ==============================================================================
# REGIME DETECTION (Simple, Robust)
# ==============================================================================
@dataclass
class MarketRegime:
    """Market state snapshot at bar level"""
    trend_strength: float  # 0-1 (ADX-like)
    volatility_regime: str  # 'Low', 'Normal', 'High', 'Extreme'
    price_regime: str  # 'Trending', 'Choppy', 'Ranging'
    atr_percentile: float  # 0-100
    efficiency_ratio: float  # 0-1 (Price efficiency)
    
    def to_dict(self):
        return asdict(self)
    
    @property
    def is_tradeable_for_momentum(self) -> bool:
        """Should we allow momentum trades?"""
        return (
            self.trend_strength > 0.35 and  # Decent trend
            self.efficiency_ratio > 0.35 and  # Not too choppy
            self.volatility_regime != 'Extreme'  # Not in chaos
        )
    
    @property
    def is_tradeable_for_mean_reversion(self) -> bool:
        """Should we allow mean reversion trades?"""
        return (
            self.price_regime in ['Ranging', 'Choppy'] and
            self.volatility_regime in ['Normal', 'High'] and
            self.trend_strength < 0.5
        )


def compute_regime_indicators(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Add regime classification columns to DataFrame
    Simple, robust indicators that work
    """
    df = df.copy()
    
    # 1. ATR-based volatility regime
    if 'ATR' not in df.columns:
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift(1))
        low_close = abs(df['Low'] - df['Close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(14).mean()
    
    # ATR percentile (0-100)
    df['atr_percentile'] = df['ATR'].rolling(lookback * 5).apply(
        lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100, raw=False
    )
    
    # Volatility regime classification
    def classify_vol(pct):
        if pct < 25: return 'Low'
        elif pct < 70: return 'Normal'
        elif pct < 90: return 'High'
        else: return 'Extreme'
    
    df['vol_regime'] = df['atr_percentile'].apply(classify_vol)
    
    # 2. Efficiency Ratio (Kaufman's)
    # How far price moved vs how much it wiggled
    price_change = abs(df['Close'] - df['Close'].shift(lookback))
    path_length = abs(df['Close'].diff()).rolling(lookback).sum()
    df['efficiency_ratio'] = (price_change / path_length).clip(0, 1).fillna(0)
    
    # 3. Trend Strength (simplified ADX concept)
    # Use directional movement
    up_move = df['High'] - df['High'].shift(1)
    down_move = df['Low'].shift(1) - df['Low']
    
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
    
    # Smooth with EMA
    plus_di = plus_dm.ewm(span=14).mean() / df['ATR']
    minus_di = minus_dm.ewm(span=14).mean() / df['ATR']
    
    dx = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['trend_strength'] = dx.ewm(span=14).mean().clip(0, 1).fillna(0)
    
    # 4. Price Regime (Trending vs Ranging)
    # Based on efficiency + trend strength
    def classify_price(row):
        if row['trend_strength'] > 0.5 and row['efficiency_ratio'] > 0.4:
            return 'Trending'
        elif row['efficiency_ratio'] < 0.25:
            return 'Choppy'
        else:
            return 'Ranging'
    
    df['price_regime'] = df.apply(classify_price, axis=1)
    
    return df


def get_bar_regime(row) -> MarketRegime:
    """Extract regime data from a single bar"""
    return MarketRegime(
        trend_strength=row.get('trend_strength', 0),
        volatility_regime=row.get('vol_regime', 'Normal'),
        price_regime=row.get('price_regime', 'Ranging'),
        atr_percentile=row.get('atr_percentile', 50),
        efficiency_ratio=row.get('efficiency_ratio', 0.5)
    )


# ==============================================================================
# TRADE ATTRIBUTION
# ==============================================================================
@dataclass
class TradeContext:
    """
    Full context for WHY a trade was taken
    This is what's missing from current implementation
    """
    # Entry reason
    entry_signal: str  # "EMA_CROSS_BULLISH", "BB_OVERSOLD", etc.
    strategy_name: str
    
    # Market state at entry
    regime: MarketRegime
    
    # Setup quality
    setup_strength: float  # 0-1 score
    signal_confidence: float  # 0-1 (could be from ML later)
    
    # Technical context
    atr_multiple: float  # Entry price vs ATR
    distance_from_ema: float  # % distance from key moving average
    rsi_value: float
    volume_ratio: float  # Current vol vs avg vol
    
    # Risk context
    expected_rr: float  # Risk-reward at entry
    
    def to_dict(self):
        return {
            'entry_signal': self.entry_signal,
            'strategy': self.strategy_name,
            'price_regime': self.regime.price_regime,
            'vol_regime': self.regime.volatility_regime,
            'trend_strength': round(self.regime.trend_strength, 2),
            'efficiency_ratio': round(self.regime.efficiency_ratio, 2),
            'setup_strength': round(self.setup_strength, 2),
            'atr_multiple': round(self.atr_multiple, 2),
            'rsi': round(self.rsi_value, 1),
            'expected_rr': round(self.expected_rr, 2)
        }


# ==============================================================================
# ENHANCED BACKTESTER WITH ATTRIBUTION
# ==============================================================================
def backtest_with_attribution(
    df: pd.DataFrame,
    strategy_name: str,
    initial_capital: float = 100000,
    risk_per_trade_pct: float = 1.0,
    stop_loss_pct: float = 0.5,
    take_profit_pct: float = 1.5,
    max_holding_bars: int = 100,
    min_holding_bars: int = 3,
    enable_regime_filter: bool = True
) -> Tuple[pd.DataFrame, List, List]:
    """
    Enhanced backtester that tracks WHY each trade happened
    
    Returns:
        trades_df: DataFrame with all trades + context
        equity: List of equity curve values
        daily_returns: Daily return series
    """
    balance = initial_capital
    position = None
    trades = []
    equity = []
    
    # Ensure regime indicators exist
    if 'trend_strength' not in df.columns:
        df = compute_regime_indicators(df)
    
    # Ensure we have RSI for context
    if 'RSI' not in df.columns:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volume ratio for context
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    for i in range(len(df)):
        bar = df.iloc[i]
        
        # Exit logic
        if position is not None:
            side = position['side']
            entry_price = position['entry_price']
            entry_bar = position['entry_bar']
            bars_held = i - entry_bar
            
            # Calculate P&L
            if side == 'LONG':
                current_pnl_pct = (bar['Close'] - entry_price) / entry_price * 100
            else:
                current_pnl_pct = (entry_price - bar['Close']) / entry_price * 100
            
            # Exit conditions
            exit_reason = None
            
            if current_pnl_pct <= -stop_loss_pct:
                exit_reason = 'SL'
            elif current_pnl_pct >= take_profit_pct:
                exit_reason = 'TP'
            elif bars_held >= max_holding_bars:
                exit_reason = 'TIME'
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
                
                # Build trade record with FULL CONTEXT
                trade_record = {
                    'Entry': df.index[entry_bar],
                    'Exit': df.index[i],
                    'Side': side,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'qty': position['qty'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'bars_held': bars_held,
                    'exit_reason': exit_reason,
                    
                    # Add context from entry
                    **position['context'].to_dict()
                }
                
                trades.append(trade_record)
                position = None
        
        # Entry logic
        if position is None:
            signal = bar.get('Signal', 0)
            
            # Get current regime
            regime = get_bar_regime(bar)
            
            # REGIME FILTER (This is key!)
            if enable_regime_filter:
                # For momentum strategies
                if strategy_name in ['Simple Momentum', 'Opening Range Breakout']:
                    if not regime.is_tradeable_for_momentum:
                        signal = 0  # Block trade
                
                # For mean reversion
                elif strategy_name in ['Mean Reversion', 'VWAP Mean Reversion']:
                    if not regime.is_tradeable_for_mean_reversion:
                        signal = 0
            
            if signal == 1:  # LONG
                qty = int((balance * risk_per_trade_pct / 100) / (bar['Close'] * stop_loss_pct / 100))
                qty = max(1, qty)
                
                # Calculate setup quality
                setup_strength = calculate_setup_strength(bar, signal, regime)
                
                # Build context
                context = TradeContext(
                    entry_signal=f"LONG_{strategy_name.upper().replace(' ', '_')}",
                    strategy_name=strategy_name,
                    regime=regime,
                    setup_strength=setup_strength,
                    signal_confidence=0.7,  # Placeholder for future ML
                    atr_multiple=bar.get('ATR', bar['Close'] * 0.02) / bar['Close'],
                    distance_from_ema=0,  # Could add EMA distance
                    rsi_value=bar.get('RSI', 50),
                    volume_ratio=bar.get('volume_ratio', 1.0),
                    expected_rr=take_profit_pct / stop_loss_pct
                )
                
                position = {
                    'side': 'LONG',
                    'entry_price': bar['Close'],
                    'entry_bar': i,
                    'qty': qty,
                    'context': context
                }
            
            elif signal == -1:  # SHORT
                qty = int((balance * risk_per_trade_pct / 100) / (bar['Close'] * stop_loss_pct / 100))
                qty = max(1, qty)
                
                setup_strength = calculate_setup_strength(bar, signal, regime)
                
                context = TradeContext(
                    entry_signal=f"SHORT_{strategy_name.upper().replace(' ', '_')}",
                    strategy_name=strategy_name,
                    regime=regime,
                    setup_strength=setup_strength,
                    signal_confidence=0.7,
                    atr_multiple=bar.get('ATR', bar['Close'] * 0.02) / bar['Close'],
                    distance_from_ema=0,
                    rsi_value=bar.get('RSI', 50),
                    volume_ratio=bar.get('volume_ratio', 1.0),
                    expected_rr=take_profit_pct / stop_loss_pct
                )
                
                position = {
                    'side': 'SHORT',
                    'entry_price': bar['Close'],
                    'entry_bar': i,
                    'qty': qty,
                    'context': context
                }
        
        equity.append(balance)
    
    trades_df = pd.DataFrame(trades)
    
    # Calculate daily returns
    equity_series = pd.Series(equity, index=df.index)
    daily_returns = equity_series.resample('D').last().pct_change().dropna()
    
    return trades_df, equity, daily_returns


def calculate_setup_strength(bar, signal: int, regime: MarketRegime) -> float:
    """
    Score how good a setup is (0-1)
    This is where you'd add your "setup quality" logic
    """
    score = 0.5  # Base score
    
    # Bonus for strong trend when taking momentum trade
    if signal != 0 and regime.trend_strength > 0.5:
        score += 0.2
    
    # Bonus for high efficiency (clean move)
    if regime.efficiency_ratio > 0.5:
        score += 0.15
    
    # Bonus for normal volatility
    if regime.volatility_regime == 'Normal':
        score += 0.15
    
    # Penalty for extreme volatility
    if regime.volatility_regime == 'Extreme':
        score -= 0.3
    
    # RSI context
    rsi = bar.get('RSI', 50)
    if signal == 1 and 40 < rsi < 60:  # Long in mild bullish
        score += 0.1
    elif signal == -1 and 40 < rsi < 60:  # Short in mild bearish
        score += 0.1
    
    return max(0, min(1, score))


# ==============================================================================
# ENHANCED BATCH RUNNER WITH ATTRIBUTION
# ==============================================================================
def run_enhanced_batch_analysis(
    symbols: list,
    strategies: dict,
    timeframe: str,
    params: dict,
    enable_regime_filter: bool = True,
    progress_callback=None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Enhanced batch runner that returns:
    1. Summary results (same as before)
    2. ALL TRADES with full attribution
    
    Returns:
        (summary_df, all_trades_df)
    """
    results = []
    all_trades = []
    
    total_runs = len(symbols) * len(strategies)
    current_run = 0
    
    DATA_DIR = Path("data/derived")
    
    for symbol in symbols:
        symbol_path = DATA_DIR / symbol / timeframe
        
        if not symbol_path.exists():
            if progress_callback:
                progress_callback(f"⚠️ No data for {symbol}")
            continue
        
        parquet_files = list(symbol_path.glob("*.parquet"))
        if not parquet_files:
            continue
        
        data_file = sorted(parquet_files)[-1]
        
        try:
            df = pd.read_parquet(data_file)
            
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
            
            df.columns = [c.title() if c.lower() in ['open', 'high', 'low', 'close', 'volume'] else c 
                          for c in df.columns]
            
            for strategy_name, (strategy_func, strategy_params) in strategies.items():
                current_run += 1
                
                if progress_callback:
                    progress_callback(f"[{current_run}/{total_runs}] {symbol} + {strategy_name}...")
                
                try:
                    # Apply strategy
                    df_strategy = strategy_func(df.copy(), **strategy_params)
                    
                    # Add regime indicators
                    df_strategy = compute_regime_indicators(df_strategy)
                    
                    # Run enhanced backtest
                    trades_df, equity, daily_returns = backtest_with_attribution(
                        df_strategy,
                        strategy_name=strategy_name,
                        initial_capital=params['initial_capital'],
                        risk_per_trade_pct=params['risk_per_trade'],
                        stop_loss_pct=params['stop_loss_pct'],
                        take_profit_pct=params['take_profit_pct'],
                        min_holding_bars=params.get('min_holding_bars', 3),
                        enable_regime_filter=enable_regime_filter
                    )
                    
                    if not trades_df.empty:
                        # Add symbol/strategy to each trade
                        trades_df['symbol'] = symbol
                        trades_df['strategy_name'] = strategy_name  # Match column name
                        trades_df['timeframe'] = timeframe
                        
                        # Add to master trade list
                        all_trades.append(trades_df)
                        
                        # Calculate metrics
                        metrics = compute_metrics(trades_df, initial_capital=params['initial_capital'])
                        
                        final_capital = equity[-1]
                        total_return = ((final_capital - params['initial_capital']) / params['initial_capital']) * 100
                        
                        sharpe = 0
                        if len(daily_returns) > 0 and daily_returns.std() > 0:
                            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
                        
                        wins = trades_df[trades_df['pnl'] > 0]
                        losses = trades_df[trades_df['pnl'] <= 0]
                        
                        avg_win_pct = wins['pnl_pct'].mean() if not wins.empty else 0
                        avg_loss_pct = abs(losses['pnl_pct'].mean()) if not losses.empty else 0
                        
                        win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0
                        expectancy = (win_rate * avg_win_pct) - ((1 - win_rate) * avg_loss_pct)
                        
                        # Calculate regime-specific metrics
                        trending_trades = trades_df[trades_df['price_regime'] == 'Trending']
                        choppy_trades = trades_df[trades_df['price_regime'] == 'Choppy']
                        
                        trending_win_rate = (trending_trades['pnl'] > 0).mean() * 100 if len(trending_trades) > 0 else 0
                        choppy_win_rate = (choppy_trades['pnl'] > 0).mean() * 100 if len(choppy_trades) > 0 else 0
                        
                        results.append({
                            'Symbol': symbol,
                            'Strategy': strategy_name,
                            'Timeframe': timeframe,
                            'Final Capital': final_capital,
                            'Total Return %': total_return,
                            'Total Trades': len(trades_df),
                            'Win Rate %': win_rate * 100,
                            'Profit Factor': metrics.get('Profit Factor', 0),
                            'Sharpe Ratio': round(sharpe, 2),
                            'Max DD %': abs(metrics.get('Max Drawdown %', 0)),
                            'Avg Win %': round(avg_win_pct, 2),
                            'Avg Loss %': round(avg_loss_pct, 2),
                            'Expectancy': round(expectancy, 2),
                            
                            # New regime-aware metrics
                            'Trending Trades': len(trending_trades),
                            'Choppy Trades': len(choppy_trades),
                            'Trending Win %': round(trending_win_rate, 1),
                            'Choppy Win %': round(choppy_win_rate, 1),
                            
                            'Status': '✅ Success'
                        })
                    else:
                        results.append({
                            'Symbol': symbol,
                            'Strategy': strategy_name,
                            'Status': '⚠️ No Trades'
                        })
                
                except Exception as e:
                    results.append({
                        'Symbol': symbol,
                        'Strategy': strategy_name,
                        'Status': f'❌ Error: {str(e)}'
                    })
        
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Failed to load {symbol}: {e}")
    
    summary_df = pd.DataFrame(results)
    all_trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    
    return summary_df, all_trades_df


# ==============================================================================
# TRADE ANALYSIS FUNCTIONS
# ==============================================================================
def analyze_trades_by_regime(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze performance by market regime
    This is THE KEY INSIGHT
    """
    if trades_df.empty:
        return pd.DataFrame()
    
    regime_analysis = []
    
    for price_regime in ['Trending', 'Choppy', 'Ranging']:
        subset = trades_df[trades_df['price_regime'] == price_regime]
        
        if len(subset) > 0:
            wins = subset[subset['pnl'] > 0]
            losses = subset[subset['pnl'] <= 0]
            
            win_rate = len(wins) / len(subset) * 100
            avg_win = wins['pnl_pct'].mean() if not wins.empty else 0
            avg_loss = abs(losses['pnl_pct'].mean()) if not losses.empty else 0
            expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)
            
            regime_analysis.append({
                'Regime': price_regime,
                'Trades': len(subset),
                'Win Rate %': round(win_rate, 1),
                'Avg Win %': round(avg_win, 2),
                'Avg Loss %': round(avg_loss, 2),
                'Expectancy': round(expectancy, 2),
                'Total P&L': subset['pnl'].sum()
            })
    
    return pd.DataFrame(regime_analysis)


def analyze_trades_by_setup_quality(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Does setup quality actually correlate with win rate?
    """
    if trades_df.empty or 'setup_strength' not in trades_df.columns:
        return pd.DataFrame()
    
    # Bin by setup strength
    trades_df['quality_bin'] = pd.cut(
        trades_df['setup_strength'],
        bins=[0, 0.4, 0.6, 0.8, 1.0],
        labels=['Poor', 'Fair', 'Good', 'Excellent']
    )
    
    quality_analysis = []
    
    for quality in ['Poor', 'Fair', 'Good', 'Excellent']:
        subset = trades_df[trades_df['quality_bin'] == quality]
        
        if len(subset) > 0:
            wins = subset[subset['pnl'] > 0]
            win_rate = len(wins) / len(subset) * 100
            avg_pnl = subset['pnl_pct'].mean()
            
            quality_analysis.append({
                'Setup Quality': quality,
                'Trades': len(subset),
                'Win Rate %': round(win_rate, 1),
                'Avg P&L %': round(avg_pnl, 2),
                'Total P&L': subset['pnl'].sum()
            })
    
    return pd.DataFrame(quality_analysis)


# ==============================================================================
# UI
# ==============================================================================
st.title("🧠 Enhanced Batch Stock Analyzer")
st.markdown("**With Regime Awareness & Trade Attribution**")

with st.expander("🔑 What's New in Enhanced Version", expanded=True):
    st.markdown("""
    ### Critical Improvements Over Basic Version:
    
    #### 1. **Trade Attribution** 🎯
    Every trade now tracks:
    - WHY it was taken (entry signal)
    - Market regime at entry (Trending/Choppy/Ranging)
    - Setup quality score (0-1)
    - Volatility state (Low/Normal/High/Extreme)
    - Technical context (RSI, volume, ATR multiple)
    
    #### 2. **Regime Filtering** 🚦
    - Momentum strategies ONLY trade in trending regimes
    - Mean reversion ONLY trades in choppy/ranging markets
    - This dramatically improves win rates by avoiding bad setups
    
    #### 3. **Setup Quality Scoring** ⭐
    - Scores each trade 0-1 based on confluence
    - You can see if "good" setups actually win more
    
    #### 4. **Expectancy by Regime** 💰
    - Shows which regimes are profitable vs bleed
    - E[R] = (Win% × AvgWin) - (Loss% × AvgLoss)
    
    #### 5. **Decision Traceability** 📋
    - Export ALL trades with full context
    - Debug why specific trades happened
    - Analyze which entry signals work best
    """)

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Symbol Selection
    st.subheader("📊 Symbols")
    
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
            "Test 5 Stocks": symbol_list[:5] if symbol_list else [],
            "Top 10 Nifty": symbol_list[:10] if symbol_list else [],
            "Top 20 Nifty": symbol_list[:20] if symbol_list else [],
            "Top 50 Nifty": symbol_list[:50] if symbol_list else []
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
        default=["Simple Momentum"]
    )
    
    strategies_to_run = {name: strategy_options[name] for name in selected_strategies}
    
    st.divider()
    
    # NEW: Regime Filter Toggle
    st.subheader("🧠 Intelligence")
    enable_regime_filter = st.checkbox(
        "Enable Regime Filtering", 
        value=True,
        help="Only trade when market regime suits the strategy"
    )
    
    if enable_regime_filter:
        st.success("✅ Smart mode: Filters by regime")
    else:
        st.warning("⚠️ Blind mode: Trades all signals")
    
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
    
    backtest_params = {
        'initial_capital': initial_capital,
        'risk_per_trade': risk_per_trade,
        'stop_loss_pct': stop_loss_pct,
        'take_profit_pct': take_profit_pct,
        'min_holding_bars': 3
    }

# ==============================================================================
# MAIN
# ==============================================================================
if not selected_symbols:
    st.warning("⚠️ Please select symbols")
    st.stop()

if not selected_strategies:
    st.warning("⚠️ Please select strategies")
    st.stop()

# Configuration summary
st.subheader("📋 Configuration")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Symbols", len(selected_symbols))
col2.metric("Strategies", len(selected_strategies))
col3.metric("Total Runs", len(selected_symbols) * len(selected_strategies))
col4.metric("Regime Filter", "ON" if enable_regime_filter else "OFF")

st.divider()

# Run button
run_batch = st.button("▶️ Run Enhanced Analysis", type="primary", use_container_width=True)

progress_bar = st.progress(0)
status_text = st.empty()
results_container = st.container()

# ==============================================================================
# RUN
# ==============================================================================
if run_batch:
    with st.spinner("Running enhanced batch analysis..."):
        
        def update_progress(message):
            status_text.text(message)
        
        summary_df, all_trades_df = run_enhanced_batch_analysis(
            symbols=selected_symbols,
            strategies=strategies_to_run,
            timeframe=timeframe,
            params=backtest_params,
            enable_regime_filter=enable_regime_filter,
            progress_callback=update_progress
        )
        
        progress_bar.progress(100)
        status_text.success(f"✅ Complete! {len(summary_df)} runs, {len(all_trades_df)} trades analyzed")
        
        st.session_state['enhanced_summary'] = summary_df
        st.session_state['enhanced_trades'] = all_trades_df
        st.session_state['batch_timestamp'] = datetime.now()

# ==============================================================================
# DISPLAY RESULTS
# ==============================================================================
if 'enhanced_summary' in st.session_state:
    summary_df = st.session_state['enhanced_summary']
    all_trades_df = st.session_state['enhanced_trades']
    timestamp = st.session_state.get('batch_timestamp', datetime.now())
    
    with results_container:
        st.divider()
        st.header("📊 Enhanced Results")
        st.caption(f"Completed: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        successful_runs = summary_df[summary_df['Status'] == '✅ Success'].copy()
        
        if successful_runs.empty:
            st.error("No successful runs")
            st.stop()
        
        # Tab layout for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Summary", 
            "🎯 Regime Analysis", 
            "⭐ Setup Quality",
            "🔍 Trade Deep Dive",
            "💾 Export"
        ])
        
        # ===== TAB 1: SUMMARY =====
        with tab1:
            st.subheader("📈 Performance Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Successful Runs", len(successful_runs))
            col2.metric("Avg Return", f"{successful_runs['Total Return %'].mean():.2f}%")
            col3.metric("Avg Win Rate", f"{successful_runs['Win Rate %'].mean():.1f}%")
            col4.metric("Avg Expectancy", f"{successful_runs['Expectancy'].mean():.2f}%")
            
            st.subheader("🏆 Top Performers")
            top_n = st.slider("Show top N", 5, 20, 10)
            
            top_performers = successful_runs.nlargest(top_n, 'Total Return %')
            
            st.dataframe(
                top_performers[[
                    'Symbol', 'Strategy', 'Total Return %', 'Win Rate %', 
                    'Expectancy', 'Trending Win %', 'Choppy Win %', 'Total Trades'
                ]].style.background_gradient(subset=['Total Return %'], cmap='RdYlGn'),
                use_container_width=True
            )
            
            # Strategy comparison
            st.subheader("📊 Strategy Comparison")
            
            strategy_summary = successful_runs.groupby('Strategy').agg({
                'Total Return %': 'mean',
                'Win Rate %': 'mean',
                'Expectancy': 'mean',
                'Trending Win %': 'mean',
                'Choppy Win %': 'mean',
                'Total Trades': 'sum'
            }).round(2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(strategy_summary, use_container_width=True)
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name='Trending',
                    x=strategy_summary.index,
                    y=strategy_summary['Trending Win %'],
                    marker_color='lightblue'
                ))
                fig.add_trace(go.Bar(
                    name='Choppy',
                    x=strategy_summary.index,
                    y=strategy_summary['Choppy Win %'],
                    marker_color='coral'
                ))
                fig.update_layout(
                    title="Win Rate by Regime",
                    barmode='group',
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # ===== TAB 2: REGIME ANALYSIS =====
        with tab2:
            st.subheader("🎯 Performance by Market Regime")
            st.markdown("**This is THE KEY INSIGHT** - See when your strategy actually works")
            
            if not all_trades_df.empty:
                regime_analysis = analyze_trades_by_regime(all_trades_df)
                
                if not regime_analysis.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.dataframe(
                            regime_analysis.style.background_gradient(
                                subset=['Expectancy'], 
                                cmap='RdYlGn'
                            ),
                            use_container_width=True
                        )
                    
                    with col2:
                        fig = go.Figure(data=[
                            go.Bar(
                                x=regime_analysis['Regime'],
                                y=regime_analysis['Expectancy'],
                                marker_color=['green' if x > 0 else 'red' 
                                             for x in regime_analysis['Expectancy']],
                                text=regime_analysis['Expectancy'].round(2),
                                textposition='auto'
                            )
                        ])
                        fig.update_layout(
                            title="Expectancy by Regime",
                            yaxis_title="Expected Return %",
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Insight box
                    best_regime = regime_analysis.loc[regime_analysis['Expectancy'].idxmax()]
                    worst_regime = regime_analysis.loc[regime_analysis['Expectancy'].idxmin()]
                    
                    st.info(f"""
                    **🎯 Key Insights:**
                    - Best Regime: **{best_regime['Regime']}** (E[R] = {best_regime['Expectancy']:.2f}%)
                    - Worst Regime: **{worst_regime['Regime']}** (E[R] = {worst_regime['Expectancy']:.2f}%)
                    - **Action**: Avoid trading in {worst_regime['Regime']} markets!
                    """)
                    
                    # Strategy-specific regime breakdown
                    st.subheader("Strategy × Regime Breakdown")
                    
                    strategy_regime_pivot = all_trades_df.pivot_table(
                        values='pnl',
                        index='strategy_name',
                        columns='price_regime',
                        aggfunc='mean',
                        fill_value=0
                    )
                    
                    st.dataframe(
                        strategy_regime_pivot.style.background_gradient(
                            cmap='RdYlGn', axis=None
                        ),
                        use_container_width=True
                    )
        
        # ===== TAB 3: SETUP QUALITY =====
        with tab3:
            st.subheader("⭐ Does Setup Quality Matter?")
            st.markdown("**Test if 'better' setups actually win more**")
            
            if not all_trades_df.empty and 'setup_strength' in all_trades_df.columns:
                quality_analysis = analyze_trades_by_setup_quality(all_trades_df)
                
                if not quality_analysis.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.dataframe(
                            quality_analysis.style.background_gradient(
                                subset=['Win Rate %'],
                                cmap='RdYlGn'
                            ),
                            use_container_width=True
                        )
                    
                    with col2:
                        fig = go.Figure(data=[
                            go.Scatter(
                                x=quality_analysis['Setup Quality'],
                                y=quality_analysis['Win Rate %'],
                                mode='lines+markers',
                                marker=dict(size=12, color='blue'),
                                line=dict(width=3)
                            )
                        ])
                        fig.update_layout(
                            title="Win Rate vs Setup Quality",
                            xaxis_title="Setup Quality",
                            yaxis_title="Win Rate %",
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Insight
                    corr = quality_analysis[['Setup Quality', 'Win Rate %']].apply(
                        lambda x: pd.factorize(x)[0] if x.dtype == 'object' else x
                    ).corr().iloc[0, 1]
                    
                    if corr > 0.5:
                        st.success(f"✅ Setup quality correlates with win rate (r={corr:.2f}). Use it!")
                    else:
                        st.warning(f"⚠️ Weak correlation (r={corr:.2f}). Setup scoring needs work.")
        
        # ===== TAB 4: TRADE DEEP DIVE =====
        with tab4:
            st.subheader("🔍 Individual Trade Analysis")
            st.markdown("**Full attribution for every trade**")
            
            if not all_trades_df.empty:
                # Filters
                col1, col2, col3 = st.columns(3)
                with col1:
                    filter_symbol = st.multiselect(
                        "Filter Symbol", 
                        all_trades_df['symbol'].unique() if 'symbol' in all_trades_df.columns else []
                    )
                with col2:
                    filter_strategy = st.multiselect(
                        "Filter Strategy",
                        all_trades_df['strategy_name'].unique() if 'strategy_name' in all_trades_df.columns else []
                    )
                with col3:
                    show_winners_only = st.checkbox("Winners Only", value=False)
                
                # Apply filters
                filtered_trades = all_trades_df.copy()
                
                if filter_symbol:
                    filtered_trades = filtered_trades[filtered_trades['symbol'].isin(filter_symbol)]
                if filter_strategy:
                    filtered_trades = filtered_trades[filtered_trades['strategy_name'].isin(filter_strategy)]
                if show_winners_only:
                    filtered_trades = filtered_trades[filtered_trades['pnl'] > 0]
                
                st.info(f"Showing {len(filtered_trades)} trades")
                
                # Display with context
                display_cols = [
                    'Entry', 'Exit', 'symbol', 'strategy_name', 'Side', 
                    'pnl_pct', 'exit_reason',
                    'price_regime', 'vol_regime', 'trend_strength',
                    'setup_strength', 'rsi', 'expected_rr'
                ]
                
                available_cols = [col for col in display_cols if col in filtered_trades.columns]
                
                st.dataframe(
                    filtered_trades[available_cols].style.applymap(
                        lambda x: 'background-color: lightgreen' if isinstance(x, (int, float)) and x > 0 
                        else 'background-color: lightcoral' if isinstance(x, (int, float)) and x < 0
                        else '',
                        subset=['pnl_pct'] if 'pnl_pct' in available_cols else []
                    ),
                    use_container_width=True,
                    height=400
                )
                
                # Random trade inspector
                st.subheader("🎲 Random Trade Inspector")
                if st.button("Show Random Trade"):
                    random_trade = filtered_trades.sample(1).iloc[0]
                    
                    st.json(random_trade.to_dict())
        
        # ===== TAB 5: EXPORT =====
        with tab5:
            st.subheader("💾 Export Enhanced Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Summary CSV
                csv_summary = successful_runs.to_csv(index=False)
                st.download_button(
                    label="📥 Download Summary CSV",
                    data=csv_summary,
                    file_name=f"enhanced_summary_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # All trades CSV (THIS IS THE GOLD)
                if not all_trades_df.empty:
                    csv_trades = all_trades_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download ALL TRADES (with context)",
                        data=csv_trades,
                        file_name=f"all_trades_attributed_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            # Regime analysis report
            if not all_trades_df.empty:
                regime_report = analyze_trades_by_regime(all_trades_df)
                
                report_text = f"""
ENHANCED BATCH ANALYSIS REPORT
==============================
Timestamp: {timestamp}
Regime Filter: {'ENABLED' if enable_regime_filter else 'DISABLED'}

SUMMARY
-------
Total Trades: {len(all_trades_df)}
Symbols: {len(selected_symbols)}
Strategies: {len(selected_strategies)}

PERFORMANCE BY REGIME
--------------------
{regime_report.to_string(index=False) if not regime_report.empty else 'N/A'}

TOP INSIGHTS
-----------
{successful_runs.nlargest(5, 'Total Return %')[['Symbol', 'Strategy', 'Total Return %', 'Expectancy']].to_string(index=False)}

KEY TAKEAWAY
-----------
The regime-specific analysis shows which market conditions are profitable.
Focus trading on regimes with positive expectancy.
Avoid regimes where expectancy is negative.
"""
                
                st.download_button(
                    label="📄 Download Analysis Report",
                    data=report_text,
                    file_name=f"regime_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

# ==============================================================================
# DOCUMENTATION
# ==============================================================================
with st.expander("📚 How to Use Enhanced Analyzer"):
    st.markdown("""
    ## 🎯 What Makes This "Enhanced"?
    
    ### 1. Trade Attribution
    Every trade now has full context:
    - Entry signal type
    - Market regime (Trending/Choppy/Ranging)
    - Volatility state (Low/Normal/High/Extreme)
    - Setup quality score
    - Technical indicators at entry
    
    ### 2. Regime Filtering
    **This is the game-changer**:
    - Momentum strategies only trade in trending regimes
    - Mean reversion only trades in choppy markets
    - Dramatically improves win rates by avoiding bad setups
    
    Toggle "Enable Regime Filtering" in sidebar to test impact.
    
    ### 3. Performance by Regime
    See EXACTLY when your strategy works:
    - Which regimes are profitable vs bleed
    - Expectancy (E[R]) per regime
    - Win rate variations across market states
    
    ### 4. Setup Quality Analysis
    Tests if "better" setups actually win more:
    - Scores trades 0-1 based on confluence
    - Shows correlation between quality and outcome
    
    ## 📊 How to Read Results
    
    ### Summary Tab
    - Overall performance metrics
    - Top symbol-strategy combinations
    - Win rates by regime (Trending vs Choppy)
    
    ### Regime Analysis Tab
    **THE MOST IMPORTANT VIEW**:
    - Shows expectancy by market regime
    - Identifies which regimes to trade vs avoid
    - Strategy-specific regime breakdowns
    
    ### Setup Quality Tab
    - Tests if setup scoring works
    - If correlation > 0.5, setup quality matters
    - Use this to refine entry logic
    
    ### Trade Deep Dive
    - Every single trade with full attribution
    - Filter by symbol, strategy, outcome
    - Debug specific trades
    
    ### Export Tab
    - Download summary results
    - **Download ALL TRADES with context** (this is gold)
    - Regime analysis report
    
    ## 💡 Key Insights You'll Get
    
    1. **When NOT to Trade**: Identify losing regimes
    2. **Best Symbol-Strategy Pairs**: Not all stocks fit all strategies
    3. **Setup Quality Impact**: Does confluence matter?
    4. **Regime Filter Impact**: Compare filtered vs unfiltered
    
    ## 🚀 Next Steps
    
    1. Run with regime filter OFF, then ON - compare results
    2. Export "All Trades" CSV and analyze in Excel
    3. Focus on strategies with positive expectancy in target regimes
    4. Avoid trading in regimes with negative expectancy
    5. Test different timeframes to see regime stability
    
    ## ⚠️ Important Notes
    
    - Regime classification uses simple, robust indicators (not complex math)
    - ATR percentile, Efficiency Ratio, Trend Strength are battle-tested
    - "Setup quality" can be customized in `calculate_setup_strength()`
    - Export ALL trades to build your own analysis in Jupyter/Excel
    """)