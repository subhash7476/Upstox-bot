# core/batch_backtester.py
"""
Batch Backtester - Test multiple stocks simultaneously
Compare performance across your entire portfolio
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Callable
import time


class BatchBacktester:
    """
    Test multiple stocks with the same strategy
    Returns comparative results table
    """
    
    def __init__(self, data_dir: Path = Path("data/derived")):
        self.data_dir = data_dir
        self.results = []
        
    def find_stocks(self, timeframe: str = "15minute", 
                    min_bars: int = 1000) -> List[Dict]:
        """
        Scan data directory for available stocks
        
        Returns:
            List of dicts with stock info
        """
        stocks = []
        
        for stock_dir in self.data_dir.iterdir():
            if not stock_dir.is_dir():
                continue
                
            tf_dir = stock_dir / timeframe
            if not tf_dir.exists():
                continue
                
            files = list(tf_dir.glob("*.parquet"))
            if not files:
                continue
                
            try:
                df = pd.read_parquet(files[0])
                
                if len(df) < min_bars:
                    continue
                    
                # Calculate basic stats
                avg_range = ((df['High'] - df['Low']) / df['Close'] * 100).mean()
                avg_volume = df['Volume'].mean() if 'Volume' in df.columns else 0
                
                stocks.append({
                    'symbol': stock_dir.name,
                    'file': files[0],
                    'bars': len(df),
                    'avg_range_pct': round(avg_range, 3),
                    'avg_volume': int(avg_volume),
                    'date_start': df.index[0],
                    'date_end': df.index[-1]
                })
                
            except Exception as e:
                print(f"Error loading {stock_dir.name}: {e}")
                continue
        
        return sorted(stocks, key=lambda x: x['avg_range_pct'], reverse=True)
    
    def run_batch(self,
                  stocks: List[str],
                  strategy_func: Callable,
                  backtest_func: Callable,
                  strategy_params: Dict,
                  backtest_params: Dict,
                  progress_callback: Callable = None) -> pd.DataFrame:
        """
        Run backtest on multiple stocks
        
        Args:
            stocks: List of stock symbols
            strategy_func: Strategy function (e.g., mean_reversion_basic)
            backtest_func: Backtest engine function
            strategy_params: Parameters for strategy
            backtest_params: Parameters for backtest (capital, risk, SL, TP)
            progress_callback: Optional callback for progress updates
            
        Returns:
            DataFrame with comparative results
        """
        results = []
        
        for i, symbol in enumerate(stocks, 1):
            if progress_callback:
                progress_callback(f"[{i}/{len(stocks)}] Testing {symbol}...")
            
            try:
                # Find and load data
                stock_dir = self.data_dir / symbol / "15minute"
                files = list(stock_dir.glob("*.parquet"))
                
                if not files:
                    results.append({
                        'Symbol': symbol,
                        'Status': 'No Data',
                        'Trades': 0,
                        'Win Rate %': 0,
                        'Profit Factor': 0,
                        'Total PnL': 0,
                        'Sharpe': 0
                    })
                    continue
                
                df = pd.read_parquet(files[0])
                
                # Ensure datetime index
                if not isinstance(df.index, pd.DatetimeIndex):
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                
                # Standardize columns
                df.columns = [c.title() if c.lower() in ['open', 'high', 'low', 'close', 'volume'] 
                             else c for c in df.columns]
                
                # Apply strategy
                df_strategy = strategy_func(df.copy(), **strategy_params)
                
                # Run backtest
                trades_df, equity_curve, daily_returns = backtest_func(
                    df_strategy,
                    **backtest_params
                )
                
                # Calculate metrics
                if not trades_df.empty:
                    wins = trades_df[trades_df['pnl'] > 0]
                    losses = trades_df[trades_df['pnl'] < 0]
                    
                    total_trades = len(trades_df)
                    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
                    total_pnl = trades_df['pnl'].sum()
                    
                    gross_win = wins['pnl'].sum()
                    gross_loss = abs(losses['pnl'].sum())
                    profit_factor = gross_win / gross_loss if gross_loss > 0 else 0
                    
                    avg_win = wins['pnl'].mean() if not wins.empty else 0
                    avg_loss = abs(losses['pnl'].mean()) if not losses.empty else 0
                    avg_win_loss = avg_win / avg_loss if avg_loss > 0 else 0
                    
                    # Sharpe ratio
                    if len(daily_returns) > 1:
                        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
                    else:
                        sharpe = 0
                    
                    # Max drawdown
                    initial_capital = backtest_params.get('initial_capital', 100000)
                    equity_series = pd.Series(equity_curve)
                    peak = equity_series.cummax()
                    drawdown = (equity_series - peak) / peak
                    max_dd_pct = drawdown.min() * 100
                    
                    results.append({
                        'Symbol': symbol,
                        'Status': '✅',
                        'Trades': total_trades,
                        'Win Rate %': round(win_rate, 1),
                        'Profit Factor': round(profit_factor, 2),
                        'Avg W/L': round(avg_win_loss, 2),
                        'Total PnL': round(total_pnl, 0),
                        'Sharpe': round(sharpe, 2),
                        'Max DD %': round(max_dd_pct, 2),
                        'Bars': len(df)
                    })
                else:
                    results.append({
                        'Symbol': symbol,
                        'Status': 'No Trades',
                        'Trades': 0,
                        'Win Rate %': 0,
                        'Profit Factor': 0,
                        'Avg W/L': 0,
                        'Total PnL': 0,
                        'Sharpe': 0,
                        'Max DD %': 0,
                        'Bars': len(df)
                    })
                
            except Exception as e:
                results.append({
                    'Symbol': symbol,
                    'Status': f'❌ {str(e)[:20]}',
                    'Trades': 0,
                    'Win Rate %': 0,
                    'Profit Factor': 0,
                    'Avg W/L': 0,
                    'Total PnL': 0,
                    'Sharpe': 0,
                    'Max DD %': 0,
                    'Bars': 0
                })
                if progress_callback:
                    progress_callback(f"Error on {symbol}: {str(e)[:50]}")
        
        # Convert to DataFrame and sort by Total PnL
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('Total PnL', ascending=False)
        
        return df_results
    
    def filter_best_stocks(self, 
                          results_df: pd.DataFrame,
                          min_profit_factor: float = 1.3,
                          min_win_rate: float = 50.0,
                          min_trades: int = 50) -> pd.DataFrame:
        """
        Filter results to show only profitable stocks
        """
        filtered = results_df[
            (results_df['Profit Factor'] >= min_profit_factor) &
            (results_df['Win Rate %'] >= min_win_rate) &
            (results_df['Trades'] >= min_trades)
        ]
        
        return filtered.sort_values('Total PnL', ascending=False)
    
    def get_portfolio_summary(self, results_df: pd.DataFrame) -> Dict:
        """
        Calculate portfolio-level statistics
        """
        profitable = results_df[results_df['Total PnL'] > 0]
        
        summary = {
            'Total Stocks Tested': len(results_df),
            'Profitable Stocks': len(profitable),
            'Win Rate (Stocks)': f"{len(profitable) / len(results_df) * 100:.1f}%",
            'Total PnL (Portfolio)': results_df['Total PnL'].sum(),
            'Avg PnL per Stock': results_df['Total PnL'].mean(),
            'Best Stock': results_df.iloc[0]['Symbol'] if not results_df.empty else 'N/A',
            'Best PnL': results_df.iloc[0]['Total PnL'] if not results_df.empty else 0,
            'Avg Profit Factor': results_df[results_df['Profit Factor'] > 0]['Profit Factor'].mean(),
            'Avg Win Rate': results_df[results_df['Win Rate %'] > 0]['Win Rate %'].mean(),
        }
        
        return summary


def quick_batch_test(stock_symbols: List[str], 
                     strategy_name: str = 'mean_reversion',
                     sl_pct: float = 0.30,
                     tp_pct: float = 0.70,
                     risk_pct: float = 2.5) -> pd.DataFrame:
    """
    Quick batch test with default parameters
    
    Usage:
        results = quick_batch_test(['ADANIGREEN', 'ADANIPOWER', 'INFY'])
    """
    from core.strategies.mean_reversion import mean_reversion_advanced
    
    # Import backtest function (you'll need to adjust this import)
    from pages.batch_strategy_lab import backtest_strategy
    
    batch = BatchBacktester()
    
    strategy_params = {
        'bb_period': 20,
        'rsi_period': 14,
        'volume_filter': True,
        'time_filter': True
    }
    
    backtest_params = {
        'initial_capital': 100000,
        'risk_per_trade_pct': risk_pct,
        'stop_loss_pct': sl_pct,
        'take_profit_pct': tp_pct,
        'min_holding_bars': 3
    }
    
    results = batch.run_batch(
        stocks=stock_symbols,
        strategy_func=mean_reversion_advanced,
        backtest_func=backtest_strategy,
        strategy_params=strategy_params,
        backtest_params=backtest_params
    )
    
    return results
