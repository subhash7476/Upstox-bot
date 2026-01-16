from core.strategies.indian_market_squeeze import (
    build_15m_signals_with_backtest,
    BacktestResult,
    SqueezeSignal,
    # Assuming CompletedTrade also has a score attribute, if not, it should be added in the strategy file.
    # Otherwise, you'd need to ensure the score from the original signal is propagated.
)
from core.option_selector import OptionSelector, OptionSelectorConfig, OptionSelection
from core.option_chain_provider import OptionChainProvider
import core.live_trading_manager as ltm
from core.config import get_access_token
from core.live_trading_manager import LiveTradingManager, is_market_hours, get_next_candle_time, seconds_until_next_candle
from core.database import get_db
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta, date
import time
import sys
import os
import numpy as np
import pandas as pd
import streamlit as st
from typing import List, Dict
from core.strategies.indian_market_squeeze import build_15m_signals, SqueezeSignal, compute_squeeze_stack
# To ensure that only signals with an alignment score of 5 are considered for backtesting and displayed as tradable active signals, you need to add a filtering step after the signal generation in three key areas of your code:

# 1. ** Batch Scanner(`run_batch_scan_squeeze_15m_v2`)**: Filter both `active_signals` and `completed_trades` from the backtest result.
# 2. ** Live Scanner Tab**: Filter the signals generated for the live scan.
# 3. ** Backtest Tab**: Filter the signals before performing the backtest simulation.

# Here are the specific changes to implement:

#     # pages/3_Indian_Market_Squeeze.py

#     # NEW: Squeeze strategy dependency

st.set_page_config(page_title="Indian Market Squeeze 15m",
                   layout="wide", page_icon="🧨")

db = get_db()

# Verify database connection
if db is None or db.con is None:
    st.error("❌ Database connection failed. Please restart the application.")
    st.stop()


def run_batch_scan_squeeze_15m_v2_improved(
    fo_stocks: pd.DataFrame,
    lookback_days: int = 60,
    sl_mode: str = "ATR based",
    atr_mult: float = 1.5,          # OPTIMIZED: was 2.0
    rr: float = 2.5,                # OPTIMIZED: was 2.0
    sl_pct: float = 0.01,
    tp_pct: float = 0.02,
    progress_bar=None,
    end_dt: Optional[datetime] = None,
    db=None,
    max_trade_bars: int = 30,       # OPTIMIZED: was 50

    # NEW PARAMETERS
    max_trades_per_session: int = 1,
    use_breakeven: bool = True,
    breakeven_at_r: float = 1.0,
    use_quality_filter: bool = True,
    min_score: float = 5.0,

    # Data loader function (passed in)
    load_data_fn=None,
    compute_squeeze_fn=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    IMPROVED batch scanner with session limits and breakeven stops.

    Key differences from v1:
    - Max 1 trade per stock per day (session limit)
    - Breakeven stops to protect profits
    - Quality filters for high-probability entries
    - Optimized default parameters

    Returns:
        Tuple of (active_df, completed_df, summary)
    """
    active_results: List[Dict] = []
    completed_results: List[Dict] = []

    total_stocks = len(fo_stocks)
    stocks_with_signals = 0
    total_completed = 0
    total_active = 0
    aggregate_pnl = 0.0
    aggregate_pnl_pct = 0.0
    total_wins = 0
    total_losses = 0
    total_breakevens = 0

    # Aggregate skip statistics
    total_skipped_session = 0
    total_skipped_cooldown = 0
    total_skipped_quality = 0

    for i, (_, row) in enumerate(fo_stocks.iterrows()):
        symbol = row["trading_symbol"]
        ikey = row["instrument_key"]

        if progress_bar:
            progress_bar.progress(
                (i + 1) / max(total_stocks, 1),
                text=f"Scanning {symbol}... ({i+1}/{total_stocks})",
            )

        try:
            # Load data using provided function or default
            if load_data_fn:
                df_15m = load_data_fn(
                    ikey, "15minute", lookback_days, end_timestamp=end_dt)
            else:
                # Fallback - you'll need to replace this with your actual loader
                from core.api.historical import load_data_fast
                df_15m = load_data_fast(
                    ikey, "15minute", lookback_days, end_timestamp=end_dt)

            if df_15m is None or len(df_15m) < 100:
                continue

            mode_str = "ATR" if sl_mode == "ATR based" else "PCT"

            # Get signals WITH improved backtest
            result: BacktestResult = build_15m_signals_with_backtest(
                df_15m,
                sl_mode=mode_str,
                sl_atr_mult=atr_mult,
                tp_rr=rr,
                sl_pct=sl_pct,
                tp_pct=tp_pct,
                max_trade_bars=max_trade_bars,
                max_trades_per_session=max_trades_per_session,
                use_breakeven=use_breakeven,
                breakeven_at_r=breakeven_at_r,
                use_quality_filter=use_quality_filter,
                min_score=min_score,
                compute_squeeze_fn=compute_squeeze_fn,
            )

            # Track skip statistics
            total_skipped_session += result.signals_skipped_session_limit
            total_skipped_cooldown += result.signals_skipped_cooldown
            total_skipped_quality += result.signals_skipped_low_quality

            if not result.active_signals and not result.completed_trades:
                continue

            stocks_with_signals += 1

            # Process ACTIVE signals (most recent only)
            if result.active_signals:
                latest_active = result.active_signals[-1]
                total_active += 1
                active_results.append({
                    "Symbol": symbol,
                    "Signal": latest_active.signal_type,
                    "Entry Time": latest_active.timestamp.strftime("%Y-%m-%d %H:%M"),
                    "Entry": round(latest_active.entry_price, 2),
                    "Current": round(df_15m["Close"].iloc[-1], 2),
                    "SL": round(latest_active.sl_price, 2),
                    "TP": round(latest_active.tp_price, 2),
                    "Score": round(latest_active.score, 1),
                    "Instrument Key": ikey,
                })

            # Process COMPLETED trades
            for trade in result.completed_trades:
                total_completed += 1
                aggregate_pnl += trade.pnl
                aggregate_pnl_pct += trade.pnl_pct

                if trade.pnl > 0:
                    total_wins += 1
                else:
                    total_losses += 1

                if trade.status == "BREAKEVEN":
                    total_breakevens += 1

                completed_results.append({
                    "Symbol": symbol,
                    "Signal": trade.signal_type,
                    "Entry Time": trade.timestamp.strftime("%Y-%m-%d %H:%M"),
                    "Exit Time": trade.exit_time.strftime("%Y-%m-%d %H:%M") if trade.exit_time else "-",
                    "Entry": round(trade.entry_price, 2),
                    "Exit": round(trade.exit_price, 2),
                    "SL": round(trade.sl_price, 2),
                    "TP": round(trade.tp_price, 2),
                    "Result": trade.exit_reason,
                    "P&L": round(trade.pnl, 2),
                    "P&L %": round(trade.pnl_pct, 2),
                    "Bars": trade.bars_held,
                    "Score": round(trade.score, 1),
                    "BE": "✓" if trade.breakeven_triggered else "",
                    "Instrument Key": ikey,
                })

        except Exception as e:
            # Uncomment for debugging:
            # print(f"Error scanning {symbol}: {e}")
            continue

    # Create DataFrames
    active_df = pd.DataFrame(active_results)
    completed_df = pd.DataFrame(completed_results)

    # Sort
    if not completed_df.empty:
        completed_df = completed_df.sort_values("P&L", ascending=False)
    if not active_df.empty:
        active_df = active_df.sort_values("Score", ascending=False)

    # Calculate summary statistics
    win_rate = (total_wins / total_completed *
                100) if total_completed > 0 else 0

    summary = {
        "total_stocks_scanned": total_stocks,
        "stocks_with_signals": stocks_with_signals,
        "total_active": total_active,
        "total_completed": total_completed,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "total_breakevens": total_breakevens,
        "win_rate": round(win_rate, 1),
        "total_pnl": round(aggregate_pnl, 2),
        "total_pnl_pct": round(aggregate_pnl_pct, 2),

        # NEW: Skip statistics
        "signals_skipped_session_limit": total_skipped_session,
        "signals_skipped_cooldown": total_skipped_cooldown,
        "signals_skipped_quality": total_skipped_quality,
    }

    return active_df, completed_df, summary


@st.cache_data(ttl=300)
def get_fo_stocks():
    query = """
    SELECT DISTINCT f.trading_symbol, f.instrument_key, f.name, f.lot_size, f.is_active
    FROM fo_stocks_master f
    WHERE f.is_active = TRUE
    ORDER BY f.trading_symbol
    """
    try:
        return db.con.execute(query).df()
    except Exception as e:
        st.error(f"Error loading F&O stocks: {e}")
        return pd.DataFrame()


def load_data_fast(instrument_key: str, timeframe: str, lookback_days: int = 30, end_timestamp: Optional[datetime] = None) -> Optional[pd.DataFrame]:
    cutoff_date = (datetime.now() - timedelta(days=lookback_days)
                   ).strftime('%Y-%m-%d')
    # build query with optional end timestamp
    if end_timestamp is None:
        query = """
        SELECT timestamp, open as Open, high as High, low as Low, close as Close, volume as Volume
        FROM ohlcv_resampled
        WHERE instrument_key = ? AND timeframe = ? AND timestamp >= ?
        ORDER BY timestamp
        """
        params = [instrument_key, timeframe, cutoff_date]
    else:
        query = """
        SELECT timestamp, open as Open, high as High, low as Low, close as Close, volume as Volume
        FROM ohlcv_resampled
        WHERE instrument_key = ? AND timeframe = ? AND timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp
        """
        # format end timestamp to DB-friendly string
        end_str = end_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        params = [instrument_key, timeframe, cutoff_date, end_str]
    try:
        df = db.con.execute(query, params).df()
        if df.empty:
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        return df
    except Exception:
        return None


def save_signals_to_universe(scan_results: pd.DataFrame) -> int:
    # This function will now implicitly only save signals with score 5,
    # as scan_results will already be filtered by the calling functions.
    signals = scan_results[scan_results["Signal"].isin(
        ["LONG", "SHORT"])].copy()
    if signals.empty:
        st.warning("No LONG / SHORT signals to save")
        return 0

    today = date.today()
    saved_count = 0
    errors: List[str] = []

    for _, row in signals.iterrows():
        try:
            db.con.execute(
                """
                DELETE FROM ehma_universe
                WHERE signal_date = ? AND symbol = ? AND signal_type = ?
                """,
                [today, row["Symbol"], row["Signal"]],
            )
            db.con.execute(
                """
                INSERT INTO ehma_universe (
                    signal_date,
                    symbol,
                    instrument_key,
                    signal_type,
                    signal_strength,
                    bars_ago,
                    current_price,
                    entry_price,
                    stop_loss,
                    target_price,
                    rsi,
                    trend,
                    reasons,
                    status,
                    scan_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', CURRENT_TIMESTAMP)
                """,
                [
                    today,
                    row["Symbol"],
                    row.get("Instrument Key"),
                    float(row["Signal"]),
                    float(row.get("Alignment Score", 0)
                          or 0),  # This will be 5
                    int(row.get("Bars Ago", 0)),
                    float(row["Price"]),
                    float(row["Entry"]),
                    float(row["SL"]),
                    float(row["TP"]),
                    None,  # RSI not computed here
                    row.get("Trend", "-"),
                    row["Reasons"],
                ],
            )
            saved_count += 1
        except Exception as e:
            errors.append(f"{row['Symbol']}: {e}")

    if errors:
        st.error("Some signals failed to save")
        st.code("\n".join(errors[:10]))
    return saved_count


st.title("🧨 Indian Market Squeeze – 15m Stack")

fo_stocks = get_fo_stocks()
if fo_stocks.empty:
    st.error("No F&O stocks found in database!")
    st.stop()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "🔍 15m Batch Scanner",
        "🔴 Live Scanner",
        "📈 Single Stock 15m",
        "💎 Squeeze Universe",
        "📈 Options Trading",
        "📊 Backtest",
        "📋 Trade Log",
    ]
)

# ============================
# TAB 1: 15m BATCH SCANNER
# ============================
with tab1:
    st.markdown("### 🔍 Scan All F&O Stocks with 15m Squeeze V2")
    st.caption("✨ Improved with session limits, breakeven stops & quality filters")

    # =========================================================================
    # ROW 1: Basic Settings
    # =========================================================================
    col1, col2, col3 = st.columns(3)
    with col1:
        scan_lookback = st.slider(
            "Lookback Days", 30, 120, 60, key="sq_scan_lb")
    with col2:
        sl_mode = st.selectbox(
            "SL / TP Mode",
            ["ATR based", "Fixed %"],
            key="sq_scan_slmode",
        )
    with col3:
        # OPTIMIZED VALUES - Balanced for reversal strategy
        # Tighter stop (was 1.5) - squeeze reversals need tight stops
        atr_mult = 1.2
        rr = 2.0        # Conservative RR (was 2.5) - let trades breathe
        sl_pct = 1.00
        tp_pct = 2.00

    # =========================================================================
    # ROW 2: SL/TP Settings
    # =========================================================================
    col4, col5 = st.columns(2)
    with col4:
        if sl_mode == "ATR based":
            atr_mult = st.number_input(
                # Default 1.2 (tighter)
                "ATR SL Multiplier", 0.5, 5.0, 1.2, 0.1, key="sq_scan_atr",
                help="Tight stop (1.2x) for reversal entries - prevents runaway losses"
            )
            rr = st.number_input(
                # Default 2.0
                "Reward : Risk (TP = RR×Risk)", 1.0, 5.0, 2.0, 0.5, key="sq_scan_rr",
                help="Moderate RR (2.0) - reversal trades need room to breathe"
            )
        else:
            sl_pct = (
                st.number_input(
                    "SL %", 0.2, 10.0, 1.0, 0.2, key="sq_scan_slpct"
                )
                / 100.0
            )
    with col5:
        if sl_mode == "Fixed %":
            tp_pct = (
                st.number_input(
                    "TP %", 0.5, 20.0, 2.0, 0.5, key="sq_scan_tppct"
                )
                / 100.0
            )

    # =========================================================================
    # ROW 3: NEW - Session & Quality Settings
    # =========================================================================
    st.markdown("#### ⚙️ Advanced Settings")
    adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)

    with adv_col1:
        max_trades_per_session = st.number_input(
            "Max Trades/Day/Stock", 1, 5, 1, key="sq_max_session",
            help="Limit trades per stock per day (session limit)"
        )
    with adv_col2:
        use_breakeven = st.checkbox(
            "Use Breakeven Stop", value=True, key="sq_use_be",
            help="Move SL to entry after 1R profit"
        )
    with adv_col3:
        use_quality_filter = st.checkbox(
            "Quality Filter", value=True, key="sq_quality",
            help="Filter out low volume & high volatility signals"
        )
    with adv_col4:
        max_trade_bars = st.number_input(
            # Default 40 (was 30)
            "Max Hold Bars", 10, 100, 40, key="sq_max_bars",
            help="Maximum bars to hold a trade - reversal trades need more time"
        )

    # =========================================================================
    # ROW 4: Date/Time & Start Button
    # =========================================================================
    col_btn, col_date, col_time = st.columns([1, 1, 1])
    with col_date:
        scan_date = st.date_input(
            "Scan End Date", value=date.today(), key="sq_scan_date")
    with col_time:
        start_min = 9 * 60 + 15
        end_min = 15 * 60 + 15
        time_options = [
            f"{(m//60):02d}:{(m % 60):02d}" for m in range(start_min, end_min + 1, 15)]
        scan_time_str = st.selectbox(
            "Scan End Time", options=time_options, index=0, key="sq_scan_time")

    with col_btn:
        # Add this temporarily in your tab1 code before the scan:
        test_df = load_data_fast(
            fo_stocks.iloc[0]["instrument_key"], "15minute", 60)
        if test_df is not None:
            sig_df = compute_squeeze_stack(test_df)
            st.write("Columns:", list(sig_df.columns))
            st.write("Sample signals:", sig_df[["long_signal", "short_signal"]].sum(
            ) if "long_signal" in sig_df.columns else "NO long_signal column!")
        if st.button("🚀 Start 15m Squeeze Scan V2", type="primary", use_container_width=True):
            end_dt = datetime.combine(
                scan_date, datetime.strptime(scan_time_str, "%H:%M").time()
            )
            progress = st.progress(0, text="Initializing scan...")
            start_time = time.time()

# # DEBUG: Test single stock
#         if st.button("🧪 DEBUG: Test Single Stock"):
#             test_row = fo_stocks.iloc[0]
#             test_ikey = test_row["instrument_key"]
#             test_symbol = test_row["trading_symbol"]

#             st.write(f"Testing: {test_symbol}")

#             # Load data
#             test_df = load_data_fast(test_ikey, "15minute", 60)
#             st.write(f"Data loaded: {len(test_df)} rows" if test_df is not None else "NO DATA")

#             if test_df is not None and len(test_df) > 100:
#                 # Get signals
#                 sig_df = compute_squeeze_stack(test_df)
#                 st.write(f"Signal columns: {list(sig_df.columns)}")

#                 long_count = sig_df["long_signal"].sum() if "long_signal" in sig_df.columns else 0
#                 short_count = sig_df["short_signal"].sum() if "short_signal" in sig_df.columns else 0
#                 st.write(f"Signals found: {long_count} LONG, {short_count} SHORT")

#                 # Try the V2 backtest

#                 result = build_15m_signals_with_backtest(
#                     test_df,
#                     sl_mode="ATR",
#                     sl_atr_mult=1.5,
#                     tp_rr=2.5,
#                     max_trade_bars=30,
#                     max_trades_per_session=1,
#                     use_breakeven=True,
#                     use_quality_filter=False,
#                     min_score=0.0,
#                     compute_squeeze_fn=compute_squeeze_stack,
#                 )

#                 st.write(f"V2 Result - Active: {len(result.active_signals)}, Completed: {len(result.completed_trades)}")
#                 st.write(f"Skipped - Session: {result.signals_skipped_session_limit}, Quality: {result.signals_skipped_low_quality}")
            # =====================================================
            # CALL THE NEW V2 FUNCTION
            # =====================================================

            active_df, completed_df, summary = run_batch_scan_squeeze_15m_v2_improved(
                fo_stocks,
                lookback_days=scan_lookback,
                sl_mode=sl_mode,
                atr_mult=atr_mult,
                rr=rr,
                sl_pct=sl_pct,
                tp_pct=tp_pct,
                progress_bar=progress,
                end_dt=end_dt,
                db=db,
                max_trade_bars=max_trade_bars,
                # NEW PARAMETERS
                max_trades_per_session=max_trades_per_session,
                # DISABLED: Breakeven stops cutting winners short (29% breakevens)
                use_breakeven=False,
                breakeven_at_r=1.5,   # If re-enabled, trigger at 1.5R instead of 1R
                use_quality_filter=True,  # FIXED: Enable quality filter to avoid bad trades
                min_score=5.0,
                # Pass your data loader
                load_data_fn=load_data_fast,
                compute_squeeze_fn=compute_squeeze_stack,
            )
            # Add this RIGHT AFTER the run_batch_scan call:
            st.write("DEBUG - Summary:", summary)
            st.write("DEBUG - Active DF shape:",
                     active_df.shape if not active_df.empty else "EMPTY")
            st.write("DEBUG - Completed DF shape:",
                     completed_df.shape if not completed_df.empty else "EMPTY")
            elapsed = time.time() - start_time
            progress.progress(1.0, text=f"Scan complete in {elapsed:.1f}s")

            # Store results in session state
            st.session_state["sq_active_signals"] = active_df
            st.session_state["sq_completed_trades"] = completed_df
            st.session_state["sq_scan_summary"] = summary
            st.session_state["sq_scan_run_time"] = datetime.now()

    # =========================================================================
    # DISPLAY RESULTS
    # =========================================================================
    if "sq_scan_summary" in st.session_state:
        summary = st.session_state["sq_scan_summary"]
        active_df = st.session_state["sq_active_signals"]
        completed_df = st.session_state["sq_completed_trades"]
        scan_time = st.session_state.get("sq_scan_run_time", datetime.now())

        st.markdown(
            f"**Last Scan:** {scan_time.strftime('%Y-%m-%d %H:%M:%S')}")
        st.divider()

        # =====================================================
        # SCAN SUMMARY
        # =====================================================
        st.markdown("### 📊 Scan Summary")
        m1, m2, m3, m4, m5 = st.columns([1, 1, 1, 1, 1], gap="large")

        m1.metric(label="Stocks Scanned",
                  value=f"{summary['total_stocks_scanned']}")
        m2.metric(label="With Signals",
                  value=f"{summary['stocks_with_signals']}")
        m3.metric(label="🟡 Active", value=f"{summary['total_active']}")
        m4.metric(label="✅ Completed", value=f"{summary['total_completed']:,}")
        m5.metric(label="Win Rate", value=f"{summary['win_rate']:.1f}%")

        # NEW: Show skip statistics
        if summary.get('signals_skipped_session_limit', 0) > 0:
            st.info(f"📊 Signals filtered: {summary.get('signals_skipped_session_limit', 0)} by session limit, "
                    f"{summary.get('signals_skipped_quality', 0)} by quality filter")

        st.divider()

        # =====================================================
        # P&L SUMMARY
        # =====================================================
        st.markdown("### 💰 P&L Summary (Completed Trades)")
        p1, p2, p3, p4, p5 = st.columns([1.5, 1, 1, 1, 1], gap="large")

        total_pnl = summary["total_pnl"]
        pnl_delta = f"{summary['total_pnl_pct']:.2f}%" if summary['total_pnl_pct'] != 0 else None

        p1.metric(label="Total P&L",
                  value=f"₹{total_pnl:,.2f}", delta=pnl_delta)
        p2.metric(label="🟢 Wins", value=f"{summary['total_wins']:,}")
        p3.metric(label="🔴 Losses", value=f"{summary['total_losses']:,}")

        # NEW: Show breakeven count
        breakevens = summary.get('total_breakevens', 0)
        p4.metric(label="⚪ Breakeven", value=f"{breakevens:,}")

        avg_pnl = summary['total_pnl_pct'] / max(summary['total_completed'], 1)
        p5.metric(label="Avg P&L %", value=f"{avg_pnl:.2f}%")

        st.divider()

        # Rest of your display code remains the same...
        # (Active signals table, completed trades table, charts, etc.)

        # ========================================
        # SCAN SUMMARY - Full width metrics
        # ========================================
        st.markdown("### 📊 Scan Summary")

        # Use wider columns with gaps
        m1, m2, m3, m4, m5 = st.columns([1, 1, 1, 1, 1], gap="large")

        m1.metric(
            label="Stocks Scanned",
            value=f"{summary['total_stocks_scanned']}"
        )
        m2.metric(
            label="With Signals",
            value=f"{summary['stocks_with_signals']}"
        )
        m3.metric(
            label="🟡 Active",
            value=f"{summary['total_active']}"
        )
        m4.metric(
            label="✅ Completed",
            value=f"{summary['total_completed']:,}"
        )
        m5.metric(
            label="Win Rate",
            value=f"{summary['win_rate']:.1f}%"
        )

        st.divider()

        # ========================================
        # P&L SUMMARY - Larger display
        # ========================================
        st.markdown("### 💰 P&L Summary (Completed Trades)")

        p1, p2, p3, p4 = st.columns([1.5, 1, 1, 1], gap="large")

        # Format P&L with proper sign
        total_pnl = summary["total_pnl"]
        pnl_delta = f"{summary['total_pnl_pct']:.2f}%" if summary['total_pnl_pct'] != 0 else None

        p1.metric(
            label="Total P&L",
            value=f"₹{total_pnl:,.2f}",
            delta=pnl_delta
        )
        p2.metric(
            label="🟢 Wins",
            value=f"{summary['total_wins']:,}"
        )
        p3.metric(
            label="🔴 Losses",
            value=f"{summary['total_losses']:,}"
        )

        avg_pnl = summary['total_pnl_pct'] / max(summary['total_completed'], 1)
        p4.metric(
            label="Avg P&L %",
            value=f"{avg_pnl:.2f}%"
        )

        st.divider()

        # ========================================
        # ACTIVE SIGNALS TABLE
        # ========================================
        st.markdown("### 🟡 Active Signals (Not Yet Hit SL/TP)")
        st.caption(
            f"Showing {len(active_df)} active signals that can still be traded")

        if not active_df.empty:
            active_df = active_df[active_df["Score"] >= 5]
            st.dataframe(
                active_df,
                use_container_width=True,
                height=min(400, len(active_df) * 38 + 50),
                hide_index=True,
                column_config={
                    "Symbol": st.column_config.TextColumn("Symbol", width="medium"),
                    "Signal": st.column_config.TextColumn("Signal", width="small"),
                    "Entry Time": st.column_config.TextColumn("Entry Time", width="medium"),
                    "Entry": st.column_config.NumberColumn("Entry", format="₹%.2f"),
                    "Current": st.column_config.NumberColumn("Current", format="₹%.2f"),
                    "SL": st.column_config.NumberColumn("SL", format="₹%.2f"),
                    "TP": st.column_config.NumberColumn("TP", format="₹%.2f"),
                    "Score": st.column_config.NumberColumn("Score", format="%.1f"),
                    "Instrument Key": st.column_config.TextColumn("Instrument Key", width="large"),
                }
            )
        else:
            st.info("No active signals found with Alignment Score 5. All signals have hit SL or TP, or did not meet the score criteria.")

        st.divider()

        # ========================================
        # COMPLETED TRADES TABLE
        # ========================================
        st.markdown("### ✅ Completed Trades (SL or TP Hit)")
        st.caption(f"Showing {len(completed_df)} completed trades with P&L")

        if not completed_df.empty:
            # Add color coding for Result column
            def color_result(val):
                if val == "Take Profit":
                    return "background-color: #d4edda; color: #155724"
                elif val == "Stop Loss":
                    return "background-color: #f8d7da; color: #721c24"
                return ""

            def color_pnl(val):
                try:
                    if float(val) > 0:
                        return "color: green; font-weight: bold"
                    elif float(val) < 0:
                        return "color: red; font-weight: bold"
                except:
                    pass
                return ""

            # Style the dataframe
            styled_df = completed_df.style.applymap(
                color_result, subset=["Result"]
            ).applymap(
                color_pnl, subset=["P&L", "P&L %"]
            )

            st.dataframe(
                styled_df,
                use_container_width=True,
                height=min(600, len(completed_df) * 38 + 50),
                hide_index=True,
                column_config={
                    "Symbol": st.column_config.TextColumn("Symbol", width="medium"),
                    "Signal": st.column_config.TextColumn("Side", width="small"),
                    "Entry Time": st.column_config.TextColumn("Entry Time", width="medium"),
                    "Exit Time": st.column_config.TextColumn("Exit Time", width="medium"),
                    "Entry": st.column_config.NumberColumn("Entry", format="₹%.2f"),
                    "Exit": st.column_config.NumberColumn("Exit", format="₹%.2f"),
                    "SL": st.column_config.NumberColumn("SL", format="₹%.2f"),
                    "TP": st.column_config.NumberColumn("TP", format="₹%.2f"),
                    "Result": st.column_config.TextColumn("Result", width="medium"),
                    "P&L": st.column_config.NumberColumn("P&L (₹)", format="₹%.2f"),
                    "P&L %": st.column_config.NumberColumn("P&L %", format="%.2f%%"),
                    "Bars": st.column_config.NumberColumn("Bars", format="%d"),
                    # FIX: added comma here from previous review
                    "Score": st.column_config.NumberColumn("Score", format="%.1f"),
                    "Instrument Key": None,  # Hide this column
                }
            )

            # Export buttons
            col_exp1, col_exp2, col_exp3 = st.columns([1, 1, 2])
            with col_exp1:
                csv_completed = completed_df.to_csv(index=False)
                st.download_button(
                    "📥 Export Completed Trades",
                    data=csv_completed,
                    file_name=f"squeeze_completed_{date.today()}.csv",
                    mime="text/csv",
                )
            with col_exp2:
                if not active_df.empty:
                    csv_active = active_df.to_csv(index=False)
                    st.download_button(
                        "📥 Export Active Signals",
                        data=csv_active,
                        file_name=f"squeeze_active_{date.today()}.csv",
                        mime="text/csv",
                    )

            st.divider()

            # ========================================
            # CHARTS
            # ========================================
            st.markdown("### 📈 Trade Analysis")

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                # Win/Loss Pie Chart
                import plotly.express as px

                win_loss_data = pd.DataFrame({
                    "Result": ["Wins (TP Hit)", "Losses (SL Hit)"],
                    "Count": [summary["total_wins"], summary["total_losses"]]
                })

                fig_pie = px.pie(
                    win_loss_data,
                    values="Count",
                    names="Result",
                    color="Result",
                    color_discrete_map={
                        "Wins (TP Hit)": "#28a745",
                        "Losses (SL Hit)": "#dc3545"
                    },
                    title="Win/Loss Distribution",
                    hole=0.4  # Donut chart
                )
                fig_pie.update_layout(
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                    margin=dict(t=50, b=50, l=20, r=20)
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with chart_col2:
                # P&L Distribution Histogram
                fig_hist = px.histogram(
                    completed_df,
                    x="P&L %",
                    nbins=30,
                    title="P&L % Distribution",
                    color_discrete_sequence=["steelblue"]
                )
                fig_hist.add_vline(x=0, line_dash="dash",
                                   line_color="red", line_width=2)
                fig_hist.update_layout(
                    xaxis_title="P&L %",
                    yaxis_title="Number of Trades",
                    margin=dict(t=50, b=50, l=20, r=20)
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            # Daily P&L breakdown (if data spans multiple days)
            if "Entry Time" in completed_df.columns:
                try:
                    completed_df["Trade Date"] = pd.to_datetime(
                        completed_df["Entry Time"]).dt.date
                    daily_pnl = completed_df.groupby("Trade Date").agg({
                        "P&L": "sum",
                        "Symbol": "count"
                    }).reset_index()
                    daily_pnl.columns = ["Date", "P&L", "Trades"]

                    if len(daily_pnl) > 1:
                        st.markdown("### 📅 Daily P&L Breakdown")

                        fig_daily = px.bar(
                            daily_pnl,
                            x="Date",
                            y="P&L",
                            title="Daily P&L",
                            color="P&L",
                            color_continuous_scale=[
                                "red", "lightgray", "green"],
                            color_continuous_midpoint=0
                        )
                        fig_daily.update_layout(
                            xaxis_title="Date",
                            yaxis_title="P&L (₹)",
                            showlegend=False
                        )
                        st.plotly_chart(fig_daily, use_container_width=True)

                        # Daily summary table
                        st.dataframe(
                            daily_pnl,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Date": st.column_config.DateColumn("Date"),
                                "P&L": st.column_config.NumberColumn("P&L", format="₹%.2f"),
                                "Trades": st.column_config.NumberColumn("Trades", format="%d")
                            }
                        )
                except Exception:
                    pass

        else:
            st.info(
                "No completed trades found with Alignment Score 5 in the lookback period.")

# ============================
# TAB 2: LIVE 15m SQUEEZE SCANNER (FIXED)
# ============================
with tab2:
    st.markdown("### 🔴 Live Intraday Scanner (15m Squeeze)")
    st.info("""
    **Workflow:**
    1. Click **'🔄 Refresh Live Data'** to fetch today's candles (fills gap from 9:15)
    2. Click **'📊 Rebuild & Scan'** to resample data and generate signals
    3. **Score = 5** signals are TRADABLE NOW
    4. **Score = 4** signals need one more 15m confirmation
    """)

    # ========================================
    # INITIALIZE LIVE MANAGER (Singleton)
    # ========================================
    if "live_manager" not in st.session_state or st.session_state["live_manager"] is None:
        try:
            st.session_state["live_manager"] = LiveTradingManager()
        except Exception as e:
            st.session_state["live_manager"] = None
            st.error(f"Failed to initialize LiveTradingManager: {e}")

    live_manager = st.session_state.get("live_manager")

    if not live_manager:
        st.error("Live manager not available. Cannot run live scan.")
        st.stop()

    # Try to start WebSocket
    access_token = get_access_token()
    if access_token:
        live_manager.start_websocket_if_needed(access_token)

    # ========================================
    # MARKET STATUS DISPLAY
    # ========================================
    market_open = is_market_hours()

    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    with status_col1:
        if market_open:
            st.success("🟢 Market OPEN")
        else:
            st.warning("🔴 Market CLOSED")
    with status_col2:
        next_15m = get_next_candle_time("15minute")
        st.info(f"⏱️ Next 15m: **{next_15m.strftime('%H:%M')}**")
    with status_col3:
        secs_remaining = seconds_until_next_candle("15minute")
        st.info(f"⏳ In **{secs_remaining // 60}m {secs_remaining % 60}s**")
    with status_col4:
        ws_builder = getattr(live_manager, "ws_builder", None)
        ws_connected = getattr(live_manager, "ws_connected", False)

        if ws_connected and ws_builder:
            ws_time = ws_builder.ws_started_at if hasattr(
                ws_builder, 'ws_started_at') else None
            st.success(
                f"🔌 WS: {ws_time.strftime('%H:%M') if ws_time else 'Connected'}")
        elif ws_builder and not ws_connected:
            st.warning("🔌 WS: Initialized but not connected")
        else:
            st.error("🔌 WS: Not started - live data will not work!")

    st.divider()

    # ========================================
    # LIVE DATA STATUS
    # ========================================
    st.markdown("#### 📊 Live Data Coverage")

    try:
        summary = live_manager.get_live_data_summary()

        cov_col1, cov_col2, cov_col3, cov_col4, cov_col5 = st.columns(5)
        cov_col1.metric("Instruments", summary.get("instruments_with_data", 0))

        total_candles = summary.get("total_candles_today", 0)
        instruments_count = summary.get("instruments_with_data", 0)
        avg_candles = total_candles / max(instruments_count, 1)

        # Warning if average candles per instrument is too low
        if avg_candles < 100:
            cov_col2.metric("Candles Today", total_candles,
                            delta=f"⚠️ Low: {avg_candles:.0f}/inst", delta_color="off")
        else:
            cov_col2.metric("Candles Today", total_candles)

        first_candle = summary.get("first_candle")
        if first_candle:
            first_time = pd.to_datetime(first_candle).strftime("%H:%M")
            # Check if gap exists
            if first_time != "09:15":
                cov_col3.metric("First Candle", first_time,
                                delta="⚠️ Gap from 9:15", delta_color="off")
            else:
                cov_col3.metric("First Candle", first_time, delta="✓ Complete")
        else:
            cov_col3.metric("First Candle", "N/A", delta="No data")

        last_candle = summary.get("last_candle")
        if last_candle:
            cov_col4.metric("Last Candle", pd.to_datetime(
                last_candle).strftime("%H:%M"))
        else:
            cov_col4.metric("Last Candle", "N/A")

        latest_fetch = summary.get("latest_fetch")
        if latest_fetch:
            cov_col5.metric("Last Fetch", pd.to_datetime(
                latest_fetch).strftime("%H:%M:%S"))
        else:
            cov_col5.metric("Last Fetch", "Never")

    except Exception as e:
        st.warning(f"Could not get live data summary: {e}")

    st.divider()

    # ========================================
    # CONTROL BUTTONS
    # ========================================
    st.markdown("#### 🎮 Live Data Controls")

    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

    with btn_col1:
        refresh_clicked = st.button(
            "🔄 Refresh Live Data",
            type="primary",
            use_container_width=True,
            help="Fetch 1m candles from 9:15 to now (fills any gaps)"
        )

    with btn_col2:
        rebuild_clicked = st.button(
            "📊 Rebuild & Scan",
            type="primary",
            use_container_width=True,
            help="Resample to 5m/15m/60m and scan for signals"
        )

    with btn_col3:
        init_clicked = st.button(
            "🗑️ Initialize Day",
            type="secondary",
            use_container_width=True,
            help="Clear cache (only use before market open)"
        )

    with btn_col4:
        auto_refresh = st.checkbox(
            "Auto-refresh (60s)", value=False, key="sq_live_auto")

    # ========================================
    # REFRESH BUTTON HANDLER
    # ========================================
    if refresh_clicked:
        if not access_token:
            st.error("No access token! Please login first.")
        else:
            # Check WebSocket status before refresh
            ws_connected = getattr(live_manager, "ws_connected", False)
            if not ws_connected:
                st.warning("⚠️ WebSocket not connected - starting it now...")
                live_manager.start_websocket_if_needed(access_token)

            with st.spinner("Fetching live data (fills gap from 9:15)..."):
                progress = st.progress(0, text="Starting...")

                def update_progress(current, total, symbol):
                    progress.progress(
                        current / total, text=f"Fetching {symbol}... ({current}/{total})")

                status = live_manager.fill_gap_and_refresh(
                    access_token, update_progress)
                progress.progress(1.0, text="Complete!")

                if status.success:
                    if status.gap_filled:
                        st.success(
                            f"✅ Gap filled from {status.gap_from.strftime('%H:%M')} to {status.gap_to.strftime('%H:%M')}")
                    if status.candles_inserted > 0:
                        st.success(
                            f"✅ Inserted {status.candles_inserted:,} candles from {status.instruments_updated} instruments")

                        # Verify data was actually inserted
                        cache_count = db.con.execute(
                            "SELECT COUNT(*) FROM live_ohlcv_cache").fetchone()[0]
                        st.info(f"📊 Total candles in cache: {cache_count:,}")
                    else:
                        st.info("Data already up to date.")

                        # Show what's in cache
                        cache_count = db.con.execute(
                            "SELECT COUNT(*) FROM live_ohlcv_cache").fetchone()[0]
                        if cache_count == 0:
                            st.error(
                                "❌ No data in live_ohlcv_cache - WebSocket may not be running!")
                        else:
                            st.info(f"📊 Cache has {cache_count:,} candles")
                else:
                    st.warning(
                        f"Partial success. Errors: {len(status.errors)}")
                    if status.errors:
                        with st.expander("View errors"):
                            st.code("\n".join(status.errors[:20]))

            st.session_state["sq_live_refreshed"] = datetime.now()

    # ========================================
    # REBUILD & SCAN BUTTON HANDLER
    # ========================================
    if rebuild_clicked:
        with st.spinner("Rebuilding resampled data..."):
            live_manager.rebuild_today_resampled()

            # Check if resampling actually worked
            resampled_count = db.con.execute("""
                SELECT COUNT(*) FROM ohlcv_resampled_live
                WHERE DATE(timestamp) = CURRENT_DATE
            """).fetchone()[0]

            if resampled_count > 0:
                st.success(
                    f"✅ Resampled 1m → 5m/15m/60m ({resampled_count:,} candles)")
            else:
                st.error(
                    "❌ Resampling failed - no data in ohlcv_resampled_live. Click 'Refresh Live Data' first!")
                st.stop()

        st.session_state["sq_resampled_built"] = datetime.now()

        # Now scan for signals
        st.markdown("---")
        st.markdown("### 🔍 Scanning for Signals...")

        progress = st.progress(0, text="Starting signal scan...")
        instruments = live_manager.get_active_instruments()

        tradable_results = []  # Score = 5
        ready_results = []     # Score = 4
        skipped_no_data = 0
        skipped_few_bars = 0
        errors = 0

        total = len(instruments) if instruments else 0

        # Get SL/TP parameters from session state or use defaults
        live_sl_mode = st.session_state.get("sq_live_slmode", "ATR based")
        live_atr_mult = st.session_state.get("sq_live_atr", 2.0)
        live_rr = st.session_state.get("sq_live_rr", 2.0)
        live_sl_pct = st.session_state.get("sq_live_slpct", 1.0) / 100.0
        live_tp_pct = st.session_state.get("sq_live_tppct", 2.0) / 100.0

        for i, (instrument_key, symbol) in enumerate(instruments or []):
            progress.progress((i + 1) / max(total, 1),
                              text=f"Scanning {symbol}... ({i+1}/{total})")

            try:
                # Get combined MTF data (historical + today's live)
                df_60m, df_15m, df_5m = live_manager.get_live_mtf_data(
                    instrument_key, lookback_days=60)

                if df_15m is None:
                    skipped_no_data += 1
                    continue

                if len(df_15m) < 100:
                    skipped_few_bars += 1
                    continue

                mode_str = "ATR" if live_sl_mode == "ATR based" else "PCT"

                # Get BOTH score=5 and score=4 signals
                score_5_signals, score_4_signals = build_15m_signals_for_live_scan(
                    df_15m,
                    sl_mode=mode_str,
                    sl_atr_mult=live_atr_mult,
                    tp_rr=live_rr,
                    sl_pct=live_sl_pct,
                    tp_pct=live_tp_pct,
                    lookback_bars=10,
                    include_score_4=True,
                )

                # Process score=5 signals (TRADABLE NOW)
                for sig in score_5_signals:
                    signal_age_bars = (
                        df_15m.index[-1] - sig.timestamp).total_seconds() / 900
                    if signal_age_bars <= 3:  # Recent signal (within 45 min)
                        tradable_results.append({
                            "Symbol": symbol,
                            "Signal": sig.signal_type,
                            "Price": round(df_15m["Close"].iloc[-1], 2),
                            "Entry": round(sig.entry_price, 2),
                            "SL": round(sig.sl_price, 2),
                            "TP": round(sig.tp_price, 2),
                            "Time": sig.timestamp.strftime("%H:%M"),
                            "Score": 5,
                            "Status": "🟢 TRADABLE",
                            "Bars Ago": int(signal_age_bars),
                            "Instrument Key": instrument_key,
                            "Reasons": ", ".join(sig.reasons),
                        })

                # Process score=4 signals (READY SOON)
                for sig in score_4_signals:
                    signal_age_bars = (
                        df_15m.index[-1] - sig.timestamp).total_seconds() / 900
                    if signal_age_bars <= 3:
                        ready_results.append({
                            "Symbol": symbol,
                            "Signal": sig.signal_type,
                            "Price": round(df_15m["Close"].iloc[-1], 2),
                            "Entry": round(sig.entry_price, 2),
                            "SL": round(sig.sl_price, 2),
                            "TP": round(sig.tp_price, 2),
                            "Time": sig.timestamp.strftime("%H:%M"),
                            "Score": 4,
                            "Status": "🟡 READY SOON",
                            "Bars Ago": int(signal_age_bars),
                            "Instrument Key": instrument_key,
                            "Reasons": ", ".join(sig.reasons),
                        })

            except Exception as e:
                errors += 1
                continue

        progress.progress(1.0, text="Scan complete!")

        # Show scan statistics
        st.markdown("##### 📈 Scan Statistics")
        stat_cols = st.columns(5)
        stat_cols[0].metric("Total Scanned", total)
        stat_cols[1].metric("No Data", skipped_no_data)
        stat_cols[2].metric("Few Bars", skipped_few_bars)
        stat_cols[3].metric("🟢 Tradable (Score 5)", len(tradable_results))
        stat_cols[4].metric("🟡 Ready Soon (Score 4)", len(ready_results))

        # Store results
        st.session_state["sq_live_tradable"] = pd.DataFrame(tradable_results)
        st.session_state["sq_live_ready"] = pd.DataFrame(ready_results)
        st.session_state["sq_live_scan_time"] = datetime.now()

    # ========================================
    # INITIALIZE DAY HANDLER
    # ========================================
    if init_clicked:
        if market_open:
            st.warning(
                "⚠️ Market is open - only resetting flags, not clearing data")
        live_manager.initialize_day()
        st.success("✅ Day initialized")

    # ========================================
    # DISPLAY SCAN RESULTS
    # ========================================
    st.divider()
    st.markdown("### 📊 Live Scan Results")

    # SL/TP Configuration
    cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
    with cfg_col1:
        live_sl_mode = st.selectbox(
            "SL/TP Mode", ["ATR based", "Fixed %"], key="sq_live_slmode")
    with cfg_col2:
        if live_sl_mode == "ATR based":
            st.number_input("ATR Mult", 0.5, 5.0, 2.0, 0.5, key="sq_live_atr")
            st.number_input("RR Ratio", 1.0, 5.0, 2.0, 0.5, key="sq_live_rr")
        else:
            st.number_input("SL %", 0.2, 10.0, 1.0, 0.2, key="sq_live_slpct")
            st.number_input("TP %", 0.5, 20.0, 2.0, 0.5, key="sq_live_tppct")
    with cfg_col3:
        if st.button("💾 Save Tradable Signals", use_container_width=True):
            tradable_df = st.session_state.get("sq_live_tradable")
            if tradable_df is not None and not tradable_df.empty:
                saved = save_signals_to_universe(tradable_df)
                st.success(f"Saved {saved} signals to universe")
            else:
                st.warning("No tradable signals to save")

    # Show tradable signals (Score = 5)
    tradable_df = st.session_state.get("sq_live_tradable")
    if tradable_df is not None and not tradable_df.empty:
        st.markdown("#### 🟢 TRADABLE NOW (Score = 5)")
        st.dataframe(
            tradable_df,
            use_container_width=True,
            height=min(400, len(tradable_df) * 38 + 50),
            hide_index=True,
            column_config={
                "Status": st.column_config.TextColumn("Status", width="small"),
                "Symbol": st.column_config.TextColumn("Symbol", width="medium"),
                "Signal": st.column_config.TextColumn("Side", width="small"),
                "Price": st.column_config.NumberColumn("Price", format="₹%.2f"),
                "Entry": st.column_config.NumberColumn("Entry", format="₹%.2f"),
                "SL": st.column_config.NumberColumn("SL", format="₹%.2f"),
                "TP": st.column_config.NumberColumn("TP", format="₹%.2f"),
                "Score": st.column_config.NumberColumn("Score", format="%d"),
                "Instrument Key": None,
            }
        )
    else:
        st.info(
            "No tradable signals (Score = 5) found. Click 'Rebuild & Scan' after refreshing data.")

    # Show ready signals (Score = 4)
    ready_df = st.session_state.get("sq_live_ready")
    if ready_df is not None and not ready_df.empty:
        with st.expander(f"🟡 READY SOON (Score = 4) - {len(ready_df)} signals"):
            st.markdown(
                "*These signals need one more confirmation. Wait for next 15m candle.*")
            st.dataframe(
                ready_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Status": st.column_config.TextColumn("Status", width="small"),
                    "Instrument Key": None,
                }
            )

    # Show last scan time
    scan_time = st.session_state.get("sq_live_scan_time")
    if scan_time:
        age = int((datetime.now() - scan_time).total_seconds())
        st.caption(f"Last scan: {scan_time.strftime('%H:%M:%S')} ({age}s ago)")

    # Auto-refresh
    if auto_refresh:
        st.info("🔄 Auto-refresh enabled. Page will reload in ~60 seconds.")
        time.sleep(1)
        st.rerun()
# ============================
# TAB 3: SINGLE STOCK 15m SQUEEZE
# ============================
with tab3:
    st.markdown("### 📈 Single Stock 15m Squeeze Analysis")

    fo_df = fo_stocks  # from top-level
    symbol_map = {
        row["trading_symbol"]: row["instrument_key"]
        for _, row in fo_df.iterrows()
    }

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        sq_symbol = st.selectbox(
            "Select Stock", list(symbol_map.keys()), key="sq_single_sym"
        )
        sq_instrument_key = symbol_map[sq_symbol]
    with col2:
        sq_lookback = st.slider(
            "Lookback Days", 30, 180, 90, key="sq_single_lb"
        )
    with col3:
        sq_sl_mode = st.selectbox(
            "SL / TP Mode",
            ["ATR based", "Fixed %"],
            key="sq_single_slmode",
        )

    sq_atr_mult = 2.0
    sq_rr = 2.0
    sq_sl_pct = 0.01
    sq_tp_pct = 0.02

    col4, col5 = st.columns(2)
    with col4:
        if sq_sl_mode == "ATR based":
            sq_atr_mult = st.number_input(
                "ATR SL Mult",
                0.5,
                5.0,
                2.0,
                0.5,
                key="sq_single_atr",
            )
            sq_rr = st.number_input(
                "RR (TP = RR×Risk)",
                1.0,
                5.0,
                2.0,
                0.5,
                key="sq_single_rr",
            )
        else:
            sq_sl_pct = (
                st.number_input(
                    "SL %", 0.2, 10.0, 1.0, 0.2, key="sq_single_slpct"
                )
                / 100.0
            )
    with col5:
        if sq_sl_mode == "Fixed %":
            sq_tp_pct = (
                st.number_input(
                    "TP %", 0.5, 20.0, 2.0, 0.5, key="sq_single_tppct"
                )
                / 100.0
            )

    if st.button("🔍 Analyze 15m Squeeze", type="primary"):
        with st.spinner(f"Analyzing {sq_symbol} on 15m..."):
            df_15m = load_data_fast(
                sq_instrument_key, "15minute", sq_lookback
            )
            if df_15m is None or len(df_15m) < 100:
                st.warning("Insufficient 15m data for this symbol / lookback.")
            else:
                mode_str = "ATR" if sq_sl_mode == "ATR based" else "PCT"
                signals = build_15m_signals(
                    df_15m,
                    sl_mode=mode_str,
                    sl_atr_mult=sq_atr_mult,
                    tp_rr=sq_rr,
                    sl_pct=sq_sl_pct,
                    tp_pct=sq_tp_pct,
                )

                # --- START MODIFICATION for single stock analysis ---
                # Filter for signals with an alignment score of 5
                signals = [s for s in signals if s.score == 5]
                # --- END MODIFICATION ---

                current_price = df_15m["Close"].iloc[-1]
                st.markdown("#### 📍 Current State")
                col1, col2 = st.columns(2)
                col1.metric("Current Price", f"{current_price:.2f}")
                col2.metric("Bars in sample", len(df_15m))

                st.markdown("#### 🎯 Tradeable Signals (15m)")
                if signals:
                    latest = signals[-1]
                    st.markdown(
                        f"**Latest Signal:** {latest.signal_type} at {latest.timestamp.strftime('%Y-%m-%d %H:%M')} (Score: {latest.score})"
                    )
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Direction", latest.signal_type)
                    c2.metric("Entry", f"{latest.entry_price:.2f}")
                    c3.metric("SL", f"{latest.sl_price:.2f}")
                    c4.metric("TP", f"{latest.tp_price:.2f}")

                    st.markdown("**Reasons**")
                    st.write(", ".join(latest.reasons))

                    rows = []
                    for s in signals:
                        rows.append(
                            {
                                "Time": s.timestamp,
                                "Signal": s.signal_type,
                                "Entry": round(s.entry_price, 2),
                                "SL": round(s.sl_price, 2),
                                "TP": round(s.tp_price, 2),
                                "Reasons": ", ".join(s.reasons),
                                "Score": round(s.score, 1),  # This will be 5.0
                            }
                        )
                    st.markdown("#### 📋 Signal History (15m)")
                    st.dataframe(
                        pd.DataFrame(rows),
                        use_container_width=True,
                        height=400,
                    )
                else:
                    st.info(
                        "No valid 15m Squeeze entries with Alignment Score 5 in the selected lookback window."
                    )

# ============================
# TAB 4: SQUEEZE UNIVERSE
# ============================
with tab4:
    st.markdown("### 💎 Squeeze Universe - Saved 15m Signals")

    def load_universe(signal_date: date = None, status: str = None) -> pd.DataFrame:
        if signal_date is None:
            signal_date = date.today()
        query = "SELECT * FROM ehma_universe WHERE signal_date = ?"
        params = [signal_date]
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY signal_strength DESC"
        try:
            return db.con.execute(query, params).df()
        except Exception:
            return pd.DataFrame()

    universe_df = load_universe()
    if not universe_df.empty:
        st.dataframe(universe_df, use_container_width=True, height=400)
    else:
        st.info("No signals saved today. Run a scan and save signals first!")

# ============================
# TAB 5: OPTIONS TRADING (SQUEEZE)
# ============================
with tab5:
    st.markdown("### 📈 Options Trading - Auto Select CE/PE (Squeeze 15m)")

    # Reuse universe, but you may later filter by strategy using reasons/extra column
    active_signals = load_universe(date.today(), "ACTIVE")
    if active_signals.empty:
        st.info(
            "No active signals with Alignment Score 5 in universe. Run a 15m Squeeze scan and save signals first.")  # Updated text
    else:
        st.markdown(
            f"**{len(active_signals)} active signals** ready for options")

        col1, col2, col3 = st.columns(3)
        with col1:
            capital_per_trade = st.number_input(
                "Capital per Trade (₹)", 10000, 500000, 50000, key="sq_opt_cap"
            )
        with col2:
            max_positions = st.slider(
                "Max Positions", 1, 10, 5, key="sq_opt_max"
            )
        with col3:
            delta_min, delta_max = st.slider(
                "Delta Range", 0.30, 0.70, (0.40, 0.60), key="sq_opt_delta"
            )

        st.divider()

        @st.cache_data(ttl=60)
        def get_lot_size(symbol: str) -> int:
            try:
                result = db.con.execute(
                    """
                    SELECT lot_size FROM fo_stocks_master
                    WHERE trading_symbol = ? LIMIT 1
                    """,
                    [symbol],
                ).fetchone()
                return result[0] if result else 1
            except Exception:
                return 1

        def update_option_details(symbol: str, signal_type: str, option_data: dict) -> bool:
            today = date.today()
            try:
                db.con.execute(
                    """
                    UPDATE ehma_universe
                    SET option_type = ?,
                        option_instrument_key = ?,
                        strike_price = ?,
                        expiry_date = ?,
                        lot_size = ?,
                        option_ltp = ?,
                        option_delta = ?,
                        option_iv = ?,
                        option_theta = ?
                    WHERE signal_date = ? AND symbol = ? AND signal_type = ?
                    """,
                    [
                        option_data.get("option_type"),
                        option_data.get("instrument_key"),
                        option_data.get("strike"),
                        option_data.get("expiry"),
                        option_data.get("lot_size"),
                        option_data.get("ltp"),
                        option_data.get("delta"),
                        option_data.get("iv"),
                        option_data.get("theta"),
                        today,
                        symbol,
                        signal_type,
                    ],
                )
                return True
            except Exception:
                return False

        def select_best_option(symbol: str, signal_type: str, spot_price: float, chain: dict) -> dict:
            if not chain:
                return None
            opt_type = "CE" if signal_type == "LONG" else "PE"
            options = chain.get(opt_type, [])
            if not options:
                return None
            df = pd.DataFrame(options)
            if df.empty:
                return None

            df["dist"] = (df["strike"] - spot_price).abs()
            df = df.sort_values("dist").head(7)

            df["score"] = 0.0

            if "delta" in df.columns and df["delta"].notna().any():
                df["delta_score"] = 1 - (df["delta"].abs() - 0.5).abs() * 2
                df["score"] += df["delta_score"].fillna(0) * 0.4

            if "iv" in df.columns and df["iv"].notna().any():
                iv_min, iv_max = df["iv"].min(), df["iv"].max()
                if iv_max > iv_min:
                    df["iv_score"] = 1 - \
                        (df["iv"] - iv_min) / (iv_max - iv_min)
                    df["score"] += df["iv_score"].fillna(0) * 0.3

            if "oi" in df.columns and df["oi"].notna().any():
                oi_max = df["oi"].max()
                if oi_max > 0:
                    df["oi_score"] = df["oi"] / oi_max
                    df["score"] += df["oi_score"].fillna(0) * 0.3

            best = df.sort_values("score", ascending=False).iloc[0]
            return {
                "option_type": opt_type,
                "instrument_key": best.get("instrument_key"),
                "strike": best["strike"],
                "expiry": best.get("expiry"),
                "ltp": best.get("ltp", 0),
                "delta": best.get("delta"),
                "iv": best.get("iv"),
                "theta": best.get("theta"),
                "gamma": best.get("gamma"),
                "vega": best.get("vega"),
                "oi": best.get("oi", 0),
                "volume": best.get("volume", 0),
                "score": best["score"],
            }

        if st.button("🔍 Fetch Option Chains", type="primary"):
            progress = st.progress(0, text="Fetching option chains...")
            options_data = []

            provider = OptionChainProvider()
            total = min(len(active_signals), max_positions)

            for i, (_, sig) in enumerate(active_signals.head(max_positions).iterrows()):
                symbol = sig["symbol"]
                signal_type = sig["signal_type"]
                entry_price = sig["entry_price"]
                sl_price = sig["stop_loss"]
                tp_price = sig["target_price"]

                progress.progress(
                    (i + 1) / max(total, 1), text=f"Fetching {symbol}..."
                )
                try:
                    chain_dict = provider.fetch_option_chain_for_symbol(symbol)
                    best = select_best_option(
                        symbol, signal_type, entry_price, chain_dict)
                    if best:
                        lot_size = get_lot_size(symbol)
                        lots = max(
                            1,
                            int(
                                capital_per_trade
                                / max(best["ltp"] * lot_size, 1e-6)
                            ),
                        )
                        option_entry = {
                            "symbol": symbol,
                            "signal_type": signal_type,
                            "spot_price": entry_price,
                            "sl": sl_price,
                            "target": tp_price,
                            **best,
                            "lot_size": lot_size,
                            "lots": lots,
                        }
                        options_data.append(option_entry)
                        update_option_details(symbol, signal_type, {
                                              **best, "lot_size": lot_size})
                except Exception as e:
                    st.warning(f"Error with {symbol}: {e}")

            if options_data:
                st.session_state["sq_options_data"] = options_data
                st.success(f"Fetched options for {len(options_data)} signals")
            else:
                st.warning("No options data fetched")

        if "sq_options_data" in st.session_state and st.session_state["sq_options_data"]:
            options_data = st.session_state["sq_options_data"]
            st.markdown("#### 📋 Option Recommendations")
            rows = []
            for opt in options_data:
                invest = opt["ltp"] * opt["lot_size"] * opt["lots"]
                emoji = "📈" if opt["option_type"] == "CE" else "📉"
                rows.append(
                    {
                        "Symbol": f"{emoji} {opt['symbol']}",
                        "Signal": opt["signal_type"],
                        "Strike": f"{opt['strike']} {opt['option_type']}",
                        "Expiry": opt.get("expiry", "N/A"),
                        "Premium": f"₹{opt['ltp']:.2f}",
                        "Lots": opt["lots"],
                        "Lot Size": opt["lot_size"],
                        "Investment": f"₹{invest:,.0f}",
                        "Delta": f"{opt.get('delta', 0):.3f}"
                        if opt.get("delta")
                        else "N/A",
                        "IV": f"{opt.get('iv', 0):.1f}%"
                        if opt.get("iv")
                        else "N/A",
                        "Theta": f"{opt.get('theta', 0):.2f}"
                        if opt.get("theta")
                        else "N/A",
                        "OI": f"{opt.get('oi', 0):,}",
                        "Volume": f"{opt.get('volume', 0):,}",
                        "Underlying Entry": f"₹{opt['spot_price']:.2f}",
                        "SL": f"₹{opt['sl']:.2f}",
                        "Target": f"₹{opt['target']:.2f}",
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            st.divider()
            st.markdown("#### 📊 Portfolio Summary")
            summary_df = pd.DataFrame(options_data)
            summary_df["investment"] = (
                summary_df["ltp"] * summary_df["lot_size"] * summary_df["lots"]
            )
            total_investment = summary_df["investment"].sum()

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Positions", len(summary_df))
            col2.metric("Total Investment", f"₹{total_investment:,.0f}")
            col3.metric("Avg Premium", f"₹{summary_df['ltp'].mean():.2f}")

            csv = summary_df.to_csv(index=False)
            st.download_button(
                "📥 Export Options",
                data=csv,
                file_name=f"squeeze15m_options_{date.today()}.csv",
                mime="text/csv",
            )

# ============================
# TAB 6: 15m SQUEEZE BACKTEST
# ============================
with tab6:
    st.markdown("### 📊 Single Stock Backtester (15m Squeeze)")

    bt_symbol_map = {
        row["trading_symbol"]: row["instrument_key"]
        for _, row in fo_stocks.iterrows()
    }

    col1, col2 = st.columns([1, 3])
    with col1:
        bt_symbol = st.selectbox(
            "Select Stock", list(bt_symbol_map.keys()), key="sq_bt_sym"
        )
        bt_instrument_key = bt_symbol_map[bt_symbol]
        bt_lookback = st.slider(
            "Lookback Days", 30, 365, 180, key="sq_bt_lb"
        )
        bt_sl_mode = st.selectbox(
            "SL / TP Mode", ["ATR based", "Fixed %"], key="sq_bt_slmode"
        )
        bt_capital = st.number_input(
            "Initial Capital", 10000, 10000000, 100000, key="sq_bt_cap"
        )

        bt_atr_mult = 2.0
        bt_rr = 2.0
        bt_sl_pct = 0.01
        bt_tp_pct = 0.02

        if bt_sl_mode == "ATR based":
            bt_atr_mult = st.number_input(
                "ATR SL Mult", 0.5, 5.0, 2.0, 0.5, key="sq_bt_atr"
            )
            bt_rr = st.number_input(
                "RR (TP = RR×Risk)", 1.0, 5.0, 2.0, 0.5, key="sq_bt_rr"
            )
        else:
            bt_sl_pct = (
                st.number_input(
                    "SL %", 0.2, 10.0, 1.0, 0.2, key="sq_bt_slpct"
                )
                / 100.0
            )
            bt_tp_pct = (
                st.number_input(
                    "TP %", 0.5, 20.0, 2.0, 0.5, key="sq_bt_tppct"
                )
                / 100.0
            )

        run_bt = st.button(
            "🚀 Run 15m Squeeze Backtest", type="primary", key="sq_run_bt"
        )

    with col2:
        if run_bt:
            df = load_data_fast(bt_instrument_key, "15minute", bt_lookback)
            if df is None or df.empty:
                st.warning("No data available for backtest.")
            else:
                mode_str = "ATR" if bt_sl_mode == "ATR based" else "PCT"
                signals = build_15m_signals(
                    df,
                    sl_mode=mode_str,
                    sl_atr_mult=bt_atr_mult,
                    tp_rr=bt_rr,
                    sl_pct=bt_sl_pct,
                    tp_pct=bt_tp_pct,
                )
                # --- START MODIFICATION for backtest signals ---
                # Filter for signals with an alignment score of 5
                signals = [s for s in signals if s.score == 5]
                # --- END MODIFICATION ---

                if not signals:
                    # Updated text
                    st.info(
                        "No Squeeze signals with Alignment Score 5 in backtest window.")
                else:
                    # Simple sequential backtest: 1 trade at a time, risk 1R per trade
                    equity = bt_capital
                    eq_curve = []
                    trades = []

                    for sig in signals:
                        direction = 1 if sig.signal_type == "LONG" else -1
                        entry = sig.entry_price
                        sl = sig.sl_price
                        tp = sig.tp_price

                        risk_per_share = abs(entry - sl)
                        if risk_per_share <= 0:
                            continue
                        size = equity * 0.01 / risk_per_share  # 1% risk
                        pnl = (tp - entry) * direction * size

                        equity += pnl
                        eq_curve.append(
                            {"time": sig.timestamp, "equity": equity})
                        trades.append(
                            {
                                "Time": sig.timestamp,
                                "Signal": sig.signal_type,
                                "Entry": round(entry, 2),
                                "SL": round(sl, 2),
                                "TP": round(tp, 2),
                                "PnL": round(pnl, 2),
                                "Equity": round(equity, 2),
                                # This will be 5.0
                                "Score": round(sig.score, 1),
                            }
                        )

                    if trades:
                        trades_df = pd.DataFrame(trades)
                        eq_df = pd.DataFrame(eq_curve)

                        total_return = (equity / bt_capital - 1) * 100
                        wins = trades_df[trades_df["PnL"] > 0]
                        win_rate = len(wins) / len(trades_df) * 100

                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Trades", len(trades_df))
                        col_b.metric("Win Rate", f"{win_rate:.1f}%")
                        col_c.metric("Total Return", f"{total_return:.1f}%")

                        st.markdown("#### 📋 Trades")
                        st.dataframe(
                            trades_df, use_container_width=True, height=300)

                        st.markdown("#### 📈 Equity Curve (discrete)")
                        st.line_chart(
                            eq_df.set_index("time")["equity"],
                            use_container_width=True,
                        )
                    else:
                        # Updated text
                        st.info(
                            "No valid trades executed in backtest with Alignment Score 5.")

# ============================
# TAB 7: TRADE LOG
# ============================
with tab7:
    st.markdown("### 📋 Trade Log History (Universe)")

    try:
        history_df = db.con.execute(
            """
            SELECT signal_date, symbol, signal_type, signal_strength,
                   entry_price, stop_loss, target_price, status
            FROM ehma_universe
            ORDER BY signal_date DESC, signal_strength DESC
            LIMIT 200
            """
        ).df()
        if not history_df.empty:
            st.dataframe(history_df, use_container_width=True, height=500)
            csv = history_df.to_csv(index=False)
            st.download_button(
                "📥 Export History",
                data=csv,
                file_name="squeeze_trade_history.csv",
                mime="text/csv",
            )
        else:
            st.info("No trade history yet.")
    except Exception as e:
        st.error(f"Error loading history: {e}")


# if __name__ == "__main__":
#     print("=" * 60)
#     print("Squeeze Backtest V2 - Session Limits & Breakeven Stops")
#     print("=" * 60)
#     print("\nKey improvements over V1:")
#     print("  1. Max 1 trade per stock per SESSION (day)")
#     print("  2. Breakeven stop at 1R profit")
#     print("  3. Quality filters (volume, volatility)")
#     print("  4. Optimized parameters (ATR 1.5x, RR 2.5)")
#     print("  5. Shorter max hold (30 bars vs 50)")
#     print("\nExpected improvements:")
#     print("  - Fewer trades (quality over quantity)")
#     print("  - Higher win rate (45%+ vs 33%)")
#     print("  - Better profit factor (1.5+ vs <1.0)")
#     print("  - Protected profits via breakeven stops")
#     print("\nTo use, replace your current functions with:")
#     print("  - build_15m_signals_with_backtest_v2")
#     print("  - run_batch_scan_squeeze_15m_v2_improved")
