# core/strategies/indian_market_squeeze.py
"""
Indian Market Squeeze Strategy - 15m Timeframe
===============================================
Version: 3.0 - Full Backtest Support with Trade Tracking

Features:
- Bollinger Bands + Keltner Channel squeeze detection
- Dual SuperTrend (fast/slow) for trend confirmation
- WaveTrend oscillator for momentum
- Williams %R for exhaustion detection
- Trade outcome tracking (SL/TP hit detection)
- P&L calculation for backtesting
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Dict
from datetime import datetime, date, time, timedelta
SignalSide = Literal["LONG", "SHORT"]
TradeStatus = Literal["ACTIVE", "SL_HIT", "TP_HIT", "EXPIRED", "OPEN"]

# Ensure date is available at module level for type hints
__all__ = ['SqueezeSignal', 'BacktestResult', 'check_trade_outcome', 
           'build_15m_signals_with_backtest', 'run_batch_scan_squeeze_15m_v2']




@dataclass
class SqueezeSignal:
    """Represents a single squeeze signal with trade tracking."""
    signal_type: Literal["LONG", "SHORT"]
    timestamp: pd.Timestamp
    entry_price: float
    sl_price: float
    tp_price: float
    reasons: List[str] = field(default_factory=list)
    score: float = 0.0
    
    # Trade outcome fields
    status: str = "ACTIVE"  # ACTIVE, SL_HIT, TP_HIT, EXPIRED, BREAKEVEN
    exit_price: float = 0.0
    exit_time: Optional[pd.Timestamp] = None
    exit_reason: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    bars_held: int = 0
    
    # New fields for session tracking
    trade_date: Optional[date] = None
    breakeven_triggered: bool = False


@dataclass
class BacktestResult:
    """Container for backtest results."""
    active_signals: List[SqueezeSignal] = field(default_factory=list)
    completed_trades: List[SqueezeSignal] = field(default_factory=list)
    skipped_signals: List[SqueezeSignal] = field(default_factory=list)  # New: track skipped
    
    # Statistics
    win_count: int = 0
    loss_count: int = 0
    breakeven_count: int = 0  # New
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Session stats
    signals_skipped_session_limit: int = 0
    signals_skipped_cooldown: int = 0
    signals_skipped_low_quality: int = 0


def compute_squeeze_stack(df_15m: pd.DataFrame,
                          bb_length: int = 20,
                          bb_mult: float = 1.6,
                          kc_length: int = 20,
                          kc_mult: float = 1.25,
                          use_true_range: bool = True,
                          wt_n1: int = 9,
                          wt_n2: int = 21,
                          wt_ob_level: int = 53,
                          wt_os_level: int = -53,
                          wr_fast_len: int = 8,
                          wr_slow_len: int = 34,
                          wr_ob_level: float = -15.0,
                          wr_os_level: float = -85.0,
                          require_all_conf: bool = True,
                          session_mask: Optional[pd.Series] = None
                          ) -> pd.DataFrame:
    """
    Re-implements Indian Market Squeeze logic on 15m candles.
    """
    df = df_15m.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # --- Bollinger Bands ---
    bb_basis = close.rolling(bb_length).mean()
    bb_dev = close.rolling(bb_length).std() * bb_mult
    bb_upper = bb_basis + bb_dev
    bb_lower = bb_basis - bb_dev

    # --- Keltner Channels ---
    if use_true_range:
        tr = np.maximum(high - low,
                        np.maximum((high - close.shift()).abs(),
                                   (low - close.shift()).abs()))
    else:
        tr = high - low
    kc_ma = close.rolling(kc_length).mean()
    kc_range_ma = tr.rolling(kc_length).mean()
    kc_upper = kc_ma + kc_range_ma * kc_mult
    kc_lower = kc_ma - kc_range_ma * kc_mult

    sqz_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    sqz_off = (bb_lower < kc_lower) & (bb_upper > kc_upper)
    no_sqz = ~sqz_on & ~sqz_off

    highest_high = high.rolling(kc_length).max()
    lowest_low = low.rolling(kc_length).min()
    mid = (highest_high + lowest_low) / 2.0
    sqz_mom = (close - (mid + kc_ma) / 2.0).rolling(kc_length).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )

    mom_up = sqz_mom > 0
    mom_dn = sqz_mom < 0
    mom_rising = sqz_mom > sqz_mom.shift()
    mom_falling = sqz_mom < sqz_mom.shift()

    # --- SuperTrend ---
    def supertrend(high_s, low_s, close_s, length, mult):
        tr_local = np.maximum(high_s - low_s,
                              np.maximum((high_s - close_s.shift()).abs(),
                                         (low_s - close_s.shift()).abs()))
        atr = tr_local.rolling(length).mean()
        hl2 = (high_s + low_s) / 2.0
        basic_upper = hl2 + mult * atr
        basic_lower = hl2 - mult * atr

        final_upper = basic_upper.copy()
        final_lower = basic_lower.copy()
        trend = pd.Series(True, index=close_s.index)

        prev_fu, prev_fl, prev_t = np.nan, np.nan, True
        for i in range(len(close_s)):
            if i == 0:
                final_upper.iat[i] = basic_upper.iat[i]
                final_lower.iat[i] = basic_lower.iat[i]
                prev_fu, prev_fl = final_upper.iat[i], final_lower.iat[i]
                continue

            bu, bl = basic_upper.iat[i], basic_lower.iat[i]
            c_prev = close_s.iat[i - 1]

            fu = bu if pd.isna(prev_fu) or bu < prev_fu or c_prev > prev_fu else prev_fu
            fl = bl if pd.isna(prev_fl) or bl > prev_fl or c_prev < prev_fl else prev_fl

            final_upper.iat[i], final_lower.iat[i] = fu, fl

            c = close_s.iat[i]
            t = True if c > prev_fu else (False if c < prev_fl else prev_t)
            trend.iat[i] = t
            prev_fu, prev_fl, prev_t = fu, fl, t

        return trend, ~trend

    st_fast_bull, st_fast_bear = supertrend(high, low, close, 7, 1.5)
    st_slow_bull, st_slow_bear = supertrend(high, low, close, 11, 2.5)
    st_fast_flip_bear = st_fast_bear & st_fast_bull.shift()
    st_fast_flip_bull = st_fast_bull & st_fast_bear.shift()
    st_both_bull = st_fast_bull & st_slow_bull
    st_both_bear = st_fast_bear & st_slow_bear

    # --- WaveTrend ---
    hlc3 = (high + low + close) / 3.0
    esa = hlc3.ewm(span=wt_n1, adjust=False).mean()
    d = (hlc3 - esa).abs().ewm(span=wt_n1, adjust=False).mean()
    ci = (hlc3 - esa) / (0.015 * d).replace(0, np.nan)
    ci = ci.fillna(0)
    wt1 = ci.ewm(span=wt_n2, adjust=False).mean()
    wt2 = wt1.rolling(4).mean()
    wt_cross_up = (wt1.shift() < wt2.shift()) & (wt1 > wt2)
    wt_cross_down = (wt1.shift() > wt2.shift()) & (wt1 < wt2)
    wt_overbought = (wt1 >= wt_ob_level).fillna(False)
    wt_oversold = (wt1 <= wt_os_level).fillna(False)

    # --- Williams %R ---
    def williams_r(h, l, c, length):
        hh, ll = h.rolling(length).max(), l.rolling(length).min()
        return ((hh - c) / (hh - ll).replace(0, np.nan) * -100.0)

    wr_fast = williams_r(high, low, close, wr_fast_len)
    wr_slow = williams_r(high, low, close, wr_slow_len)
    wr_bull_exhaustion = (wr_fast >= wr_ob_level) & (wr_slow >= wr_ob_level) & (wr_fast < wr_fast.shift())
    wr_bear_exhaustion = (wr_fast <= wr_os_level) & (wr_slow <= wr_os_level) & (wr_fast > wr_fast.shift())

    # --- Session filter ---
    in_session = session_mask.reindex(df.index).fillna(False).astype(bool) if session_mask is not None else pd.Series(True, index=df.index)

    recent_sqz = sqz_on.shift(1) | sqz_on.shift(2) | sqz_on.shift(3)

    # Long conditions
    long_squeeze = recent_sqz & mom_up & mom_rising
    long_st = st_both_bull
    long_wt = wt_cross_up | ((wt1 > wt2) & (wt1 > wt1.shift()))
    long_entry = long_squeeze & long_st & (long_wt if require_all_conf else True) & ~wt_overbought & in_session
    long_entry = long_entry.fillna(False).astype(bool)
    long_signal = long_entry & ~long_entry.shift().fillna(False).astype(bool)

    # Short conditions
    short_squeeze = recent_sqz & mom_dn & mom_falling
    short_st = st_both_bear
    short_wt = wt_cross_down | ((wt1 < wt2) & (wt1 < wt1.shift()))
    short_entry = short_squeeze & short_st & (short_wt if require_all_conf else True) & ~wt_oversold & in_session
    short_entry = short_entry.fillna(False).astype(bool)
    short_signal = short_entry & ~short_entry.shift().fillna(False).astype(bool)

    # Exits
    long_exit = wr_bull_exhaustion | st_fast_flip_bear
    short_exit = wr_bear_exhaustion | st_fast_flip_bull

    out = pd.DataFrame(index=df.index)
    out["long_signal"] = long_signal
    out["short_signal"] = short_signal
    out["long_exit"] = long_exit
    out["short_exit"] = short_exit
    out["st_both_bull"] = st_both_bull
    out["st_both_bear"] = st_both_bear
    out["wt_cross_up"] = wt_cross_up
    out["wt_cross_down"] = wt_cross_down
    out["recent_sqz"] = recent_sqz
    out["mom_rising"] = mom_rising
    out["mom_falling"] = mom_falling
    out["wr_bull_exhaustion"] = wr_bull_exhaustion
    out["wr_bear_exhaustion"] = wr_bear_exhaustion
    
    return out


def check_trade_outcome(

    df_15m: pd.DataFrame,
    signal: SqueezeSignal,
    max_bars: int = 30,
    use_breakeven: bool = True,
    breakeven_at_r: float = 1.0,
    use_trailing: bool = False,
    trailing_atr_mult: float = 1.5,
) -> SqueezeSignal:
    """
    Enhanced trade outcome checker with breakeven stop.
    
    Args:
        df_15m: Full OHLCV DataFrame
        signal: The signal to check
        max_bars: Maximum bars to hold trade
        use_breakeven: Enable breakeven stop logic
        breakeven_at_r: R-multiple at which to move SL to breakeven
        use_trailing: Enable trailing stop (advanced)
        trailing_atr_mult: ATR multiplier for trailing stop
        
    Returns:
        Updated SqueezeSignal with outcome details
    """
    try:
        signal_idx = df_15m.index.get_loc(signal.timestamp)
    except KeyError:
        signal.status = "ACTIVE"
        return signal
    
    end_idx = min(signal_idx + max_bars + 1, len(df_15m))
    future_bars = df_15m.iloc[signal_idx + 1:end_idx]
    
    if future_bars.empty:
        signal.status = "ACTIVE"
        return signal
    
    # Calculate initial risk for breakeven logic
    initial_risk = abs(signal.entry_price - signal.sl_price)
    current_sl = signal.sl_price
    breakeven_triggered = False
    
    for bar_num, (idx, bar) in enumerate(future_bars.iterrows(), 1):
        high_price = bar["High"]
        low_price = bar["Low"]
        close_price = bar["Close"]
        
        if signal.signal_type == "LONG":
            # Calculate current profit
            current_profit = close_price - signal.entry_price
            
            # CHECK 1: Breakeven trigger (before checking SL/TP)
            if use_breakeven and not breakeven_triggered:
                if current_profit >= (initial_risk * breakeven_at_r):
                    current_sl = signal.entry_price + (initial_risk * 0.5)  # FIXED: 0.5R profit lock (was 0.1R)
                    breakeven_triggered = True
                    signal.breakeven_triggered = True
            
            # CHECK 2: SL hit (use current_sl which may be at breakeven)
            if low_price <= current_sl:
                signal.status = "SL_HIT" if not breakeven_triggered else "BREAKEVEN"
                signal.exit_price = current_sl
                signal.exit_time = idx
                signal.exit_reason = "Stop Loss" if not breakeven_triggered else "Breakeven Stop"
                signal.pnl = current_sl - signal.entry_price
                signal.pnl_pct = (signal.pnl / signal.entry_price) * 100
                signal.bars_held = bar_num
                return signal
            
            # CHECK 3: TP hit
            if high_price >= signal.tp_price:
                signal.status = "TP_HIT"
                signal.exit_price = signal.tp_price
                signal.exit_time = idx
                signal.exit_reason = "Take Profit"
                signal.pnl = signal.tp_price - signal.entry_price
                signal.pnl_pct = (signal.pnl / signal.entry_price) * 100
                signal.bars_held = bar_num
                return signal
                
        else:  # SHORT
            # Calculate current profit
            current_profit = signal.entry_price - close_price
            
            # CHECK 1: Breakeven trigger
            if use_breakeven and not breakeven_triggered:
                if current_profit >= (initial_risk * breakeven_at_r):
                    current_sl = signal.entry_price - (initial_risk * 0.5)  # FIXED: 0.5R profit lock (was 0.1R)
                    breakeven_triggered = True
                    signal.breakeven_triggered = True
            
            # CHECK 2: SL hit
            if high_price >= current_sl:
                signal.status = "SL_HIT" if not breakeven_triggered else "BREAKEVEN"
                signal.exit_price = current_sl
                signal.exit_time = idx
                signal.exit_reason = "Stop Loss" if not breakeven_triggered else "Breakeven Stop"
                signal.pnl = signal.entry_price - current_sl
                signal.pnl_pct = (signal.pnl / signal.entry_price) * 100
                signal.bars_held = bar_num
                return signal
            
            # CHECK 3: TP hit
            if low_price <= signal.tp_price:
                signal.status = "TP_HIT"
                signal.exit_price = signal.tp_price
                signal.exit_time = idx
                signal.exit_reason = "Take Profit"
                signal.pnl = signal.entry_price - signal.tp_price
                signal.pnl_pct = (signal.pnl / signal.entry_price) * 100
                signal.bars_held = bar_num
                return signal
    
    # Neither SL nor TP hit within max_bars - EXPIRED
    if len(future_bars) >= max_bars:
        signal.status = "EXPIRED"
        signal.exit_price = future_bars["Close"].iloc[-1]
        signal.exit_time = future_bars.index[-1]
        signal.exit_reason = "Max bars reached"
        if signal.signal_type == "LONG":
            signal.pnl = signal.exit_price - signal.entry_price
        else:
            signal.pnl = signal.entry_price - signal.exit_price
        signal.pnl_pct = (signal.pnl / signal.entry_price) * 100
        signal.bars_held = max_bars
    else:
        signal.status = "ACTIVE"
    
    return signal


def build_15m_signals_with_backtest(
df_15m: pd.DataFrame,
    sl_mode: Literal["ATR", "PCT"] = "ATR",
    sl_atr_mult: float = 1.5,        # Tighter SL (was 2.0)
    tp_rr: float = 2.5,              # Higher RR (was 2.0)
    sl_pct: float = 0.01,
    tp_pct: float = 0.02,
    max_trade_bars: int = 30,        # Shorter holding (was 50)
    capital_per_trade: float = 10000.0,
    
    # Session limit options
    max_trades_per_session: int = 1,  # KEY FIX: Limit trades per day
    cooldown_bars: int = 4,           # Bars to wait after trade closes
    
    # Breakeven options
    use_breakeven: bool = True,
    breakeven_at_r: float = 1.0,
    
    # Quality filters
    use_quality_filter: bool = True,
    min_volume_ratio: float = 0.7,
    min_score: float = 5.0,           # Only take score 5 signals
    
    # Compute squeeze function (passed in)
    compute_squeeze_fn=None,
) -> BacktestResult:
    """
    Build signals WITH session limits and breakeven stops.
    
    Key improvements:
    1. Max 1 trade per stock per SESSION (trading day)
    2. Breakeven stop at 1R profit
    3. Quality filters for high-probability entries
    4. Cooldown between trades
    
    Args:
        df_15m: OHLCV DataFrame with DatetimeIndex
        sl_mode: "ATR" or "PCT"
        sl_atr_mult: ATR multiplier for SL
        tp_rr: Risk:Reward ratio
        sl_pct: Fixed SL percentage (if PCT mode)
        tp_pct: Fixed TP percentage (if PCT mode)
        max_trade_bars: Max bars to hold trade
        max_trades_per_session: Max trades per day (SESSION LIMIT)
        cooldown_bars: Bars to wait after trade closes
        use_breakeven: Enable breakeven stops
        breakeven_at_r: R-multiple for breakeven trigger
        use_quality_filter: Enable entry quality checks
        min_volume_ratio: Min volume vs average
        min_score: Minimum alignment score required
        compute_squeeze_fn: Function to compute squeeze signals
        
    Returns:
        BacktestResult with all trade details and statistics
    """
    
    # Import compute_squeeze_stack if not provided
    if compute_squeeze_fn is None:
        try:
            from core.strategies.squeeze_momentum import compute_squeeze_stack
            compute_squeeze_fn = compute_squeeze_stack
        except ImportError:
            raise ImportError("compute_squeeze_stack not found. Please provide compute_squeeze_fn.")
    
    sig_df = compute_squeeze_fn(df_15m)
    atr = (df_15m["High"] - df_15m["Low"]).rolling(14).mean()
    
    # CRITICAL FIX: Merge signal columns with price data
    # sig_df only has signal columns, we need to join with df_15m for prices
    merged_df = df_15m.copy()
    for col in sig_df.columns:
        if col not in merged_df.columns:
            merged_df[col] = sig_df[col]
    
    all_signals: List[SqueezeSignal] = []
    skipped_signals: List[SqueezeSignal] = []
    
    # SESSION TRACKING
    trades_per_session: Dict[date, int] = {}  # date -> count
    last_trade_exit_idx: Optional[int] = None  # For cooldown
    active_trade: Optional[SqueezeSignal] = None  # Track if we're in a trade
    
    # Statistics for skipped signals
    skipped_session_limit = 0
    skipped_cooldown = 0
    skipped_low_quality = 0
    skipped_low_score = 0
    
    for ts, row in merged_df.iterrows():
        close = float(row["Close"])  # Now row has Close since we merged
        current_idx = merged_df.index.get_loc(ts)
        current_date = ts.date()
        
        # Get signal type - handle both boolean and numeric values
        signal_type = None
        long_sig = row.get("long_signal", False)
        short_sig = row.get("short_signal", False)
        
        # Convert to boolean safely
        is_long = bool(long_sig) if pd.notna(long_sig) else False
        is_short = bool(short_sig) if pd.notna(short_sig) else False
        
        if is_long:
            signal_type = "LONG"
        elif is_short:
            signal_type = "SHORT"
        else:
            continue  # No signal
        
        # Compute alignment score (FIXED: Exhaustion disqualifies, doesn't reduce score)
        score = 0.0

        # First check for exhaustion - if present, skip this signal entirely
        if signal_type == "LONG" and row.get("wr_bull_exhaustion"):
            continue  # Skip long signals with bullish exhaustion
        if signal_type == "SHORT" and row.get("wr_bear_exhaustion"):
            continue  # Skip short signals with bearish exhaustion

        # Build positive score (max 5.0)
        if row.get("st_both_bull") and row.get("long_signal"): score += 2.0
        if row.get("st_both_bear") and row.get("short_signal"): score += 2.0
        if row.get("wt_cross_up") and row.get("long_signal"): score += 1.0
        if row.get("wt_cross_down") and row.get("short_signal"): score += 1.0
        if row.get("recent_sqz"): score += 1.0
        if row.get("mom_rising") and row.get("long_signal"): score += 1.0
        if row.get("mom_falling") and row.get("short_signal"): score += 1.0
        
        # =====================================================================
        # FILTER 1: Minimum score check
        # =====================================================================
        if score < min_score:
            skipped_low_score += 1
            continue
        
        # =====================================================================
        # FILTER 2: Session limit check (MAX 1 TRADE PER DAY)
        # =====================================================================
        session_trades = trades_per_session.get(current_date, 0)
        if session_trades >= max_trades_per_session:
            skipped_session_limit += 1
            continue
        
        # =====================================================================
        # FILTER 3: Cooldown check (wait after previous trade)
        # =====================================================================
        if last_trade_exit_idx is not None:
            bars_since_exit = current_idx - last_trade_exit_idx
            if bars_since_exit < cooldown_bars:
                skipped_cooldown += 1
                continue
        
        # =====================================================================
        # FILTER 4: Active trade check (no overlapping trades)
        # =====================================================================
        if active_trade is not None and active_trade.status == "ACTIVE":
            # Check if active trade has now completed
            active_trade = check_trade_outcome(
                df_15m, active_trade, max_trade_bars,
                use_breakeven=use_breakeven, breakeven_at_r=breakeven_at_r
            )
            if active_trade.status == "ACTIVE":
                # Still in trade, skip this signal
                continue
            else:
                # Trade completed, record exit
                last_trade_exit_idx = df_15m.index.get_loc(active_trade.exit_time) if active_trade.exit_time else current_idx
                active_trade = None
        
        # =====================================================================
        # FILTER 5: Quality filter (volume, volatility, position)
        # =====================================================================
        if use_quality_filter:
            is_valid, reason = is_high_probability_entry(
                df_15m, ts, signal_type, 
                min_volume_ratio=min_volume_ratio
            )
            if not is_valid:
                skipped_low_quality += 1
                continue
        
        # =====================================================================
        # CALCULATE SL/TP
        # =====================================================================
        if sl_mode == "ATR":
            atr_val = atr.loc[ts]
            if pd.isna(atr_val):
                continue
            
            if signal_type == "LONG":
                sl = close - sl_atr_mult * float(atr_val)
                risk = close - sl
                tp = close + (risk * tp_rr)
            else:  # SHORT
                sl = close + sl_atr_mult * float(atr_val)
                risk = sl - close
                tp = close - (risk * tp_rr)
        else:  # PCT mode
            if signal_type == "LONG":
                sl = close * (1 - sl_pct)
                tp = close * (1 + tp_pct)
            else:
                sl = close * (1 + sl_pct)
                tp = close * (1 - tp_pct)
        
        # =====================================================================
        # CREATE SIGNAL
        # =====================================================================
        signal = SqueezeSignal(
            signal_type=signal_type,
            timestamp=ts,
            entry_price=close,
            sl_price=sl,
            tp_price=tp,
            reasons=["Squeeze", "ST aligned", "WT aligned"],
            score=score,
            trade_date=current_date,
        )
        
        # =====================================================================
        # CHECK TRADE OUTCOME
        # =====================================================================
        signal = check_trade_outcome(
            df_15m, signal, max_trade_bars,
            use_breakeven=use_breakeven,
            breakeven_at_r=breakeven_at_r
        )
        
        # =====================================================================
        # UPDATE SESSION TRACKING
        # =====================================================================
        trades_per_session[current_date] = session_trades + 1
        
        if signal.status == "ACTIVE":
            active_trade = signal
        else:
            # Trade completed immediately or within lookforward
            if signal.exit_time:
                last_trade_exit_idx = df_15m.index.get_loc(signal.exit_time)
        
        all_signals.append(signal)
    
    # =========================================================================
    # FINALIZE: Check any remaining active trade
    # =========================================================================
    if active_trade and active_trade.status == "ACTIVE":
        active_trade = check_trade_outcome(
            df_15m, active_trade, max_trade_bars,
            use_breakeven=use_breakeven, breakeven_at_r=breakeven_at_r
        )
        # Update in list
        for i, sig in enumerate(all_signals):
            if sig.timestamp == active_trade.timestamp:
                all_signals[i] = active_trade
                break
    
    # =========================================================================
    # SEPARATE ACTIVE AND COMPLETED
    # =========================================================================
    active = [s for s in all_signals if s.status == "ACTIVE"]
    completed = [s for s in all_signals if s.status in ["SL_HIT", "TP_HIT", "EXPIRED", "BREAKEVEN"]]
    
    # =========================================================================
    # CALCULATE STATISTICS
    # =========================================================================
    result = BacktestResult(
        active_signals=active,
        completed_trades=completed,
        skipped_signals=skipped_signals,
        signals_skipped_session_limit=skipped_session_limit,
        signals_skipped_cooldown=skipped_cooldown,
        signals_skipped_low_quality=skipped_low_quality,
    )
    
    if completed:
        wins = [t for t in completed if t.pnl > 0]
        losses = [t for t in completed if t.pnl <= 0]
        breakevens = [t for t in completed if t.status == "BREAKEVEN"]
        
        result.win_count = len(wins)
        result.loss_count = len(losses)
        result.breakeven_count = len(breakevens)
        result.total_pnl = sum(t.pnl for t in completed)
        result.total_pnl_pct = sum(t.pnl_pct for t in completed)
        result.win_rate = (len(wins) / len(completed) * 100) if completed else 0
        result.avg_win = (sum(t.pnl for t in wins) / len(wins)) if wins else 0
        result.avg_loss = (sum(t.pnl for t in losses) / len(losses)) if losses else 0
        
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
        result.profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
    
    return result

def is_high_probability_entry(
    df_15m: pd.DataFrame,
    timestamp: pd.Timestamp,
    signal_type: str,
    min_volume_ratio: float = 0.5,  # RELAXED: 0.5 instead of 0.7 (less strict)
    max_volatility_ratio: float = 2.0,  # RELAXED: 2.0 instead of 1.5
) -> Tuple[bool, str]:
    """
    Filter out low-quality signals.
    
    Returns:
        Tuple of (is_valid, rejection_reason)
    """
    try:
        idx = df_15m.index.get_loc(timestamp)
    except KeyError:
        return False, "timestamp_not_found"
    
    if idx < 20:  # Need enough history
        return False, "insufficient_history"
    
    close = df_15m["Close"].iloc[idx]
    
    # 1. VOLUME CHECK - Need at least min_volume_ratio of average
    if "Volume" in df_15m.columns:
        current_volume = df_15m["Volume"].iloc[idx]
        avg_volume = df_15m["Volume"].iloc[idx-20:idx].mean()
        
        if avg_volume > 0 and current_volume < (avg_volume * min_volume_ratio):
            return False, "low_volume"
    
    # 2. VOLATILITY CHECK - Reject if too choppy
    recent_range = (df_15m["High"].iloc[idx-10:idx] - df_15m["Low"].iloc[idx-10:idx]).mean()
    avg_range = (df_15m["High"].iloc[idx-50:idx] - df_15m["Low"].iloc[idx-50:idx]).mean()
    
    if avg_range > 0 and recent_range > (avg_range * max_volatility_ratio):
        return False, "high_volatility"
    
    # 3. POSITION IN RANGE CHECK - Relaxed for squeeze reversals
    # Squeeze signals often fire after price has moved off extremes
    # Original thresholds (0.85/0.15) were too strict
    recent_high = df_15m["High"].iloc[idx-5:idx+1].max()
    recent_low = df_15m["Low"].iloc[idx-5:idx+1].min()
    range_size = recent_high - recent_low

    if range_size > 0:
        position_in_range = (close - recent_low) / range_size

        # Relaxed thresholds: only reject absolute extremes
        if signal_type == "LONG" and position_in_range > 0.95:
            return False, "buying_at_high"
        elif signal_type == "SHORT" and position_in_range < 0.05:
            return False, "selling_at_low"
    
    # 4. TREND CONFIRMATION (removed momentum check - squeeze is a reversal strategy)
    # For squeeze reversals, we expect:
    # - LONG signals after pullbacks (price down, then WaveTrend crosses up)
    # - SHORT signals after rallies (price up, then WaveTrend crosses down)
    # The original momentum check was backwards and rejecting best entries.

    # Instead, check that we're not in extreme trend exhaustion
    if idx >= 10:
        # Check for extreme moves that suggest blow-off top/bottom
        price_change_10bar = abs(df_15m["Close"].iloc[idx] - df_15m["Close"].iloc[idx-10])
        atr_10bar = (df_15m["High"].iloc[idx-10:idx] - df_15m["Low"].iloc[idx-10:idx]).mean()

        # Reject if price moved > 3 ATRs in 10 bars (too extended)
        if atr_10bar > 0 and price_change_10bar > (3.0 * atr_10bar):
            return False, "overextended"

    return True, "passed"

# Convenience functions for different use cases

def build_15m_signals(
    df_15m: pd.DataFrame,
    sl_mode: Literal["ATR", "PCT"],
    sl_atr_mult: float = 2.0,
    tp_rr: float = 2.0,
    sl_pct: float = 0.01,
    tp_pct: float = 0.02,
) -> List[SqueezeSignal]:
    """
    Original function - returns all signals without outcome tracking.
    For backward compatibility.
    """
    result = build_15m_signals_with_backtest(
        df_15m, sl_mode, sl_atr_mult, tp_rr, sl_pct, tp_pct
    )
    return result.active_signals + result.completed_trades


def build_15m_signals_for_live_scan(
    df_15m: pd.DataFrame,
    sl_mode: Literal["ATR", "PCT"],
    sl_atr_mult: float = 2.0,
    tp_rr: float = 2.0,
    sl_pct: float = 0.01,
    tp_pct: float = 0.02,
    lookback_bars: int = 10,
    include_score_4: bool = False,
) -> Tuple[List[SqueezeSignal], List[SqueezeSignal]]:
    """
    For live scanning - returns (score_5_signals, score_4_signals) from recent bars.

    Returns:
        Tuple of (score_5_signals, score_4_signals)
    """
    # Only scan recent bars
    if len(df_15m) > lookback_bars + 100:
        df_scan = df_15m.iloc[-(lookback_bars + 100):]
    else:
        df_scan = df_15m

    result = build_15m_signals_with_backtest(
        df_scan, sl_mode, sl_atr_mult, tp_rr, sl_pct, tp_pct,
        max_trade_bars=20,
        use_quality_filter=True,  # Enable quality filter for live trades
        min_score=0.0  # Get all signals, we'll filter by score below
    )

    # Only return signals from last N bars that are still active
    cutoff_idx = len(df_15m) - lookback_bars
    recent_active = [
        s for s in result.active_signals
        if df_15m.index.get_loc(s.timestamp) >= cutoff_idx
    ]

    # Separate by score
    score_5_signals = [s for s in recent_active if s.score >= 5.0]
    score_4_signals = [s for s in recent_active if 4.0 <= s.score < 5.0] if include_score_4 else []

    return score_5_signals, score_4_signals