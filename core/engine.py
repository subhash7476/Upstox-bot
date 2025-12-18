import pandas as pd

# =========================
# CONFIG
# =========================
SLIPPAGE_PCT = 0.03          # 0.03%
TRANSACTION_COST_PCT = 0.08 # 0.08%
SHORT_MARGIN_RATIO = 0.30



def run_backtest(
    df: pd.DataFrame,
    capital: float,
    risk_pct: float,
    sl_points: float,
    rr: float,
    direction: str = "Both",
    enable_costs: bool = True,
):
    """
    Model A backtest engine

    - Signal evaluated on bar i
    - Entry executed on bar i+1 open
    - SL/TP evaluated using High/Low of bar i
    - Exit executed on bar i+1 open
    """

    cash = float(capital)
    equity_curve = []
    trades = []
    open_trade = None

    for i in range(len(df) - 1):
        signal_bar = i
        exec_bar = i + 1

        row = df.iloc[signal_bar]
        next_row = df.iloc[exec_bar]

        # =========================
        # EXIT LOGIC
        # =========================
        if open_trade is not None:
            side = open_trade["side"]
            sl = open_trade["sl"]
            tp = open_trade["tp"]

            low = row["Low"]
            high = row["High"]

            if side == "LONG":
                sl_hit = low <= sl
                tp_hit = high >= tp
            else:
                sl_hit = high >= sl
                tp_hit = low <= tp

            if sl_hit or tp_hit:
                exit_price = float(next_row["Open"])
                exit_time = next_row.name

                # Slippage
                if enable_costs:
                    slip = exit_price * SLIPPAGE_PCT / 100
                    exit_price = exit_price - slip if side == "LONG" else exit_price + slip

                # PnL
                if side == "LONG":
                    pnl = (exit_price - open_trade["entry_price"]) * open_trade["qty"]
                    cash += open_trade["qty"] * exit_price
                else:
                    pnl = (open_trade["entry_price"] - exit_price) * open_trade["qty"]
                    cash += open_trade["qty"] * open_trade["entry_price"] * SHORT_MARGIN_RATIO
                    cash += pnl

                # Enforce SL loss
                if sl_hit:
                    pnl = -abs(pnl)

                # Costs
                cost = 0.0
                if enable_costs:
                    cost = abs(exit_price * open_trade["qty"] * TRANSACTION_COST_PCT / 100)
                    cash -= cost
                    pnl -= cost

                R = pnl / open_trade["risk"] if open_trade["risk"] > 0 else 0.0

                trades.append({
                    "side": side,
                    "entry_time": open_trade["entry_time"],
                    "exit_time": exit_time,
                    "entry_price": open_trade["entry_price"],
                    "exit_price": exit_price,
                    "qty": open_trade["qty"],
                    "risk": open_trade["risk"],
                    "pnl": pnl,
                    "R": round(R, 3),
                    "reason": "SL" if sl_hit and not tp_hit else "TP",
                    "cost": cost,
                })

                open_trade = None

        # =========================
        # ENTRY LOGIC
        # =========================
        if open_trade is None:
            prev_trend = df.iloc[signal_bar - 1]["Trend"] if signal_bar > 0 else 0
            curr_trend = row["Trend"]

            long_signal = (prev_trend == -1 and curr_trend == 1)
            short_signal = (prev_trend == 1 and curr_trend == -1)

            entry_price = float(next_row["Open"])
            entry_time = next_row.name

            risk_amt = cash * risk_pct / 100
            qty = max(1, int(risk_amt / sl_points))
            risk_per_trade = sl_points * qty

            # LONG
            if long_signal and direction in ("Long", "Both"):
                if enable_costs:
                    entry_price *= (1 + SLIPPAGE_PCT / 100)

                sl = entry_price - sl_points
                tp = entry_price + sl_points * rr

                cost = qty * entry_price
                fee = cost * TRANSACTION_COST_PCT / 100 if enable_costs else 0

                if cash >= cost + fee:
                    cash -= (cost + fee)
                    open_trade = {
                        "side": "LONG",
                        "entry_price": entry_price,
                        "entry_time": entry_time,
                        "sl": sl,
                        "tp": tp,
                        "qty": qty,
                        "risk": risk_per_trade,
                    }

            # SHORT
            elif short_signal and direction in ("Short", "Both"):
                if enable_costs:
                    entry_price *= (1 + SLIPPAGE_PCT / 100)

                sl = entry_price + sl_points
                tp = entry_price - sl_points * rr

                margin = qty * entry_price * SHORT_MARGIN_RATIO
                fee = qty * entry_price * TRANSACTION_COST_PCT / 100 if enable_costs else 0

                if cash >= margin + fee:
                    cash -= (margin + fee)
                    open_trade = {
                        "side": "SHORT",
                        "entry_price": entry_price,
                        "entry_time": entry_time,
                        "sl": sl,
                        "tp": tp,
                        "qty": qty,
                        "risk": risk_per_trade,
                    }

        equity_curve.append(cash)

    return pd.DataFrame(trades), equity_curve
