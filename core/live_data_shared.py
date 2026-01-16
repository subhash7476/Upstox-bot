# core/live_data_shared.py
"""
Shared Live Data Access
=======================
Import this module in any page that needs live data access.
Ensures all pages use the SAME LiveTradingManager instance.

Usage in any page:
    from core.live_data_shared import get_shared_live_manager, ensure_live_data_ready
    
    live_manager = get_shared_live_manager()
    if live_manager:
        df_60m, df_15m, df_5m = live_manager.get_live_mtf_data(instrument_key, lookback_days=60)
"""

import streamlit as st
from typing import Optional, Tuple
import pandas as pd
from datetime import datetime

# Lazy import to avoid circular dependencies
_live_manager_class = None


def _get_live_manager_class():
    global _live_manager_class
    if _live_manager_class is None:
        from core.live_trading_manager import LiveTradingManager
        _live_manager_class = LiveTradingManager
    return _live_manager_class


def get_shared_live_manager():
    """
    Get or create the SINGLE LiveTradingManager instance.

    All pages MUST use this function to access live data.
    This ensures WebSocket connection is shared and data is consistent.

    Returns:
        LiveTradingManager instance or None if initialization failed
    """
    # Use a consistent key across ALL pages
    SESSION_KEY = "live_manager"

    if SESSION_KEY not in st.session_state or st.session_state[SESSION_KEY] is None:
        try:
            LiveTradingManager = _get_live_manager_class()
            st.session_state[SESSION_KEY] = LiveTradingManager()
        except Exception as e:
            print(f"[LIVE] Failed to initialize LiveTradingManager: {e}")
            st.session_state[SESSION_KEY] = None

    return st.session_state[SESSION_KEY]


def ensure_websocket_connected(access_token: str) -> bool:
    """
    Ensure WebSocket is connected and receiving data.

    Args:
        access_token: Upstox API access token

    Returns:
        True if connected, False otherwise
    """
    live_manager = get_shared_live_manager()
    if not live_manager:
        return False

    live_manager.start_websocket_if_needed(access_token)
    return getattr(live_manager, "ws_connected", False)


def get_live_data_status() -> dict:
    """
    Get current status of live data.

    Returns:
        Dict with instruments_with_data, total_candles_today, first_candle, last_candle
    """
    live_manager = get_shared_live_manager()
    if not live_manager:
        return {
            "instruments_with_data": 0,
            "total_candles_today": 0,
            "first_candle": None,
            "last_candle": None,
            "error": "Live manager not available"
        }

    return live_manager.get_live_data_summary()


def get_mtf_data_for_scanning(
    instrument_key: str,
    lookback_days: int = 60
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Get MTF data (60m, 15m, 5m) properly combined with historical data.

    This is the CORRECT way to get data for live scanning.
    It combines historical data + today's live data for proper indicator warmup.

    Args:
        instrument_key: Upstox instrument key
        lookback_days: Days of historical data to include (default 60)

    Returns:
        Tuple of (df_60m, df_15m, df_5m) DataFrames
    """
    live_manager = get_shared_live_manager()
    if not live_manager:
        return None, None, None

    return live_manager.get_live_mtf_data(instrument_key, lookback_days=lookback_days)


def rebuild_live_resampled():
    """
    Rebuild today's resampled data (5m/15m/60m) from 1m cache.
    Call this after WebSocket has collected new data.
    """
    live_manager = get_shared_live_manager()
    if live_manager:
        live_manager.rebuild_today_resampled()


def display_live_status_widget():
    """
    Display a compact live data status widget.
    Can be added to any page's sidebar or header.
    """
    status = get_live_data_status()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("📊 Live Instruments", status.get("instruments_with_data", 0))
    with col2:
        last_candle = status.get("last_candle")
        if last_candle:
            st.metric("🕐 Last Candle", pd.to_datetime(
                last_candle).strftime("%H:%M"))
        else:
            st.metric("🕐 Last Candle", "N/A")


# ============================================================
# MIGRATION HELPERS
# ============================================================

def migrate_old_session_keys():
    """
    Migrate old page-specific session keys to the shared key.
    Call this once at app startup if needed.
    """
    OLD_KEYS = [
        "sq_live_manager",      # Page 4 old key
        "ehma_live_manager",    # Potential old key
    ]

    for old_key in OLD_KEYS:
        if old_key in st.session_state and st.session_state[old_key] is not None:
            # If we don't have a shared manager yet, use the old one
            if "live_manager" not in st.session_state or st.session_state["live_manager"] is None:
                st.session_state["live_manager"] = st.session_state[old_key]
            # Clean up old key
            del st.session_state[old_key]
