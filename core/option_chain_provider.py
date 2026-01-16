# core/option_chain_provider.py
"""
Option Chain Provider - Fetches option chain data from Upstox API
Works with DuckDB instruments table schema
"""

import requests
import pandas as pd
from datetime import datetime
from core.config import get_access_token
from core.database import get_db

UPSTOX_OPTION_CHAIN_URL = "https://api.upstox.com/v2/option/chain"


class OptionChainProvider:

    def __init__(self):
        self.token = get_access_token()

    def _get_headers(self):
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json"
        }

    def _get_instrument_key(self, symbol: str) -> str:
        """
        Look up instrument_key for a symbol from DuckDB instruments table.

        DuckDB schema:
        - instrument_key VARCHAR PRIMARY KEY
        - trading_symbol VARCHAR
        - segment VARCHAR (NSE_EQ, NSE_FO, NSE_INDEX)
        """
        # Index mappings (hardcoded for speed)
        index_map = {
            "NIFTY": "NSE_INDEX|Nifty 50",
            "BANKNIFTY": "NSE_INDEX|Nifty Bank",
            "FINNIFTY": "NSE_INDEX|Nifty Fin Service",
            "MIDCPNIFTY": "NSE_INDEX|Nifty Midcap Select",
        }

        if symbol.upper() in index_map:
            return index_map[symbol.upper()]

        # For stocks, look up from DuckDB instruments table
        try:
            db = get_db()
            result = db.con.execute("""
                SELECT instrument_key 
                FROM instruments
                WHERE trading_symbol = ?
                  AND segment = 'NSE_EQ'
                LIMIT 1
            """, [symbol.upper()]).fetchone()

            if result:
                print(f"Found instrument_key for {symbol}: {result[0]}")
                return result[0]

            # Try partial match
            result = db.con.execute("""
                SELECT instrument_key, trading_symbol
                FROM instruments
                WHERE trading_symbol LIKE ?
                  AND segment = 'NSE_EQ'
                LIMIT 1
            """, [f"{symbol.upper()}%"]).fetchone()

            if result:
                print(
                    f"Found instrument_key via partial match: {result[0]} ({result[1]})")
                return result[0]

        except Exception as e:
            print(f"DB lookup error: {e}")

        print(f"Could not find instrument_key for {symbol}")
        return None

    def get_nearest_expiry(self, symbol: str) -> str:
        """
        Find nearest expiry for options on a given underlying symbol.

        Looks in instruments table for options (CE/PE) where trading_symbol
        starts with the underlying symbol name.
        """
        try:
            db = get_db()

            # Options have trading_symbol like 'RELIANCE25JAN1500CE'
            # instrument_type will be 'CE' or 'PE'
            result = db.con.execute("""
                SELECT DISTINCT expiry
                FROM instruments
                WHERE trading_symbol LIKE ? || '%'
                  AND (instrument_type = 'CE' OR instrument_type = 'PE')
                  AND expiry >= CURRENT_DATE
                  AND expiry IS NOT NULL
                ORDER BY expiry ASC
                LIMIT 1
            """, [symbol.upper()]).fetchone()

            if result and result[0]:
                expiry_val = result[0]
                if hasattr(expiry_val, 'strftime'):
                    return expiry_val.strftime("%Y-%m-%d")
                return str(expiry_val).split(" ")[0]

        except Exception as e:
            print(f"Expiry lookup error: {e}")

        print(
            f"No expiry found for {symbol}, will try API without expiry filter")
        return None

    def fetch_option_chain(self, signal) -> dict:
        """
        Fetch option chain for a signal's underlying.

        Args:
            signal: UnderlyingSignal object with .symbol attribute

        Returns:
            Normalized chain dict {'CE': [...], 'PE': [...]}
        """
        symbol = signal.symbol  # e.g. RELIANCE

        # 1. Get instrument_key for the underlying
        instrument_key = self._get_instrument_key(symbol)
        if not instrument_key:
            print(f"ERROR: Could not find instrument_key for {symbol}")
            return {"CE": [], "PE": []}

        # 2. Get nearest expiry
        expiry = self.get_nearest_expiry(symbol)

        # 3. Make API call
        params = {"instrument_key": instrument_key}
        if expiry:
            params["expiry_date"] = expiry

        print(
            f"Fetching option chain: instrument_key={instrument_key}, expiry={expiry}")

        try:
            resp = requests.get(
                UPSTOX_OPTION_CHAIN_URL,
                headers=self._get_headers(),
                params=params,
                timeout=10
            )

            print(f"API Status: {resp.status_code}")

            if resp.status_code != 200:
                print(f"API Error Response: {resp.text[:500]}")
                return {"CE": [], "PE": []}

            raw = resp.json()

            # Debug: print structure
            if raw.get("data"):
                print(f"API returned {len(raw['data'])} strikes")
            else:
                print(f"API response structure: {list(raw.keys())}")

            return self._normalize_chain(raw)

        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return {"CE": [], "PE": []}

    def _normalize_chain(self, raw: dict) -> dict:
        """
        Convert Upstox option/chain response to normalized structure.

        Upstox v2 option/chain returns:
        {
            "status": "success",
            "data": [
                {
                    "expiry": "2026-01-29",
                    "pcr": 0.85,
                    "strike_price": 1200.0,
                    "underlying_key": "NSE_EQ|INE002A01018",
                    "underlying_spot_price": 1250.5,
                    "call_options": {
                        "instrument_key": "NSE_FO|...",
                        "market_data": {"ltp": 50.5, "volume": 1000, "oi": 5000, ...},
                        "option_greeks": {"iv": 25.5, "delta": 0.55, "gamma": 0.02, "theta": -5.2, "vega": 0.8}
                    },
                    "put_options": {...}
                },
                ...
            ]
        }
        """
        chain = {"CE": [], "PE": []}

        data = raw.get("data", [])
        if not data:
            print(f"No data in response. Full response: {raw}")
            return chain

        for item in data:
            strike = item.get("strike_price")
            expiry = item.get("expiry")

            # Process CALL options
            call = item.get("call_options")
            if call:
                market_data = call.get("market_data", {}) or {}
                greeks = call.get("option_greeks", {}) or {}

                chain["CE"].append({
                    "instrument_key": call.get("instrument_key"),
                    "strike_price": strike,
                    "strike": strike,  # Alias
                    "ltp": market_data.get("ltp", 0),
                    "volume": market_data.get("volume", 0),
                    "oi": market_data.get("oi", 0),
                    "iv": greeks.get("iv"),
                    "delta": greeks.get("delta"),
                    "gamma": greeks.get("gamma"),
                    "theta": greeks.get("theta"),
                    "vega": greeks.get("vega"),
                    "expiry": expiry,
                    "option_type": "CE",
                })

            # Process PUT options
            put = item.get("put_options")
            if put:
                market_data = put.get("market_data", {}) or {}
                greeks = put.get("option_greeks", {}) or {}

                chain["PE"].append({
                    "instrument_key": put.get("instrument_key"),
                    "strike_price": strike,
                    "strike": strike,  # Alias
                    "ltp": market_data.get("ltp", 0),
                    "volume": market_data.get("volume", 0),
                    "oi": market_data.get("oi", 0),
                    "iv": greeks.get("iv"),
                    "delta": greeks.get("delta"),
                    "gamma": greeks.get("gamma"),
                    "theta": greeks.get("theta"),
                    "vega": greeks.get("vega"),
                    "expiry": expiry,
                    "option_type": "PE",
                })

        print(
            f"Normalized: {len(chain['CE'])} CE, {len(chain['PE'])} PE options")
        return chain
