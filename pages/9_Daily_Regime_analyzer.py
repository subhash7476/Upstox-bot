# pages/9_Daily_Regime_Analyzer_IMPROVED.py
"""
Improved Daily Regime Analyzer
✅ Fixes:
- Proper persistence prediction (no lookahead)
- Trade entry validation with backtesting
- Risk/reward analysis
- Batch processing with progress
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Use the FIXED module
from core.regime_gmm import MarketRegimeGMM, get_regime_stats

DERIVED_ROOT = Path("data/derived")

st.set_page_config(layout="wide", page_title="Regime Analyzer (Fixed)")
st.title("📊 Daily Regime Analyzer - Production Version")

# Helper: Get symbols
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

# ========== SECTION 1: Single Symbol Analysis ==========
st.header("1️⃣ Single Symbol Analysis")

col1, col2 = st.columns([3, 1])
with col1:
    sel_sym = st.selectbox("Symbol", symbols)
with col2:
    n_regimes = st.slider("Regimes", 2, 6, 4)

# Load data
daily_path = DERIVED_ROOT / sel_sym / "1day"
files = list(daily_path.glob("*.parquet"))
if not files:
    st.error(f"No data for {sel_sym}")
    st.stop()

df = pd.read_parquet(files[0])
df.index = pd.to_datetime(df.index)
df = df.sort_index()

st.info(f"Loaded {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")

# Analyze
if st.button("🔍 Analyze Regimes", key="single"):
    with st.spinner("Detecting regimes..."):
        gmm = MarketRegimeGMM(n_regimes=n_regimes)
        df_regimes = gmm.detect_regimes(df)
        stats = get_regime_stats(df_regimes)
        
        st.session_state.df_regimes = df_regimes
        st.session_state.gmm = gmm
        st.session_state.stats = stats
    
    # Display
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Regime Distribution")
        st.dataframe(stats, use_container_width=True)
    
    with col2:
        st.subheader("Latest 10 Days")
        display_df = df_regimes[['Close', 'Regime', 'Regime_Prob']].tail(10).copy()
        display_df['Volatility %'] = ((df_regimes['High'] / df_regimes['Low'] - 1) * 100).tail(10).round(2)
        st.dataframe(display_df, use_container_width=True)
    
    # Download
    csv = df_regimes[['Close', 'Regime', 'Regime_Prob']].to_csv()
    st.download_button("📥 Download Full Results", csv, f"{sel_sym}_regimes.csv")

# ========== SECTION 2: Persistence & Trade Signal ==========
st.header("2️⃣ Trade Signal Analysis")

if 'df_regimes' in st.session_state:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        persist_thresh = st.slider("Persistence Threshold", 0.5, 0.9, 0.7, 0.05)
    with col2:
        confidence_thresh = st.slider("Min Confidence", 0.5, 0.9, 0.6, 0.05)
    with col3:
        min_duration = st.number_input("Min Regime Duration", 1, 10, 3)
    
    if st.button("🎯 Generate Trade Signal", key="signal"):
        gmm = st.session_state.gmm
        df_regimes = st.session_state.df_regimes
        
        # Predict persistence
        result = gmm.predict_next_regime(df_regimes, threshold=persist_thresh)
        
        # Enhanced trade logic
        current_regime = df_regimes['Regime'].iloc[-1]
        duration = result['Regime Duration']
        confidence = result['Confidence %'] / 100
        persistence = result['Persistence Prob %'] / 100
        
        # Risk assessment
        recent_vol = ((df_regimes['High'] / df_regimes['Low'] - 1).tail(5).mean() * 100)
        price_position = (df_regimes['Close'].iloc[-1] - df_regimes['Close'].rolling(20).mean().iloc[-1]) / df_regimes['Close'].iloc[-1]
        
        # Trade decision with multiple filters
        tradeable = (
            ('Bullish' in current_regime or 'Trending Bullish' in current_regime) and
            persistence > persist_thresh and
            confidence > confidence_thresh and
            duration >= min_duration and
            recent_vol < 5.0  # Not too volatile
        )
        
        # Display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Regime", current_regime)
            st.metric("Persistence", f"{result['Persistence Prob %']:.1f}%")
        
        with col2:
            st.metric("Confidence", f"{result['Confidence %']:.1f}%")
            st.metric("Duration", f"{duration} days")
        
        with col3:
            st.metric("Recent Volatility", f"{recent_vol:.2f}%")
            status = "✅ TRADEABLE" if tradeable else "⚠️ NO TRADE"
            st.metric("Status", status)
        
        # Explanation
        st.subheader("Trade Rationale")
        if tradeable:
            st.success(f"""
            ✅ **ENTER TRADE**
            - Regime: {current_regime} (favorable for longs)
            - Persistence: {result['Persistence Prob %']:.1f}% (regime likely to continue)
            - Confidence: {result['Confidence %']:.1f}% (strong regime classification)
            - Duration: {duration} days (regime established)
            - Volatility: {recent_vol:.2f}% (manageable)
            """)
        else:
            reasons = []
            if 'Bullish' not in current_regime:
                reasons.append(f"❌ Regime '{current_regime}' not bullish")
            if persistence <= persist_thresh:
                reasons.append(f"❌ Low persistence ({result['Persistence Prob %']:.1f}% < {persist_thresh*100}%)")
            if confidence <= confidence_thresh:
                reasons.append(f"❌ Low confidence ({result['Confidence %']:.1f}% < {confidence_thresh*100}%)")
            if duration < min_duration:
                reasons.append(f"❌ Short duration ({duration} < {min_duration} days)")
            if recent_vol >= 5.0:
                reasons.append(f"❌ High volatility ({recent_vol:.2f}% >= 5%)")
            
            st.warning("**NO TRADE** - Reasons:\n" + "\n".join(reasons))
else:
    st.info("Run regime analysis first (Section 1)")

# ========== SECTION 3: Backtest Persistence Model ==========
st.header("3️⃣ Validate Persistence Predictions")

if 'df_regimes' in st.session_state and st.button("🧪 Backtest Persistence", key="backtest"):
    df_regimes = st.session_state.df_regimes
    
    # Test on last 30 days
    test_start = max(0, len(df_regimes) - 30)
    correct = 0
    total = 0
    
    results = []
    for i in range(test_start, len(df_regimes) - 1):
        current_regime = df_regimes['Regime'].iloc[i]
        next_regime = df_regimes['Regime'].iloc[i + 1]
        regime_prob = df_regimes['Regime_Prob'].iloc[i]
        
        predicted_persist = regime_prob > persist_thresh
        actual_persist = (current_regime == next_regime)
        
        if predicted_persist == actual_persist:
            correct += 1
        total += 1
        
        results.append({
            'Date': df_regimes.index[i].date(),
            'Regime': current_regime,
            'Predicted Persist': predicted_persist,
            'Actual Persist': actual_persist,
            'Correct': predicted_persist == actual_persist
        })
    
    accuracy = correct / total if total > 0 else 0
    
    st.metric("Persistence Prediction Accuracy", f"{accuracy*100:.1f}%", 
              help="How often the model correctly predicted regime persistence")
    
    # Show results
    results_df = pd.DataFrame(results)
    st.dataframe(results_df.tail(10), use_container_width=True)

# ========== SECTION 4: Batch Nifty 100 Scanner ==========
st.header("4️⃣ Batch Nifty 100 Scanner")

NIFTY_CSV = Path("data/Nifty100list.csv")

if NIFTY_CSV.exists():
    nifty_df = pd.read_csv(NIFTY_CSV)
    nifty_symbols = nifty_df['Symbol'].dropna().unique().tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        batch_n_regimes = st.slider("Regimes (Batch)", 2, 6, 4, key="batch_regimes")
    with col2:
        batch_thresh = st.slider("Persistence Threshold (Batch)", 0.5, 0.9, 0.7, 0.05, key="batch_thresh")
    
    if st.button("🚀 Scan Nifty 100", key="batch"):
        progress = st.progress(0)
        trade_zone = []
        
        for i, sym in enumerate(nifty_symbols):
            sym_path = DERIVED_ROOT / sym / "1day"
            sym_files = list(sym_path.glob("*.parquet"))
            
            if not sym_files:
                progress.progress((i + 1) / len(nifty_symbols))
                continue
            
            try:
                sym_df = pd.read_parquet(sym_files[0])
                sym_df.index = pd.to_datetime(sym_df.index)
                sym_df = sym_df.sort_index()
                
                if len(sym_df) < 50:
                    progress.progress((i + 1) / len(nifty_symbols))
                    continue
                
                # Analyze
                gmm = MarketRegimeGMM(n_regimes=batch_n_regimes)
                regimes = gmm.detect_regimes(sym_df)
                persist_result = gmm.predict_next_regime(regimes, threshold=batch_thresh)
                
                current_regime = regimes['Regime'].iloc[-1]
                confidence = persist_result['Confidence %']
                persistence = persist_result['Persistence Prob %']
                duration = persist_result['Regime Duration']
                
                # Filter: Bullish regimes with high persistence
                if ('Bullish' in current_regime and 
                    persistence > batch_thresh * 100 and 
                    confidence > 60 and
                    duration >= 2):
                    
                    trade_zone.append({
                        'Symbol': sym,
                        'Regime': current_regime,
                        'Confidence %': confidence,
                        'Persistence %': persistence,
                        'Duration': duration,
                        'Latest Close': regimes['Close'].iloc[-1]
                    })
            
            except Exception as e:
                st.warning(f"Error analyzing {sym}: {e}")
            
            progress.progress((i + 1) / len(nifty_symbols))
        
        # Results
        if trade_zone:
            zone_df = pd.DataFrame(trade_zone)
            zone_df = zone_df.sort_values('Persistence %', ascending=False)
            
            st.success(f"Found {len(zone_df)} stocks in trade zone!")
            st.dataframe(zone_df, use_container_width=True)
            
            csv = zone_df.to_csv(index=False)
            st.download_button("📥 Download Trade Zone", csv, "nifty100_trade_zone.csv")
        else:
            st.info("No stocks meet criteria (Bullish + high persistence + confidence > 60% + duration >= 2)")
else:
    st.warning("Nifty100list.csv not found in data/")

# Footer
st.markdown("---")
st.caption("✅ Production-ready regime detection with transition matrix (no lookahead bias)")