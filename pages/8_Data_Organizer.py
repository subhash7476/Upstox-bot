"""
Data Organizer V2 - Enhanced Batch Processing
Resample 1-minute data to multiple timeframes for multiple stocks in one go
"""

import sys
import os
from pathlib import Path

# Ensure root is in path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd
from datetime import datetime
import time

from data.resampler import load_1m_data, resample_from_1m, build_derived_parquet

# Page config
st.set_page_config(
    page_title="Data Organizer V2 - Enhanced",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Data Organizer V2 - Batch Processing")
st.markdown("**Resample 1-minute data to multiple timeframes for multiple stocks**")

# Timeframe mapping
TIMEFRAME_MAP = {
    "5 Minute": "5minute",
    "15 Minute": "15minute",
    "30 Minute": "30minute",
    "1 Hour": "60minute",
    "4 Hour": "240minute",
    "Daily": "1day"
}

# Helper Functions
@st.cache_data(ttl=300)
def get_available_stocks():
    """Scan data/stocks/ for symbols with 1-minute data"""
    stocks_dir = Path(ROOT) / "data" / "stocks"
    
    if not stocks_dir.exists():
        return []
    
    available_stocks = []
    for symbol_dir in stocks_dir.iterdir():
        if symbol_dir.is_dir():
            minute_dir = symbol_dir / "1minute"
            if minute_dir.exists():
                # Check if there's actual data
                year_dirs = list(minute_dir.glob("year=*"))
                if year_dirs:
                    available_stocks.append(symbol_dir.name)
    
    return sorted(available_stocks)


def count_1m_files(symbol):
    """Count number of 1-minute partition files for a symbol"""
    stocks_dir = Path(ROOT) / "data" / "stocks"
    minute_dir = stocks_dir / symbol / "1minute"
    
    if not minute_dir.exists():
        return 0
    
    count = 0
    for parquet_file in minute_dir.rglob("*.parquet"):
        count += 1
    
    return count


def check_derived_exists(symbol, timeframe):
    """Check if derived file already exists"""
    derived_dir = Path(ROOT) / "data" / "derived" / symbol / timeframe
    
    if not derived_dir.exists():
        return False
    
    merged_file = derived_dir / f"merged_{symbol}_{timeframe}.parquet"
    return merged_file.exists()


def process_single_stock(symbol, timeframes, overwrite=False):
    """
    Process a single stock for selected timeframes
    Returns: dict with results
    """
    results = {
        'symbol': symbol,
        'timeframes': {},
        'success': True,
        'error': None
    }
    
    try:
        # Load 1-minute data
        df_1m = load_1m_data(symbol)
        
        if df_1m is None or df_1m.empty:
            results['success'] = False
            results['error'] = "No 1-minute data found"
            return results
        
        # Process each timeframe
        for tf_name, tf_key in timeframes.items():
            try:
                # Check if exists
                exists = check_derived_exists(symbol, tf_key)
                
                if exists and not overwrite:
                    results['timeframes'][tf_name] = {
                        'status': 'skipped',
                        'rows': None,
                        'message': 'Already exists'
                    }
                    continue
                
                # Resample
                df_resampled = resample_from_1m(df_1m, tf_key)
                
                if df_resampled is None or df_resampled.empty:
                    results['timeframes'][tf_name] = {
                        'status': 'failed',
                        'rows': 0,
                        'message': 'Resampling returned empty'
                    }
                    continue
                
                # Save
                output_file = build_derived_parquet(symbol, tf_key, overwrite=overwrite)
                
                results['timeframes'][tf_name] = {
                    'status': 'success',
                    'rows': len(df_resampled),
                    'message': f'Saved to {output_file.name}'
                }
                
            except Exception as e:
                results['timeframes'][tf_name] = {
                    'status': 'error',
                    'rows': None,
                    'message': str(e)
                }
        
        return results
        
    except Exception as e:
        results['success'] = False
        results['error'] = str(e)
        return results


# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Overwrite option
    overwrite_existing = st.checkbox(
        "Overwrite Existing Files",
        value=False,
        help="If checked, will regenerate files even if they exist"
    )
    
    st.divider()
    
    # Info
    st.info("""
    **Process Flow:**
    1. Select stocks with 1-min data
    2. Choose target timeframes
    3. Click 'Process Selected'
    4. Monitor progress
    """)
    
    # Stats
    available_stocks = get_available_stocks()
    st.metric("Available Stocks", len(available_stocks))

# Main Content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📁 Stock Selection")
    
    if not available_stocks:
        st.warning("No stocks with 1-minute data found in `data/stocks/`")
        st.stop()
    
    # Multi-select for stocks
    selected_stocks = st.multiselect(
        "Select Stocks to Process",
        options=available_stocks,
        default=None,
        help="Select one or more stocks. Use Ctrl/Cmd to select multiple."
    )
    
    # Quick selection buttons
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        if st.button("📊 Select All", use_container_width=True):
            selected_stocks = available_stocks
            st.rerun()
    with col_b:
        if st.button("🔄 Clear", use_container_width=True):
            selected_stocks = []
            st.rerun()
    with col_c:
        if st.button("🔝 First 10", use_container_width=True):
            selected_stocks = available_stocks[:10]
            st.rerun()
    with col_d:
        if st.button("🎲 Random 10", use_container_width=True):
            import random
            selected_stocks = random.sample(available_stocks, min(10, len(available_stocks)))
            st.rerun()

with col2:
    st.subheader("⏱️ Timeframe Selection")
    
    # Multi-select for timeframes
    selected_timeframes = st.multiselect(
        "Target Timeframes",
        options=list(TIMEFRAME_MAP.keys()),
        default=["15 Minute", "30 Minute", "1 Hour"],
        help="Select one or more timeframes"
    )
    
    # Quick select
    if st.button("Select All Timeframes", use_container_width=True):
        selected_timeframes = list(TIMEFRAME_MAP.keys())
        st.rerun()

st.divider()

# Preview Section
if selected_stocks and selected_timeframes:
    st.subheader("📋 Processing Preview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Stocks Selected", len(selected_stocks))
    with col2:
        st.metric("Timeframes Selected", len(selected_timeframes))
    with col3:
        total_tasks = len(selected_stocks) * len(selected_timeframes)
        st.metric("Total Tasks", total_tasks)
    
    # Show sample of what will be processed
    with st.expander("🔍 View Details"):
        preview_data = []
        for stock in selected_stocks[:10]:  # Show first 10
            file_count = count_1m_files(stock)
            for tf_name in selected_timeframes:
                tf_key = TIMEFRAME_MAP[tf_name]
                exists = check_derived_exists(stock, tf_key)
                status = "⚠️ Exists" if exists else "✅ New"
                
                preview_data.append({
                    'Symbol': stock,
                    'Timeframe': tf_name,
                    'Status': status,
                    '1m Files': file_count
                })
        
        if len(selected_stocks) > 10:
            st.caption(f"Showing first 10 of {len(selected_stocks)} stocks...")
        
        st.dataframe(
            pd.DataFrame(preview_data),
            use_container_width=True,
            hide_index=True
        )
    
    st.divider()
    
    # Process Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_button = st.button(
            "🚀 Process Selected Stocks",
            type="primary",
            use_container_width=True
        )
    
    # Processing Logic
    if process_button:
        st.subheader("⚙️ Processing...")
        
        # Prepare timeframe dict
        timeframes_to_process = {
            tf_name: TIMEFRAME_MAP[tf_name] 
            for tf_name in selected_timeframes
        }
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Results storage
        all_results = []
        success_count = 0
        failed_count = 0
        skipped_count = 0
        
        # Process each stock
        start_time = time.time()
        
        for idx, symbol in enumerate(selected_stocks):
            # Update progress
            progress = (idx) / len(selected_stocks)
            progress_bar.progress(progress)
            status_text.text(f"Processing {symbol} ({idx + 1}/{len(selected_stocks)})...")
            
            # Process
            result = process_single_stock(
                symbol, 
                timeframes_to_process, 
                overwrite=overwrite_existing
            )
            
            all_results.append(result)
            
            # Count outcomes
            if result['success']:
                for tf_result in result['timeframes'].values():
                    if tf_result['status'] == 'success':
                        success_count += 1
                    elif tf_result['status'] == 'skipped':
                        skipped_count += 1
                    else:
                        failed_count += 1
            else:
                failed_count += len(timeframes_to_process)
        
        # Complete
        progress_bar.progress(1.0)
        status_text.text("✅ Processing Complete!")
        
        elapsed_time = time.time() - start_time
        
        st.success(f"""
        **Processing Complete!**
        - ✅ Success: {success_count}
        - ⚠️ Skipped: {skipped_count}
        - ❌ Failed: {failed_count}
        - ⏱️ Time: {elapsed_time:.2f}s
        """)
        
        # Detailed Results
        with st.expander("📊 Detailed Results", expanded=True):
            results_data = []
            
            for result in all_results:
                symbol = result['symbol']
                
                if not result['success']:
                    results_data.append({
                        'Symbol': symbol,
                        'Timeframe': 'ALL',
                        'Status': '❌ Failed',
                        'Rows': None,
                        'Message': result['error']
                    })
                    continue
                
                for tf_name, tf_result in result['timeframes'].items():
                    status_icon = {
                        'success': '✅',
                        'skipped': '⚠️',
                        'failed': '❌',
                        'error': '❌'
                    }.get(tf_result['status'], '❓')
                    
                    results_data.append({
                        'Symbol': symbol,
                        'Timeframe': tf_name,
                        'Status': f"{status_icon} {tf_result['status'].title()}",
                        'Rows': tf_result['rows'] if tf_result['rows'] else '-',
                        'Message': tf_result['message']
                    })
            
            results_df = pd.DataFrame(results_data)
            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                "📥 Download Results CSV",
                csv,
                f"data_organizer_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )

else:
    st.info("👆 Select stocks and timeframes above to begin")

# Footer
st.divider()
st.caption(f"""
**Data Organizer V2 Enhanced** | Batch processing enabled | 
Data Location: `{Path(ROOT) / 'data'}` | 
Timeframes: 5m, 15m, 30m, 1h, 4h, Daily
""")