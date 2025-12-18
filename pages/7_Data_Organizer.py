import streamlit as st
import pandas as pd
from pathlib import Path
import sys, os
from datetime import datetime, date, timedelta
import time

# ensure root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(layout="wide", page_title="Data Organizer v2")
st.title("📦 Data Organizer v2 — Batch Processing & Daily Updates")

RAW_ROOT = Path("data/stocks")
DERIVED_ROOT = Path("data/derived")

# =========================================================
# IMPORTS WITH ERROR HANDLING
# =========================================================
try:
    from data.resampler import (
        list_1m_partitions,
        load_1m_data,
        resample_from_1m,
        build_derived_parquet,
    )
    from data.data_manager import fetch_historical_range
    imports_ok = True
except Exception as e:
    st.error(f"Import Error: {e}")
    st.info("Please check if data.resampler and data.data_manager modules exist")
    imports_ok = False

# =========================================================
# HELPER FUNCTIONS
# =========================================================
@st.cache_data(ttl=60)
def get_all_stocks_with_1m_data():
    """Get list of all stocks that have 1-minute data"""
    try:
        if not RAW_ROOT.exists():
            return []
        
        stocks = []
        for symbol_dir in RAW_ROOT.iterdir():
            if symbol_dir.is_dir():
                minute_dir = symbol_dir / "1minute"
                if minute_dir.exists() and list(minute_dir.rglob("*.parquet")):
                    stocks.append(symbol_dir.name)
        
        return sorted(stocks)
    except Exception as e:
        st.error(f"Error scanning stocks: {e}")
        return []


def get_last_data_date(symbol):
    """Find the last date for which data exists"""
    try:
        minute_dir = RAW_ROOT / symbol / "1minute"
        
        if not minute_dir.exists():
            return None
        
        year_dirs = sorted(minute_dir.glob("year=*"), reverse=True)
        
        for year_dir in year_dirs:
            month_dirs = sorted(year_dir.glob("month=*"), reverse=True)
            for month_dir in month_dirs:
                day_dirs = sorted(month_dir.glob("day=*"), reverse=True)
                for day_dir in day_dirs:
                    year = int(year_dir.name.split("=")[1])
                    month = int(month_dir.name.split("=")[1])
                    day = int(day_dir.name.split("=")[1])
                    try:
                        return date(year, month, day)
                    except ValueError:
                        continue
        return None
    except Exception as e:
        st.warning(f"Error getting last date for {symbol}: {e}")
        return None


def check_derived_exists(symbol, timeframe):
    """Check if derived file already exists"""
    try:
        derived_dir = DERIVED_ROOT / symbol / timeframe
        if not derived_dir.exists():
            return False
        merged_file = derived_dir / f"merged_{symbol}_{timeframe}.parquet"
        return merged_file.exists()
    except Exception as e:
        return False


# =========================================================
# SYMBOL LOADER
# =========================================================
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
    except Exception as e:
        st.warning(f"Error loading Nifty100list.csv: {e}")

# Get stocks with data
stocks_with_data = get_all_stocks_with_1m_data()

# Combine both lists
all_symbols = sorted(list(set(symbol_list + stocks_with_data)))

# Debug info
with st.expander("🔍 Debug Info (expand if page is blank)"):
    st.write(f"**ROOT Path:** {ROOT}")
    st.write(f"**RAW_ROOT exists:** {RAW_ROOT.exists()}")
    st.write(f"**Imports OK:** {imports_ok}")
    st.write(f"**Nifty100list.csv exists:** {LIST_FILE.exists()}")
    st.write(f"**Symbols from CSV:** {len(symbol_list)}")
    st.write(f"**Stocks with data:** {len(stocks_with_data)}")
    st.write(f"**Combined symbols:** {len(all_symbols)}")
    if all_symbols:
        st.write(f"**First 5 symbols:** {all_symbols[:5]}")

# Stop if no imports
if not imports_ok:
    st.stop()

# Stop if no symbols
if not all_symbols:
    st.error("❌ No symbols found!")
    st.info("""
    **Possible issues:**
    1. No `data/Nifty100list.csv` file found
    2. No stocks with 1-minute data in `data/stocks/` directory
    
    **To fix:**
    - Go to **Page 2 (Fetch & Manage Data)** and download data first
    - Or ensure `data/Nifty100list.csv` exists
    """)
    st.stop()

# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================
if 'selected_symbols_org' not in st.session_state:
    default_symbol = "RELIANCE" if "RELIANCE" in all_symbols else all_symbols[0]
    st.session_state.selected_symbols_org = [default_symbol]

if 'update_stocks_list' not in st.session_state:
    st.session_state.update_stocks_list = stocks_with_data[:5] if len(stocks_with_data) > 5 else stocks_with_data

# =========================================================
# TABS FOR ORGANIZATION
# =========================================================
tab1, tab2 = st.tabs(["📊 Organize Data", "🔄 Daily Update"])

# =========================================================
# TAB 1: ORGANIZE DATA (BATCH PROCESSING)
# =========================================================
with tab1:
    st.markdown("### Resample 1-minute data to higher timeframes")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Quick selection buttons (BEFORE multiselect)
        st.markdown("**Quick Select:**")
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            if st.button("📊 All", key="select_all_org", use_container_width=True):
                st.session_state.selected_symbols_org = all_symbols
        with col_b:
            if st.button("🔄 Clear", key="clear_org", use_container_width=True):
                st.session_state.selected_symbols_org = []
        with col_c:
            if st.button("🔝 First 10", key="first10_org", use_container_width=True):
                st.session_state.selected_symbols_org = all_symbols[:10]
        with col_d:
            if st.button("🎲 Random 10", key="random10_org", use_container_width=True):
                import random
                st.session_state.selected_symbols_org = random.sample(all_symbols, min(10, len(all_symbols)))
        
        st.markdown("---")
        
        # Multi-select for stocks
        selected_symbols = st.multiselect(
            "Select Stocks to Process",
            options=all_symbols,
            default=st.session_state.selected_symbols_org,
            help="Select one or more stocks to process",
            key="multiselect_org"
        )
        # Sync session_state
        st.session_state.selected_symbols_org = selected_symbols
        
        # Multi-select for timeframes
        target_timeframes = st.multiselect(
            "Target Timeframes",
            options=["5minute", "15minute", "30minute", "60minute", "240minute", "1day"],
            default=["15minute", "30minute", "60minute"],
            help="240minute = 4 Hour, 1day = Daily"
        )
        
        overwrite = st.checkbox(
            "Overwrite existing derived files (rebuild)",
            value=False,
            key="overwrite_org"
        )
    
    with col2:
        st.markdown("### Preview")
        if selected_symbols and target_timeframes:
            total_tasks = len(selected_symbols) * len(target_timeframes)
            st.metric("Stocks Selected", len(selected_symbols))
            st.metric("Timeframes Selected", len(target_timeframes))
            st.metric("Total Tasks", total_tasks)
            
            # Show first few stocks with data status
            st.markdown("**Data Status:**")
            for sym in selected_symbols[:5]:
                try:
                    parts = list_1m_partitions(sym)
                    if parts:
                        st.success(f"✅ {sym}: {len(parts)} days", icon="✅")
                    else:
                        st.warning(f"⚠️ {sym}: No data", icon="⚠️")
                except:
                    st.error(f"❌ {sym}: Error", icon="❌")
            if len(selected_symbols) > 5:
                st.caption(f"...and {len(selected_symbols) - 5} more")
        else:
            st.info("Select stocks and timeframes")
    
    st.divider()
    
    # Run button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_organize = st.button(
            "▶️ Generate Derived Data",
            use_container_width=True,
            type="primary",
            key="run_org",
            disabled=not (selected_symbols and target_timeframes)
        )
    
    # Processing
    if run_organize:
        st.markdown("### ⚙️ Processing...")
        
        results = []
        success_count = 0
        failed_count = 0
        skipped_count = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        start_time = time.time()
        
        for idx, symbol in enumerate(selected_symbols):
            # Check if symbol has 1-minute data
            try:
                parts = list_1m_partitions(symbol)
            except Exception as e:
                parts = []
            
            if not parts:
                for tf in target_timeframes:
                    results.append({
                        'Symbol': symbol,
                        'Timeframe': tf,
                        'Status': '❌ No 1m Data',
                        'Rows': 0,
                        'Message': 'No 1-minute partitions found'
                    })
                    failed_count += 1
                continue
            
            # Process each timeframe
            for tf in target_timeframes:
                # Update progress
                total_tasks = len(selected_symbols) * len(target_timeframes)
                current_task = idx * len(target_timeframes) + target_timeframes.index(tf) + 1
                progress = current_task / total_tasks
                progress_bar.progress(progress)
                status_text.text(f"Processing {symbol} - {tf} ({current_task}/{total_tasks})...")
                
                # Check if exists
                exists = check_derived_exists(symbol, tf)
                if exists and not overwrite:
                    results.append({
                        'Symbol': symbol,
                        'Timeframe': tf,
                        'Status': '⚠️ Skipped',
                        'Rows': '-',
                        'Message': 'File exists (enable overwrite to rebuild)'
                    })
                    skipped_count += 1
                    continue
                
                # Process
                try:
                    df_1m = load_1m_data(symbol)
                    if df_1m is None or df_1m.empty:
                        raise RuntimeError("Failed to load 1-minute data")
                    
                    df_tf = resample_from_1m(df_1m, tf)
                    if df_tf.empty:
                        raise RuntimeError("Resampled dataframe is empty")
                    
                    out_file = build_derived_parquet(symbol, tf, overwrite=overwrite)
                    
                    results.append({
                        'Symbol': symbol,
                        'Timeframe': tf,
                        'Status': '✅ Success',
                        'Rows': len(df_tf),
                        'Message': f'Saved to {out_file.name}'
                    })
                    success_count += 1
                    
                except FileExistsError:
                    results.append({
                        'Symbol': symbol,
                        'Timeframe': tf,
                        'Status': '⚠️ Exists',
                        'Rows': '-',
                        'Message': 'File exists (enable overwrite)'
                    })
                    skipped_count += 1
                    
                except Exception as e:
                    results.append({
                        'Symbol': symbol,
                        'Timeframe': tf,
                        'Status': '❌ Failed',
                        'Rows': 0,
                        'Message': str(e)[:80]
                    })
                    failed_count += 1
        
        # Complete
        progress_bar.progress(1.0)
        status_text.text("✅ Processing Complete!")
        
        elapsed_time = time.time() - start_time
        
        st.success(f"""
        **Processing Complete!**
        - ✅ Success: {success_count}
        - ⚠️ Skipped: {skipped_count}
        - ❌ Failed: {failed_count}
        - ⏱️ Time: {elapsed_time:.2f}s ({elapsed_time/60:.1f} minutes)
        """)
        
        # Results table
        with st.expander("📊 Detailed Results", expanded=True):
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            # Download
            csv = results_df.to_csv(index=False)
            st.download_button(
                "📥 Download Results CSV",
                csv,
                f"organize_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )

# =========================================================
# TAB 2: DAILY UPDATE
# =========================================================
with tab2:
    st.markdown("### Automatic Incremental Updates")
    st.info("🕐 Best run after market close (4:00 PM IST). Updates only missing days since last fetch.")
    
    # Configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Quick buttons (BEFORE multiselect)
        st.markdown("**Quick Select:**")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📊 Select All Stocks", use_container_width=True, key="select_all_update"):
                st.session_state.update_stocks_list = stocks_with_data
        with col_b:
            if st.button("🔄 Clear Selection", use_container_width=True, key="clear_update"):
                st.session_state.update_stocks_list = []
        
        st.markdown("---")
        
        # Stock selection for update
        update_stocks = st.multiselect(
            "Stocks to Update",
            options=stocks_with_data if stocks_with_data else all_symbols,
            default=st.session_state.update_stocks_list,
            help="Select stocks to fetch missing data for",
            key="update_stocks_multi"
        )
        # Sync session_state
        st.session_state.update_stocks_list = update_stocks
        
        # Auto-resample option
        auto_resample = st.checkbox(
            "Auto-Resample After Fetch",
            value=True,
            help="Automatically update derived timeframes after fetching new data",
            key="auto_resample"
        )
        
        if auto_resample:
            resample_tfs = st.multiselect(
                "Timeframes to Update",
                ["15minute", "30minute", "60minute", "240minute", "1day"],
                default=["15minute", "30minute", "60minute"],
                key="resample_tfs"
            )
    
    with col2:
        st.markdown("### Status")
        if update_stocks:
            # Check status
            up_to_date = 0
            needs_update = 0
            total_days_missing = 0
            
            for sym in update_stocks:
                last_date = get_last_data_date(sym)
                if last_date:
                    days_behind = (date.today() - last_date).days
                    if days_behind <= 1:
                        up_to_date += 1
                    else:
                        needs_update += 1
                        total_days_missing += days_behind
            
            st.metric("Total Stocks", len(update_stocks))
            st.metric("Up to Date", up_to_date)
            st.metric("Needs Update", needs_update)
            st.metric("Days to Fetch", total_days_missing)
        else:
            st.info("Select stocks above")
    
    st.divider()
    
    # Show detailed status
    if update_stocks:
        with st.expander("📋 Detailed Stock Status"):
            status_data = []
            for sym in update_stocks[:20]:  # Show first 20
                last_date = get_last_data_date(sym)
                if last_date:
                    days_behind = (date.today() - last_date).days
                    status_data.append({
                        'Symbol': sym,
                        'Last Date': last_date.strftime('%Y-%m-%d'),
                        'Days Behind': days_behind,
                        'Status': '✅ Current' if days_behind <= 1 else f'⚠️ {days_behind} days'
                    })
                else:
                    status_data.append({
                        'Symbol': sym,
                        'Last Date': 'No data',
                        'Days Behind': '-',
                        'Status': '❌ No data'
                    })
            
            if len(update_stocks) > 20:
                st.caption(f"Showing first 20 of {len(update_stocks)} stocks")
            
            st.dataframe(pd.DataFrame(status_data), use_container_width=True, hide_index=True)
    
    # Run update button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_update = st.button(
            "🔄 Run Daily Update",
            use_container_width=True,
            type="primary",
            key="run_update",
            disabled=not update_stocks
        )
    
    # Processing
    if run_update:
        st.markdown("### ⚙️ Updating...")
        
        results = []
        success_count = 0
        failed_count = 0
        no_data_count = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        start_time = time.time()
        
        for idx, symbol in enumerate(update_stocks):
            # Update progress
            progress = idx / len(update_stocks)
            progress_bar.progress(progress)
            status_text.text(f"Updating {symbol} ({idx + 1}/{len(update_stocks)})...")
            
            # Get last date
            last_date = get_last_data_date(symbol)
            if not last_date:
                results.append({
                    'Symbol': symbol,
                    'Status': '❌ No Base Data',
                    'Days Fetched': 0,
                    'Message': 'No existing data found'
                })
                failed_count += 1
                continue
            
            # Calculate date range
            start_date = last_date + timedelta(days=1)
            end_date = date.today()
            
            # Skip if already up to date
            if start_date > end_date:
                results.append({
                    'Symbol': symbol,
                    'Status': '✅ Up to Date',
                    'Days Fetched': 0,
                    'Message': f'Last data: {last_date}'
                })
                no_data_count += 1
                continue
            
            # Fetch missing data
            try:
                saved_files = fetch_historical_range(
                    symbol=symbol,
                    segment="NSE_EQ",
                    interval="1minute",
                    from_date=start_date,
                    to_date=end_date,
                    force=False
                )
                
                if saved_files:
                    results.append({
                        'Symbol': symbol,
                        'Status': '✅ Success',
                        'Days Fetched': len(saved_files),
                        'Message': f'Updated from {start_date} to {end_date}'
                    })
                    success_count += 1
                    
                    # Auto-resample
                    if auto_resample and resample_tfs:
                        for tf in resample_tfs:
                            try:
                                build_derived_parquet(symbol, tf, overwrite=True)
                            except:
                                pass
                else:
                    results.append({
                        'Symbol': symbol,
                        'Status': '⚠️ No Data',
                        'Days Fetched': 0,
                        'Message': 'No data returned (holidays/weekends)'
                    })
                    no_data_count += 1
                    
            except Exception as e:
                results.append({
                    'Symbol': symbol,
                    'Status': '❌ Failed',
                    'Days Fetched': 0,
                    'Message': str(e)[:80]
                })
                failed_count += 1
        
        # Complete
        progress_bar.progress(1.0)
        status_text.text("✅ Update Complete!")
        
        elapsed_time = time.time() - start_time
        
        st.success(f"""
        **Update Complete!**
        - ✅ Success: {success_count}
        - ⚠️ No Data/Up to Date: {no_data_count}
        - ❌ Failed: {failed_count}
        - ⏱️ Time: {elapsed_time:.2f}s ({elapsed_time/60:.1f} minutes)
        """)
        
        # Results
        with st.expander("📊 Detailed Results", expanded=True):
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                "📥 Download Results CSV",
                csv,
                f"daily_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
        
        st.info("💡 Data updated! Refresh to see latest status.")

# =========================================================
# FOOTER
# =========================================================
st.divider()
st.caption("""
**Data Organizer v2** | Batch Processing & Daily Updates | 
Timeframes: 5m, 15m, 30m, 1h, 4h, Daily | 
Best update time: After 4:00 PM IST
""")