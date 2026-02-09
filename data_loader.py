"""
Centralized data loader for Kuuma Booking Analyzer.
Provides shared data loading and caching functions used across all pages.
"""

import streamlit as st
import pandas as pd
import hashlib
import time
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Dict, Optional, Callable
from datetime import datetime, timedelta

# Demo mode: obfuscates sensitive data so the dashboard can be shown externally.
# Set to False to see real data.
DEMO_MODE = True
_DEMO_MULTIPLIER = 1.37


def apply_demo_transform(df):
    """Apply demo transformations to obfuscate sensitive business data.

    - Multiplies financial columns (Total gross) by a fixed factor
    - Multiplies participant counts by the same factor (rounded to int)
    - Anonymizes email addresses with consistent hashing
    """
    if df is None or not DEMO_MODE:
        return df

    df = df.copy()

    # Financial columns
    if 'Total gross' in df.columns:
        df['Total gross'] = (df['Total gross'] * _DEMO_MULTIPLIER).round(2)

    # Participant counts (must stay integer)
    for col in ['Participants', 'Adults']:
        if col in df.columns:
            df[col] = (df[col] * _DEMO_MULTIPLIER).round().astype(int)

    # Anonymize email addresses with consistent hash (same email -> same fake email)
    if 'Email address' in df.columns:
        def anonymize_email(email):
            if pd.isna(email) or email == '':
                return email
            h = hashlib.md5(str(email).encode()).hexdigest()[:8]
            return f"customer_{h}@demo.com"
        df['Email address'] = df['Email address'].apply(anonymize_email)

    # Anonymize names
    if 'Name' in df.columns:
        def anonymize_name(name):
            if pd.isna(name) or name == '':
                return name
            h = hashlib.md5(str(name).encode()).hexdigest()[:6]
            return f"Customer {h.upper()}"
        df['Name'] = df['Name'].apply(anonymize_name)

    return df


def init_session_state():
    """Initialize all session state variables for data storage."""
    defaults = {
        # Existing keys
        'df1': None,           # Booking creation dates
        'df2': None,           # Visit dates
        'google_ads_df': None, # Google Ads data
        'meta_ads_df': None,   # Meta Ads data
        'drive_loaded': False, # Flag for Drive loading
        'processed_data': None, # Cached processed data
        'data_hash': None,     # Hash to detect data changes

        # Bookeo-specific state
        'bookeo_loaded': False,        # Flag for Bookeo API loading
        'bookeo_last_refresh': None,   # Timestamp of last API refresh
        'bookeo_errors': {},           # {account_key: error_message}
        'data_source': 'auto',         # 'auto', 'bookeo', 'drive', 'upload'
        'loading_status': {},          # {account_key: 'pending'|'loading'|'complete'|'error'}
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def is_data_loaded():
    """Check if booking data is loaded in session state."""
    return st.session_state.get('df1') is not None or st.session_state.get('df2') is not None


def is_marketing_data_loaded():
    """Check if marketing data is loaded in session state."""
    return (st.session_state.get('google_ads_df') is not None or
            st.session_state.get('meta_ads_df') is not None)


def is_bookeo_data_loaded() -> bool:
    """Check if Bookeo data is currently loaded in session state."""
    return st.session_state.get('bookeo_loaded', False)


def estimate_loading_time(days: int, num_accounts: int = 3) -> str:
    """
    Estimate loading time based on date range and number of accounts.

    Based on observed performance (calibrated 2026-01-10):
    - ~5 seconds per day of data (API pagination + rate limits)
    - 3 accounts fetch in parallel but each has overhead
    - Historical data (>7 days old) is cached after first load

    Args:
        days: Number of days in the date range
        num_accounts: Number of Bookeo accounts configured

    Returns:
        Formatted string like "~2 minutes" or "~15 minutes",
        or empty string if estimate not needed (< 7 days)
    """
    if days < 7:
        return ""

    # Base constants (calibrated from actual testing: 160 days ≈ 15 min)
    SECONDS_PER_DAY = 4.5  # ~4.5 seconds per day (includes API pagination)
    BASE_OVERHEAD = 10  # Initial connection overhead

    # Calculate base time
    estimated_seconds = BASE_OVERHEAD + (days * SECONDS_PER_DAY)

    # Account overhead (accounts fetch in parallel, ~25% slower than single)
    if num_accounts > 1:
        estimated_seconds *= 1.25

    # Round to reasonable precision
    estimated_seconds = round(estimated_seconds)

    # Format the output
    if estimated_seconds < 60:
        return f"~{estimated_seconds} seconds"
    else:
        minutes = round(estimated_seconds / 60)
        return f"~{minutes} minute{'s' if minutes > 1 else ''}"


def _fetch_account_data(
    account_key: str,
    start_date: datetime,
    end_date: datetime,
    include_canceled: bool
) -> Tuple[str, Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
    """Fetch data for a single account. Returns (account_key, df1, df2, error)."""
    from bookeo_transformer import fetch_and_transform_bookings

    try:
        df1, df2, error = fetch_and_transform_bookings(
            account_key=account_key,
            start_date=start_date,
            end_date=end_date,
            include_canceled=include_canceled
        )
        return (account_key, df1, df2, error)
    except Exception as e:
        return (account_key, None, None, f"Unexpected error: {e}")


def load_bookeo_data(
    start_date,
    end_date,
    include_canceled: bool = False,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    max_workers: int = 3
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, str]]:
    """
    Load booking data from all configured Bookeo accounts in parallel.

    Returns (df1, df2, errors_dict) where errors_dict maps account_key -> error message.
    """
    from bookeo_config import get_bookeo_accounts, is_bookeo_configured
    from bookeo_transformer import merge_multi_account_data

    def update_progress(percent: int, text: str):
        if progress_callback:
            progress_callback(percent, text)

    if not is_bookeo_configured():
        return None, None, {'config': 'Bookeo API not configured in secrets'}

    accounts = get_bookeo_accounts()
    if not accounts:
        return None, None, {'config': 'No Bookeo accounts configured'}

    account_data = {}
    errors = {}
    total = len(accounts)

    # Initialize per-account loading status
    st.session_state.loading_status = {acc.key: 'pending' for acc in accounts}

    update_progress(10, f"Fetching from {total} accounts in parallel...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_account = {
            executor.submit(_fetch_account_data, acc.key, start_date, end_date, include_canceled): acc
            for acc in accounts
        }

        for i, future in enumerate(as_completed(future_to_account), 1):
            account = future_to_account[future]

            try:
                account_key, df1, df2, error = future.result()
                if error:
                    errors[account_key] = error
                    st.session_state.loading_status[account_key] = 'error'
                elif df1 is not None and len(df1) > 0:
                    account_data[account_key] = (df1, df2)
                    st.session_state.loading_status[account_key] = 'complete'
                else:
                    st.session_state.loading_status[account_key] = 'complete'
            except Exception as e:
                errors[account.key] = f"Thread error: {e}"
                st.session_state.loading_status[account.key] = 'error'

            update_progress(int((i / total) * 70) + 20, f"Loaded {account.name} ({i}/{total})")

    if not account_data:
        return None, None, errors

    update_progress(92, "Merging data from all accounts...")
    merged_df1, merged_df2 = merge_multi_account_data(account_data)
    update_progress(98, "Finalizing...")

    return merged_df1, merged_df2, errors


def _format_elapsed_time(seconds: float) -> str:
    """Format elapsed time as 'Xs' or 'M:SS'."""
    if seconds < 60:
        return f"{int(seconds)}s"
    else:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"


def _inject_ticking_timer(days_range: int, time_estimate: str):
    """Inject JavaScript to create a ticking timer in the status label."""
    estimate_text = f" / {time_estimate}" if time_estimate else ""
    js_code = f"""
    <script>
    (function() {{
        // Find the status label (summary element in details)
        const findStatusLabel = () => {{
            const summaries = document.querySelectorAll('details[data-testid="stExpander"] summary span');
            for (const span of summaries) {{
                if (span.textContent && span.textContent.includes('Fetching')) {{
                    return span;
                }}
            }}
            return null;
        }};

        let startTime = Date.now();
        let timerInterval = null;

        const formatTime = (seconds) => {{
            if (seconds < 60) {{
                return seconds + 's';
            }} else {{
                const mins = Math.floor(seconds / 60);
                const secs = seconds % 60;
                return mins + ':' + String(secs).padStart(2, '0');
            }}
        }};

        const updateTimer = () => {{
            const label = findStatusLabel();
            if (label && label.textContent.includes('Fetching')) {{
                const elapsed = Math.floor((Date.now() - startTime) / 1000);
                const elapsedStr = formatTime(elapsed);
                label.textContent = 'Fetching {days_range} days of booking data... (' + elapsedStr + '{estimate_text})';
            }} else if (label && (label.textContent.includes('Complete') || label.textContent.includes('error'))) {{
                // Stop timer when loading completes
                if (timerInterval) {{
                    clearInterval(timerInterval);
                    timerInterval = null;
                }}
            }}
        }};

        // Start timer after a short delay to let Streamlit render
        setTimeout(() => {{
            timerInterval = setInterval(updateTimer, 1000);
        }}, 100);

        // Clean up after 5 minutes (safety)
        setTimeout(() => {{
            if (timerInterval) {{
                clearInterval(timerInterval);
            }}
        }}, 300000);
    }})();
    </script>
    """
    st.components.v1.html(js_code, height=0)


def load_bookeo_data_with_status(
    start_date,
    end_date,
    include_canceled: bool = False,
    date_basis: str = 'created'
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, str]]:
    """
    Load booking data with detailed st.status() UI showing per-account progress.

    Args:
        start_date: Start of date range
        end_date: End of date range
        include_canceled: Include canceled bookings
        date_basis: 'created' to filter by booking creation date, 'visit' for visit date

    Returns (df1, df2, errors_dict) where errors_dict maps account_key -> error message.
    """
    from bookeo_config import get_bookeo_accounts, is_bookeo_configured
    from bookeo_transformer import merge_multi_account_data, fetch_and_transform_bookings

    if not is_bookeo_configured():
        return None, None, {'config': 'Bookeo API not configured in secrets'}

    accounts = get_bookeo_accounts()
    if not accounts:
        return None, None, {'config': 'No Bookeo accounts configured'}

    # Fetch exactly what the user selected - no hidden buffers
    fetch_start = start_date
    fetch_end = end_date

    # Calculate days for display and time estimate
    days_range = (end_date - start_date).days
    fetch_days = (fetch_end - fetch_start).days
    time_estimate = estimate_loading_time(fetch_days, len(accounts))

    # Track elapsed time
    start_time = time.time()

    def get_time_label():
        """Get status label with elapsed time vs estimate."""
        elapsed = time.time() - start_time
        elapsed_str = _format_elapsed_time(elapsed)
        if time_estimate:
            return f"Fetching {days_range} days of booking data... ({elapsed_str} / {time_estimate})"
        else:
            return f"Fetching {days_range} days of booking data... ({elapsed_str})"

    account_data = {}
    errors = {}

    # Inject JavaScript ticking timer
    _inject_ticking_timer(days_range, time_estimate)

    with st.status(get_time_label(), expanded=True) as status:
        for i, account in enumerate(accounts, 1):
            # Update status label with current elapsed time
            status.update(label=get_time_label())

            # Create placeholder for live updates
            status_placeholder = st.empty()
            status_placeholder.write(f"**{account.name}** - Connecting...")

            # Callback to update status in real-time
            def make_status_callback(placeholder, name, status_obj, time_fn):
                def callback(msg):
                    placeholder.write(f"**{name}** - {msg}")
                    # Update elapsed time in status label
                    status_obj.update(label=time_fn())
                return callback

            try:
                df1, df2, error = fetch_and_transform_bookings(
                    account_key=account.key,
                    start_date=fetch_start,
                    end_date=fetch_end,
                    include_canceled=include_canceled,
                    status_callback=make_status_callback(status_placeholder, account.name, status, get_time_label),
                    date_basis=date_basis
                )

                if error:
                    errors[account.key] = error
                    status_placeholder.write(f"**{account.name}** - :material/error: {error}")
                elif df1 is not None and len(df1) > 0:
                    account_data[account.key] = (df1, df2)
                    booking_count = len(df1)
                    visit_count = len(df2) if df2 is not None else 0
                    status_placeholder.write(f"**{account.name}** - :material/check: {booking_count:,} bookings, {visit_count:,} visits")
                else:
                    status_placeholder.write(f"**{account.name}** - :material/check: No bookings found")

            except Exception as e:
                errors[account.key] = f"Error: {e}"
                status_placeholder.write(f"**{account.name}** - :material/error: {e}")

        # Merge data from all accounts
        if account_data:
            st.write("Merging data from all accounts...")
            merged_df1, merged_df2 = merge_multi_account_data(account_data)

            total_bookings = len(merged_df1) if merged_df1 is not None else 0
            elapsed = time.time() - start_time
            elapsed_str = _format_elapsed_time(elapsed)
            status.update(label=f"Complete! {total_bookings:,} bookings loaded in {elapsed_str}", state="complete", expanded=False)
            return merged_df1, merged_df2, errors
        else:
            elapsed = time.time() - start_time
            elapsed_str = _format_elapsed_time(elapsed)
            status.update(label=f"No data loaded ({elapsed_str})", state="error", expanded=False)
            return None, None, errors


def refresh_bookeo_cache():
    """Clear Bookeo data cache to force fresh fetch."""
    from bookeo_transformer import _fetch_historical_bookings, _fetch_recent_bookings

    # Clear both cached functions
    _fetch_historical_bookings.clear()
    _fetch_recent_bookings.clear()

    # Reset session state
    st.session_state.bookeo_loaded = False
    st.session_state.bookeo_last_refresh = None
    st.session_state.loading_status = {}
    st.session_state.df1 = None
    st.session_state.df2 = None


def render_bookeo_settings(page_key: str = "default"):
    """
    Render the Bookeo API Settings component in the main content area.

    Args:
        page_key: Unique key prefix for this page's widgets to avoid duplicate key errors
    """
    from datetime import datetime, timedelta
    from bookeo_config import is_bookeo_configured, get_bookeo_accounts
    import time

    if not is_bookeo_configured():
        return

    # Shared preload cache (same as app.py)
    @st.cache_resource
    def get_preload_cache():
        return {'complete': False, 'loading': False, 'status': ''}

    preload_cache = get_preload_cache()
    is_preloading = preload_cache.get('loading') and not preload_cache.get('complete')

    # If preloading just completed, transfer data to session state
    if preload_cache.get('complete') and 'df1' in preload_cache and not st.session_state.get('bookeo_loaded'):
        st.session_state.df1 = preload_cache['df1']
        st.session_state.df2 = preload_cache['df2']
        st.session_state.bookeo_start_date = preload_cache['start_date']
        st.session_state.bookeo_end_date = preload_cache['end_date']
        st.session_state.bookeo_loaded = True
        st.session_state.bookeo_last_refresh = preload_cache['timestamp']
        st.session_state.drive_loaded = True
        st.session_state.data_source = 'bookeo'
        st.rerun()

    # Initialize bookeo date range in session state (default: last 7 days)
    if 'bookeo_start_date' not in st.session_state:
        st.session_state.bookeo_start_date = datetime.now() - timedelta(days=7)
    if 'bookeo_end_date' not in st.session_state:
        st.session_state.bookeo_end_date = datetime.now()

    # Create expander for settings
    with st.expander("Select a date range", expanded=not st.session_state.get('bookeo_loaded', False)):
        col1, col2 = st.columns(2)
        with col1:
            bookeo_start = st.date_input(
                "From",
                value=st.session_state.bookeo_start_date,
                key=f"{page_key}_bookeo_start_input",
                format="DD/MM/YYYY"
            )
        with col2:
            bookeo_end = st.date_input(
                "To",
                value=st.session_state.bookeo_end_date,
                key=f"{page_key}_bookeo_end_input",
                format="DD/MM/YYYY"
            )

        # Store dates in session state
        st.session_state.bookeo_start_date = datetime.combine(bookeo_start, datetime.min.time())
        st.session_state.bookeo_end_date = datetime.combine(bookeo_end, datetime.max.time())

        include_canceled = st.checkbox("Include canceled bookings", value=True, key=f"{page_key}_bookeo_include_canceled")

        # Default to creation date (when booking was made) - matches Mollie revenue
        date_basis = 'created'

        # Load/Refresh buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            load_bookeo = st.button("Load", key=f"{page_key}_load_bookeo_btn", use_container_width=True)
        with col2:
            refresh_bookeo = st.button("Refresh", key=f"{page_key}_refresh_bookeo_btn", use_container_width=True)
        with col3:
            if st.session_state.get('bookeo_last_refresh'):
                st.caption(f"Last updated: {st.session_state.bookeo_last_refresh.strftime('%H:%M')}")

        # Show loading status (for background preload)
        if is_preloading:
            status_text = preload_cache.get('status', 'Loading booking data...')
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f":material/sync: {status_text}")
            with col2:
                if st.button("Check status", key=f"{page_key}_check_preload_status"):
                    st.rerun()

        # Handle Bookeo loading (user clicked Load or Refresh)
        if load_bookeo or refresh_bookeo:
            if refresh_bookeo:
                refresh_bookeo_cache()

            # Load with detailed status UI
            df1, df2, errors = load_bookeo_data_with_status(
                start_date=st.session_state.bookeo_start_date,
                end_date=st.session_state.bookeo_end_date,
                include_canceled=include_canceled,
                date_basis=date_basis
            )

            if errors:
                for account_key, error_msg in errors.items():
                    if account_key != 'config':  # Don't show config errors twice
                        st.warning(f"{account_key}: {error_msg}")

            if df1 is not None and len(df1) > 0:
                st.session_state.df1 = df1
                st.session_state.df2 = df2
                st.session_state.bookeo_loaded = True
                st.session_state.bookeo_last_refresh = datetime.now()
                st.session_state.drive_loaded = True
                st.session_state.data_source = 'bookeo'
                st.rerun()  # Refresh to update UI with new data and timestamp
            else:
                if not errors:
                    st.warning("No bookings found in selected date range")


def get_data_hash(df1, df2):
    """Generate a hash to detect if data has changed."""
    if df1 is None and df2 is None:
        return None

    hash_parts = []
    if df1 is not None:
        hash_parts.append(str(len(df1)))
        hash_parts.append(str(df1.columns.tolist()))
    if df2 is not None:
        hash_parts.append(str(len(df2)))
        hash_parts.append(str(df2.columns.tolist()))

    return hash(tuple(hash_parts))


@st.cache_data
def load_excel_file(uploaded_file):
    """Load a single Excel file with caching."""
    try:
        file_name = uploaded_file.name
        if file_name.endswith('.xls'):
            engine = 'xlrd'
        else:
            engine = 'openpyxl'
        df = pd.read_excel(uploaded_file, engine=engine)
        return df, None
    except Exception as e:
        return None, str(e)


def load_and_merge_files(uploaded_files):
    """Load multiple Excel files and merge them into one dataframe."""
    if not uploaded_files:
        return None, None, []

    dfs = []
    file_info = []
    errors = []

    for uploaded_file in uploaded_files:
        df, error = load_excel_file(uploaded_file)
        if error:
            errors.append(f"{uploaded_file.name}: {error}")
        elif df is not None:
            dfs.append(df)
            file_info.append(f"{uploaded_file.name} ({len(df):,} rows)")

    if errors:
        return None, errors, []

    if not dfs:
        return None, None, []

    merged_df = pd.concat(dfs, ignore_index=True)

    # Merge Activity and Tour columns into unified Location column
    if 'Activity' in merged_df.columns and 'Tour' in merged_df.columns:
        merged_df['Location'] = merged_df['Activity'].fillna(merged_df['Tour'])
    elif 'Activity' in merged_df.columns:
        merged_df['Location'] = merged_df['Activity']
    elif 'Tour' in merged_df.columns:
        merged_df['Location'] = merged_df['Tour']

    return merged_df, None, file_info


@st.cache_data
def process_booking_data(df1, df2, id_col_1, date_col_1, id_col_2, visit_col_2, location_col):
    """
    Process and merge booking data with caching.
    This is the main data processing function used across pages.
    """
    # Prepare df1 (booking creation dates)
    df1_prep = df1[[id_col_1, date_col_1]].copy()
    df1_prep.columns = ['booking_id', 'booking_date']

    # Add location if available
    if location_col != "None" and location_col in df1.columns:
        df1_prep['location'] = df1[location_col]

    # Prepare df2 (visit dates)
    df2_prep = df2[[id_col_2, visit_col_2]].copy()
    df2_prep.columns = ['booking_id', 'visit_date']

    # Merge on booking ID
    merged = df1_prep.merge(df2_prep, on='booking_id', how='inner')

    # Convert dates
    merged['booking_date'] = pd.to_datetime(merged['booking_date'], errors='coerce')
    merged['visit_date'] = pd.to_datetime(merged['visit_date'], errors='coerce')

    # Remove invalid dates
    invalid_mask = merged['booking_date'].isna() | merged['visit_date'].isna()
    merged = merged[~invalid_mask].copy()

    # Calculate interval
    merged['interval_days'] = (merged['visit_date'] - merged['booking_date']).dt.days

    # Remove negative intervals
    merged = merged[merged['interval_days'] >= 0].copy()

    return merged


@st.cache_data
def prepare_chart_data(df, group_col, value_col, agg_func='sum'):
    """
    Prepare aggregated data for charts with caching.
    Reduces repeated groupby operations.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    result = df.groupby(group_col).agg({value_col: agg_func}).reset_index()
    return result


@st.cache_data
def calculate_distribution_data(interval_categories_tuple):
    """
    Calculate lead time distribution data with caching.
    Takes a tuple of interval categories for hashability.
    Returns category counts and percentages.
    """
    category_order = ["Same day", "1-3 days", "4-7 days", "1-2 weeks", "2+ weeks"]
    series = pd.Series(interval_categories_tuple)
    distribution = series.value_counts()
    distribution = distribution.reindex(category_order, fill_value=0)
    total = distribution.sum()
    distribution_pct = (distribution / total * 100).round(1) if total > 0 else distribution
    return distribution, distribution_pct


@st.cache_data
def calculate_location_stats(df_values, location_col_values, interval_col_values):
    """
    Calculate location-wise statistics with caching.
    Takes column values as tuples for hashability.
    """
    import pandas as pd
    df = pd.DataFrame({
        'location': location_col_values,
        'interval_days': interval_col_values
    })

    stats = df.groupby('location').agg({
        'interval_days': ['count', 'mean', 'median']
    }).round(1)
    stats.columns = ['Total Bookings', 'Avg Lead Time (days)', 'Median Lead Time (days)']
    stats = stats.sort_values('Total Bookings', ascending=False)
    stats['Total Bookings'] = stats['Total Bookings'].astype(int)
    return stats


@st.cache_data
def calculate_heatmap_data(booking_hours, booking_dows):
    """
    Calculate heatmap data for booking time analysis with caching.
    """
    import pandas as pd
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    df = pd.DataFrame({'booking_hour': list(booking_hours), 'booking_dow': list(booking_dows)})
    df['booking_dow'] = pd.Categorical(df['booking_dow'], categories=day_order, ordered=True)

    heatmap_pivot = df.groupby(['booking_hour', 'booking_dow']).size().unstack(fill_value=0)
    heatmap_pivot = heatmap_pivot.reindex(columns=day_order, fill_value=0)

    peak_hour = int(df['booking_hour'].mode().iloc[0]) if len(df) > 0 else 0
    peak_day = str(df['booking_dow'].mode().iloc[0]) if len(df) > 0 else 'Unknown'
    evening_pct = (df['booking_hour'] >= 18).sum() / len(df) * 100 if len(df) > 0 else 0

    return heatmap_pivot, peak_hour, peak_day, evening_pct


def get_location_column(df):
    """Get the appropriate location column from a dataframe."""
    if df is None:
        return None

    if 'Location' in df.columns:
        return 'Location'
    elif 'Activity' in df.columns:
        return 'Activity'
    elif 'Tour' in df.columns:
        return 'Tour'
    return None


def get_available_locations(df, location_col=None):
    """Get sorted list of available locations from dataframe."""
    if df is None:
        return []

    if location_col is None:
        location_col = get_location_column(df)

    if location_col is None or location_col not in df.columns:
        return []

    # Filter out NaN values, non-location entries (UTM test, etc.), and sort
    locations = df[location_col].dropna().unique().tolist()
    return sorted([loc for loc in locations if pd.notna(loc) and str(loc).lower().startswith('kuuma')])
