import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import numpy as np
from io import BytesIO, StringIO
import time

# Google Drive imports
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Kuuma Booking Analyzer",
    page_icon="🔥",
    layout="wide"
)

# Hide default Streamlit navigation
hide_default_nav = """
<style>
[data-testid="stSidebarNav"] {
    display: none;
}
</style>
"""
st.markdown(hide_default_nav, unsafe_allow_html=True)

# Header with logo
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://kuuma.nl/wp-content/themes/kuuma/images/logo.svg", width=120)
with col2:
    st.title("Kuuma Booking Analyzer")
    st.markdown("**Customer insights & booking intelligence**")

# Reserve container for date range selector (filled after data loads)
date_range_container = st.container()

# Initialize session state
if 'df1' not in st.session_state:
    st.session_state.df1 = None
if 'df2' not in st.session_state:
    st.session_state.df2 = None
if 'google_ads_df' not in st.session_state:
    st.session_state.google_ads_df = None
if 'meta_ads_df' not in st.session_state:
    st.session_state.meta_ads_df = None
if 'drive_loaded' not in st.session_state:
    st.session_state.drive_loaded = False

# Google Drive functions
@st.cache_resource
def get_drive_service():
    """Connect to Google Drive using service account credentials."""
    if not GOOGLE_DRIVE_AVAILABLE:
        return None

    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        service = build('drive', 'v3', credentials=credentials)
        return service
    except Exception as e:
        st.error(f"Failed to connect to Google Drive: {e}")
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def list_drive_files(folder_id):
    """List all Excel and CSV files in a Google Drive folder."""
    service = get_drive_service()
    if not service:
        return []

    try:
        results = service.files().list(
            q=f"'{folder_id}' in parents and (mimeType='application/vnd.ms-excel' or mimeType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' or mimeType='text/csv')",
            fields="files(id, name, mimeType)"
        ).execute()
        return results.get('files', [])
    except Exception as e:
        st.error(f"Failed to list Drive files: {e}")
        return []

@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def download_drive_file_bytes(file_id, file_name):
    """Download a file from Google Drive and return as bytes (cached)."""
    service = get_drive_service()
    if not service:
        return None

    try:
        request = service.files().get_media(fileId=file_id)
        file_buffer = BytesIO()
        downloader = MediaIoBaseDownload(file_buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return file_buffer.getvalue()  # Return bytes for caching
    except Exception as e:
        return None

def download_drive_file(file_id, file_name):
    """Download a file from Google Drive and return as BytesIO."""
    file_bytes = download_drive_file_bytes(file_id, file_name)
    if file_bytes is None:
        return None
    file_buffer = BytesIO(file_bytes)
    file_buffer.name = file_name
    return file_buffer

def load_files_from_drive():
    """Load booking, visit date, and marketing files from Google Drive."""
    if "google_drive" not in st.secrets:
        return None, None, None, None, "No Google Drive folder configured"

    folder_id = st.secrets["google_drive"]["folder_id"]
    files = list_drive_files(folder_id)

    if not files:
        return None, None, None, None, "No files found in Drive folder"

    # Separate files by type
    booking_files = []
    visit_files = []
    google_ads_files = []
    meta_ads_files = []

    for f in files:
        name_lower = f['name'].lower()

        # CSV files for marketing
        if name_lower.endswith('.csv'):
            if 'google' in name_lower:
                google_ads_files.append(f)
            elif 'meta' in name_lower:
                meta_ads_files.append(f)
            continue

        # Excel files - Detect file type by name - matches Bookeo export naming
        if 'date booked' in name_lower or 'dated booked' in name_lower:
            # "the date booked" = when customer visited
            visit_files.append(f)
        elif 'day the booking was made' in name_lower or 'booking was made' in name_lower:
            # "the day the booking was made" = when booking was created
            booking_files.append(f)
        elif 'visit' in name_lower or 'start' in name_lower or 'datum' in name_lower:
            visit_files.append(f)
        elif 'booking' in name_lower or 'created' in name_lower or 'boeking' in name_lower:
            booking_files.append(f)

    # Load and merge booking files
    df1 = None
    if booking_files:
        dfs = []
        for f in booking_files:
            file_buffer = download_drive_file(f['id'], f['name'])
            if file_buffer:
                try:
                    engine = 'xlrd' if f['name'].endswith('.xls') else 'openpyxl'
                    df = pd.read_excel(file_buffer, engine=engine)
                    dfs.append(df)
                except Exception as e:
                    st.warning(f"Could not read {f['name']}: {e}")
        if dfs:
            df1 = pd.concat(dfs, ignore_index=True)
            # Merge Activity and Tour columns into unified Location column
            if 'Activity' in df1.columns and 'Tour' in df1.columns:
                df1['Location'] = df1['Activity'].fillna(df1['Tour'])
            elif 'Activity' in df1.columns:
                df1['Location'] = df1['Activity']
            elif 'Tour' in df1.columns:
                df1['Location'] = df1['Tour']

    # Load and merge visit files
    df2 = None
    if visit_files:
        dfs = []
        for f in visit_files:
            file_buffer = download_drive_file(f['id'], f['name'])
            if file_buffer:
                try:
                    engine = 'xlrd' if f['name'].endswith('.xls') else 'openpyxl'
                    df = pd.read_excel(file_buffer, engine=engine)
                    dfs.append(df)
                except Exception as e:
                    st.warning(f"Could not read {f['name']}: {e}")
        if dfs:
            df2 = pd.concat(dfs, ignore_index=True)
            # Merge Activity and Tour columns into unified Location column
            if 'Activity' in df2.columns and 'Tour' in df2.columns:
                df2['Location'] = df2['Activity'].fillna(df2['Tour'])
            elif 'Activity' in df2.columns:
                df2['Location'] = df2['Activity']
            elif 'Tour' in df2.columns:
                df2['Location'] = df2['Tour']

    # Load Google Ads CSV files
    google_ads_df = None
    if google_ads_files:
        dfs = []
        for f in google_ads_files:
            file_buffer = download_drive_file(f['id'], f['name'])
            if file_buffer:
                try:
                    # Google Ads CSV has 2-line header, skip them
                    content = file_buffer.getvalue().decode('utf-8')
                    lines = content.split('\n')
                    csv_content = '\n'.join(lines[2:])
                    df = pd.read_csv(StringIO(csv_content))
                    df['Platform'] = 'Google Ads'
                    # Apply column mapping
                    column_mapping = {
                        'Campaign': 'campaign_name',
                        'Cost': 'spend',
                        'Conversions': 'conversions',
                        'Conv. value': 'conversion_value',
                        'Impr.': 'impressions',
                        'Clicks': 'clicks',
                        'CTR': 'ctr',
                        'Avg. CPC': 'cpc'
                    }
                    df = df.rename(columns=column_mapping)
                    # Convert numeric columns
                    numeric_cols = ['spend', 'conversions', 'conversion_value', 'impressions', 'clicks']
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    dfs.append(df)
                except Exception as e:
                    st.warning(f"Could not read {f['name']}: {e}")
        if dfs:
            google_ads_df = pd.concat(dfs, ignore_index=True)

    # Load Meta Ads CSV files
    meta_ads_df = None
    if meta_ads_files:
        dfs = []
        for f in meta_ads_files:
            file_buffer = download_drive_file(f['id'], f['name'])
            if file_buffer:
                try:
                    df = pd.read_csv(file_buffer)
                    df['Platform'] = 'Meta Ads'
                    # Apply column mapping
                    column_mapping = {
                        'Campaign name': 'campaign_name',
                        'Amount spent (EUR)': 'spend',
                        'Purchases': 'conversions',
                        'Purchases conversion value': 'conversion_value',
                        'Reach': 'reach',
                        'Link clicks': 'clicks',
                        'CTR (link click-through rate)': 'ctr',
                        'CPC (cost per link click) (EUR)': 'cpc',
                        'CPM (cost per 1,000 impressions) (EUR)': 'cpm',
                        'Results': 'results'
                    }
                    df = df.rename(columns=column_mapping)
                    # Convert numeric columns
                    numeric_cols = ['spend', 'conversions', 'conversion_value', 'reach', 'clicks', 'results']
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    dfs.append(df)
                except Exception as e:
                    st.warning(f"Could not read {f['name']}: {e}")
        if dfs:
            meta_ads_df = pd.concat(dfs, ignore_index=True)

    return df1, df2, google_ads_df, meta_ads_df, None

# Parse uploaded files
@st.cache_data
def load_excel_file(uploaded_file):
    try:
        # Auto-detect file type and use appropriate engine
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

    # Merge all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)

    # Merge Activity and Tour columns into unified Location column
    if 'Activity' in merged_df.columns and 'Tour' in merged_df.columns:
        merged_df['Location'] = merged_df['Activity'].fillna(merged_df['Tour'])
    elif 'Activity' in merged_df.columns:
        merged_df['Location'] = merged_df['Activity']
    elif 'Tour' in merged_df.columns:
        merged_df['Location'] = merged_df['Tour']

    return merged_df, None, file_info

# Reserve container for navigation at top of sidebar
nav_container = st.sidebar.container()

# Sidebar - Data section
st.sidebar.header("Data")

# Try to load from Google Drive automatically
if GOOGLE_DRIVE_AVAILABLE and "gcp_service_account" in st.secrets and "google_drive" in st.secrets:
    if not st.session_state.drive_loaded:
        progress_bar = st.sidebar.progress(0, text="Connecting to Google Drive...")

        # Animate progress while loading
        progress_bar.progress(10, text="Connecting to Google Drive...")
        df1, df2, google_ads_df, meta_ads_df, error = load_files_from_drive()

        if error:
            progress_bar.empty()
            st.sidebar.warning(f"Drive: {error}")
        else:
            progress_bar.progress(50, text="Processing booking data...")
            time.sleep(0.2)
            if df1 is not None:
                st.session_state.df1 = df1

            progress_bar.progress(70, text="Processing visit data...")
            time.sleep(0.2)
            if df2 is not None:
                st.session_state.df2 = df2

            progress_bar.progress(85, text="Processing marketing data...")
            time.sleep(0.2)
            if google_ads_df is not None:
                st.session_state.google_ads_df = google_ads_df
            if meta_ads_df is not None:
                st.session_state.meta_ads_df = meta_ads_df

            progress_bar.progress(100, text="Complete!")
            time.sleep(0.3)
            progress_bar.empty()
            st.session_state.drive_loaded = True

    # Show loaded status
    if st.session_state.df1 is not None:
        st.sidebar.success(f"Booking data: {len(st.session_state.df1):,} rows")
    if st.session_state.df2 is not None:
        st.sidebar.success(f"Visit data: {len(st.session_state.df2):,} rows")
    if st.session_state.google_ads_df is not None:
        st.sidebar.success(f"Google Ads: {len(st.session_state.google_ads_df):,} rows")
    if st.session_state.meta_ads_df is not None:
        st.sidebar.success(f"Meta Ads: {len(st.session_state.meta_ads_df):,} rows")

    # Manual upload option (in expander)
    with st.sidebar.expander("Upload different files"):
        uploaded_files1 = st.file_uploader(
            "Booking Creation Dates (.xls/.xlsx)",
            type=["xls", "xlsx"],
            help="Upload files to replace Google Drive data",
            accept_multiple_files=True,
            key="manual_upload_1"
        )

        uploaded_files2 = st.file_uploader(
            "Visit Dates (.xls/.xlsx)",
            type=["xls", "xlsx"],
            help="Upload files to replace Google Drive data",
            accept_multiple_files=True,
            key="manual_upload_2"
        )

        if uploaded_files1:
            df1, errors1, file_info1 = load_and_merge_files(uploaded_files1)
            if errors1:
                for error in errors1:
                    st.error(f"Error: {error}")
            elif df1 is not None:
                st.session_state.df1 = df1
                st.success(f"Loaded: {len(df1):,} rows")

        if uploaded_files2:
            df2, errors2, file_info2 = load_and_merge_files(uploaded_files2)
            if errors2:
                for error in errors2:
                    st.error(f"Error: {error}")
            elif df2 is not None:
                st.session_state.df2 = df2
                st.success(f"Loaded: {len(df2):,} rows")

else:
    # No Google Drive configured - show standard upload
    st.sidebar.info("Upload your Bookeo export files")

    uploaded_files1 = st.sidebar.file_uploader(
        "Booking Creation Dates (.xls/.xlsx)",
        type=["xls", "xlsx"],
        help="Upload files containing when bookings were created. You can select multiple files from different Bookeo instances.",
        accept_multiple_files=True
    )

    uploaded_files2 = st.sidebar.file_uploader(
        "Visit Dates (.xls/.xlsx)",
        type=["xls", "xlsx"],
        help="Upload files containing when customers actually visited. You can select multiple files from different Bookeo instances.",
        accept_multiple_files=True
    )

    # Load and merge File 1 (Booking Creation Dates)
    if uploaded_files1:
        df1, errors1, file_info1 = load_and_merge_files(uploaded_files1)
        if errors1:
            for error in errors1:
                st.sidebar.error(f"Error: {error}")
        elif df1 is not None:
            st.session_state.df1 = df1
            if len(file_info1) == 1:
                st.sidebar.success(f"Loaded: {len(df1):,} rows")
            else:
                st.sidebar.success(f"Merged {len(file_info1)} files: {len(df1):,} total rows")
                with st.sidebar.expander("File details"):
                    for info in file_info1:
                        st.write(f"- {info}")

    # Load and merge File 2 (Visit Dates)
    if uploaded_files2:
        df2, errors2, file_info2 = load_and_merge_files(uploaded_files2)
        if errors2:
            for error in errors2:
                st.sidebar.error(f"Error: {error}")
        elif df2 is not None:
            st.session_state.df2 = df2
            if len(file_info2) == 1:
                st.sidebar.success(f"Loaded: {len(df2):,} rows")
            else:
                st.sidebar.success(f"Merged {len(file_info2)} files: {len(df2):,} total rows")
                with st.sidebar.expander("File details"):
                    for info in file_info2:
                        st.write(f"- {info}")

# Fill navigation container (now that files are loaded)
if st.session_state.df1 is not None and st.session_state.df2 is not None:
    with nav_container:
        st.markdown("### Navigation")
        st.page_link("app.py", label="Booking Patterns", icon=":material/bar_chart:")
        st.page_link("pages/3_Customers.py", label="Recurring Customers", icon=":material/group:")
        st.page_link("pages/4_Revenue.py", label="Revenue & Value", icon=":material/payments:")
        st.page_link("pages/5_Promotions.py", label="Promotions", icon=":material/sell:")
        st.page_link("pages/6_Capacity.py", label="Capacity Analysis", icon=":material/analytics:")
        st.page_link("pages/7_Marketing.py", label="Marketing", icon=":material/campaign:")
        st.markdown("---")

# Column mapping section
if st.session_state.df1 is not None and st.session_state.df2 is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Column Mapping")

    df1 = st.session_state.df1
    df2 = st.session_state.df2

    # Smart defaults based on actual data structure
    default_id_col = "Booking number" if "Booking number" in df1.columns else df1.columns[0]
    default_created_col = "Created" if "Created" in df1.columns else df1.columns[0]
    default_start_col = "Start" if "Start" in df2.columns else df2.columns[0]
    # Check for Location (unified), Tour or Activity as location column
    if "Location" in df1.columns:
        default_location_col = "Location"
    elif "Tour" in df1.columns:
        default_location_col = "Tour"
    elif "Activity" in df1.columns:
        default_location_col = "Activity"
    else:
        default_location_col = None

    # Booking ID columns
    id_col_1 = st.sidebar.selectbox(
        "Booking ID (File 1)",
        options=df1.columns.tolist(),
        index=df1.columns.tolist().index(default_id_col) if default_id_col in df1.columns else 0,
        help="Column that uniquely identifies each booking"
    )

    # Date columns
    date_col_1 = st.sidebar.selectbox(
        "Booking Creation Date (File 1)",
        options=df1.columns.tolist(),
        index=df1.columns.tolist().index(default_created_col) if default_created_col in df1.columns else 0,
        help="When the customer made the booking"
    )

    id_col_2 = st.sidebar.selectbox(
        "Booking ID (File 2)",
        options=df2.columns.tolist(),
        index=df2.columns.tolist().index(default_id_col) if default_id_col in df2.columns else 0,
        help="Column that uniquely identifies each booking"
    )

    visit_col_2 = st.sidebar.selectbox(
        "Visit Date (File 2)",
        options=df2.columns.tolist(),
        index=df2.columns.tolist().index(default_start_col) if default_start_col in df2.columns else 0,
        help="When the customer actually visited"
    )

    # Location column (optional)
    location_options = ["None"] + df1.columns.tolist()
    location_default_index = location_options.index(default_location_col) if default_location_col in location_options else 0

    location_col = st.sidebar.selectbox(
        "Location (optional)",
        options=location_options,
        index=location_default_index,
        help="Column containing location/branch names for comparison"
    )

    # Email column (for recurring customer analysis)
    default_email_col = "Email address" if "Email address" in df1.columns else None
    email_options = ["None"] + df1.columns.tolist()
    email_default_index = email_options.index(default_email_col) if default_email_col in email_options else 0

    email_col = st.sidebar.selectbox(
        "Email Address (optional)",
        options=email_options,
        index=email_default_index,
        help="Column containing customer email for recurring customer analysis"
    )

    # Data processing
    @st.cache_data
    def process_data(df1, df2, id_col_1, date_col_1, id_col_2, visit_col_2, location_col):
        # Prepare dataframes with renamed columns
        df1_prep = df1[[id_col_1, date_col_1]].copy()
        df1_prep.columns = ['booking_id', 'booking_date']

        df2_prep = df2[[id_col_2, visit_col_2]].copy()
        df2_prep.columns = ['booking_id', 'visit_date']

        # Add location if selected
        if location_col != "None":
            df1_prep['location'] = df1[location_col]

        # Merge on booking ID
        merged = df1_prep.merge(df2_prep, on='booking_id', how='inner')

        # Count unmatched records
        unmatched_count = len(df1) + len(df2) - (2 * len(merged))

        # Convert dates to datetime
        merged['booking_date'] = pd.to_datetime(merged['booking_date'], errors='coerce')
        merged['visit_date'] = pd.to_datetime(merged['visit_date'], errors='coerce')

        # Calculate interval
        merged['interval_days'] = (merged['visit_date'] - merged['booking_date']).dt.days

        # Filter invalid records
        invalid_dates = merged['booking_date'].isna() | merged['visit_date'].isna()
        negative_intervals = merged['interval_days'] < 0

        invalid_count = invalid_dates.sum()
        negative_count = negative_intervals.sum()

        # Keep only valid records
        merged_clean = merged[~invalid_dates & ~negative_intervals].copy()

        # Categorize intervals
        def categorize_interval(days):
            if days == 0:
                return "Same day"
            elif 1 <= days <= 3:
                return "1-3 days"
            elif 4 <= days <= 7:
                return "4-7 days"
            elif 8 <= days <= 14:
                return "1-2 weeks"
            else:
                return "2+ weeks"

        merged_clean['interval_category'] = merged_clean['interval_days'].apply(categorize_interval)

        return merged_clean, unmatched_count, invalid_count, negative_count

    # Temperature analysis functions
    @st.cache_data
    def get_temperature_data(start_date, end_date, latitude=52.37, longitude=4.89):
        """
        Fetch daily average temperature data with fallback strategy
        Primary: Open-Meteo API
        Fallback: Meteostat library (no API key needed)
        Default coordinates: Amsterdam center (can be adjusted per location)
        """
        # Try Open-Meteo first
        try:
            api_url = (
                f"https://archive-api.open-meteo.com/v1/archive"
                f"?latitude={latitude}&longitude={longitude}"
                f"&start_date={start_date.strftime('%Y-%m-%d')}"
                f"&end_date={end_date.strftime('%Y-%m-%d')}"
                f"&daily=temperature_2m_mean"
                f"&timezone=Europe%2FAmsterdam"
            )
            resp = requests.get(api_url, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if 'daily' in data:
                temp_dates = pd.to_datetime(data['daily']['time'])
                temp_values = data['daily']['temperature_2m_mean']
                temp_df = pd.DataFrame({'date': temp_dates, 'temperature': temp_values})
                return temp_df
        except Exception:
            # Silently try fallback
            pass

        # Fallback to Meteostat
        try:
            from meteostat import Point, Daily

            location = Point(latitude, longitude)
            data = Daily(location, start_date, end_date)
            meteo_df = data.fetch()

            if meteo_df is not None and not meteo_df.empty and 'tavg' in meteo_df.columns:
                # Reset index to get dates as a column
                meteo_df = meteo_df.reset_index()
                # The index becomes 'time' column after reset
                temp_df = pd.DataFrame({
                    'date': pd.to_datetime(meteo_df['time']),
                    'temperature': meteo_df['tavg']
                })
                # Remove NaN temperatures
                temp_df = temp_df.dropna(subset=['temperature'])
                return temp_df
        except Exception:
            # Silently fail
            return None

        return None

    def add_temperature_to_bookings(bookings_df):
        """Add daily average temperature data to bookings based on booking date"""
        if 'booking_date' not in bookings_df.columns:
            return bookings_df

        # Get date range with buffer
        start_date = bookings_df['booking_date'].min() - pd.Timedelta(days=1)
        end_date = bookings_df['booking_date'].max() + pd.Timedelta(days=1)

        # Cap end_date to today (API doesn't support future dates)
        today = pd.Timestamp.now().normalize()
        if end_date > today:
            end_date = today

        # Fetch temperature data
        temp_df = get_temperature_data(start_date, end_date)

        if temp_df is None:
            return bookings_df

        # Merge by date (not hour)
        bookings_with_temp = bookings_df.copy()
        bookings_with_temp['booking_date_only'] = bookings_with_temp['booking_date'].dt.date
        temp_df['date_only'] = temp_df['date'].dt.date

        merged = bookings_with_temp.merge(
            temp_df,
            left_on='booking_date_only',
            right_on='date_only',
            how='left'
        )

        # Categorize temperature
        merged['temp_category'] = pd.cut(
            merged['temperature'],
            bins=[-np.inf, 5, 10, 13, 16, 20, np.inf],
            labels=['Below 5°C', '5-10°C', '10-13°C', '13-16°C', '16-20°C', 'Above 20°C']
        )

        return merged

    # Process the data
    with st.spinner("Processing data..."):
        processed_data, unmatched, invalid, negative = process_data(
            df1, df2, id_col_1, date_col_1, id_col_2, visit_col_2, location_col
        )

    # Check if we have data to display
    if len(processed_data) == 0:
        st.error("No matching booking IDs found between files. Please check your column selections.")
    else:
        # Date range selector in reserved container (under header)
        min_date = processed_data['visit_date'].min().date()
        max_date = processed_data['visit_date'].max().date()

        with date_range_container:
            date_col1, date_col2 = st.columns([2, 4])
            with date_col1:
                date_range = st.date_input(
                    "Date Range (Visit Date)",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    help="Filter bookings by visit date",
                    key="main_date_range"
                )

        # Location filter in sidebar
        st.sidebar.markdown("---")
        st.sidebar.subheader("Filters")

        # Location filter (if location column selected)
        if location_col != "None" and 'location' in processed_data.columns:
            # Filter out NaN values and sort
            all_locations = sorted([loc for loc in processed_data['location'].unique() if pd.notna(loc)])
            selected_locations = st.sidebar.multiselect(
                "Locations",
                options=all_locations,
                default=all_locations,
                help="Select which locations to include"
            )
        else:
            selected_locations = None

        # Temperature analysis toggle
        st.sidebar.markdown("---")
        show_temperature = st.sidebar.checkbox(
            "Include Temperature Analysis",
            value=True,
            help="Analyze how weather affects booking behavior (fetches data from Open-Meteo API)"
        )

        # Apply filters
        filtered_data = processed_data.copy()

        # Date filter
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_data = filtered_data[
                (filtered_data['visit_date'].dt.date >= start_date) &
                (filtered_data['visit_date'].dt.date <= end_date)
            ]

        # Location filter
        if selected_locations and location_col != "None":
            filtered_data = filtered_data[filtered_data['location'].isin(selected_locations)]

        # Calculate metrics
        if len(filtered_data) > 0:
            avg_interval = filtered_data['interval_days'].mean()
            median_interval = filtered_data['interval_days'].median()
            total_bookings = len(filtered_data)
            same_day_pct = (filtered_data['interval_days'] == 0).sum() / total_bookings * 100

            # Display metrics
            st.markdown("### Key Metrics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Average Lead Time",
                    f"{avg_interval:.1f} days",
                    help="Mean number of days between booking creation and visit date."
                )

            with col2:
                st.metric(
                    "Median Lead Time",
                    f"{median_interval:.1f} days",
                    help="Middle value of lead times. Less affected by outliers than average."
                )

            with col3:
                st.metric(
                    "Total Bookings",
                    f"{total_bookings:,}",
                    help="Total number of successfully matched bookings in the dataset."
                )

            with col4:
                st.metric(
                    "Same-Day Bookings",
                    f"{same_day_pct:.1f}%",
                    help="Percentage of bookings made on the same day as the visit."
                )

            # Data quality summary
            st.info(f"**Data Quality:** {len(processed_data):,} matched records | "
                   f"{unmatched:,} unmatched | {invalid:,} invalid dates | {negative:,} negative intervals")

            # Distribution chart
            st.markdown("### Booking Lead Time Distribution")

            category_order = ["Same day", "1-3 days", "4-7 days", "1-2 weeks", "2+ weeks"]
            distribution = filtered_data['interval_category'].value_counts()
            distribution = distribution.reindex(category_order, fill_value=0)

            # Calculate percentages
            total_bookings = distribution.sum()
            distribution_pct = (distribution / total_bookings * 100).round(1)

            # Format text labels with percentage
            text_labels = [f"{pct}%" for pct in distribution_pct.values]

            fig_dist = px.bar(
                x=distribution_pct.index,
                y=distribution_pct.values,
                labels={'x': 'Lead Time', 'y': 'Percentage of Total Bookings'},
                title="How far in advance do customers book?"
            )
            fig_dist.update_traces(marker_color='#1f77b4', text=text_labels, textposition='outside')
            fig_dist.update_layout(
                showlegend=False,
                height=400,
                yaxis=dict(range=[0, 100])  # Always show 0-100% for percentage data
            )

            st.plotly_chart(fig_dist, use_container_width=True)

            # Explanation box
            with st.expander("What does 'Lead Time' mean?", expanded=False):
                st.markdown("""
                **Lead time** is the time gap between when a customer makes a booking and when they actually visit.

                **Examples:**
                - **0 days (Same-day)**: Customer books today for today's visit
                - **1 day**: Customer books on Monday for Tuesday
                - **7 days**: Customer books a week in advance
                - **14+ days**: Customer books 2+ weeks ahead

                **Why it matters:**
                - **Short lead time (0-3 days)**: Spontaneous bookings, requires flexible staffing
                - **Long lead time (7+ days)**: Planned visits, allows advance scheduling optimization

                Different locations may show different booking patterns, helping you understand customer behavior per branch.
                """)

            # Location breakdown
            if location_col != "None" and len(selected_locations) > 0:
                st.markdown("### Breakdown by Location")

                st.markdown("Compare booking behavior across different locations.")

                with st.expander("Understanding Average vs Median"):
                    st.markdown("""
                    - **Average** - Includes all bookings, including customers who book far in advance (e.g., 30 days ahead)
                    - **Median** - Shows what a typical customer actually does, ignoring extreme values

                    **Example:** If most customers book 0-2 days ahead, but a few book 30 days ahead:
                    - Average might be 5 days (pulled up by advance planners)
                    - Median would be 1 day (the typical customer)

                    **What to look for:** A large gap between Average and Median means you have mostly last-minute bookers
                    with some advance planners. A small gap means consistent booking behavior.
                    """)

                location_stats = filtered_data.groupby('location').agg({
                    'booking_id': 'count',
                    'interval_days': ['mean', 'median']
                }).round(1)

                location_stats.columns = ['Total Bookings', 'Avg Lead Time (days)', 'Median Lead Time (days)']
                location_stats = location_stats.sort_values('Total Bookings', ascending=False)
                location_stats['Total Bookings'] = location_stats['Total Bookings'].astype(int)

                location_stats_config = {
                    'Total Bookings': st.column_config.NumberColumn('Total Bookings', help='Number of bookings for this location'),
                    'Avg Lead Time (days)': st.column_config.NumberColumn('Avg Lead Time (days)', help='Average days between booking and visit'),
                    'Median Lead Time (days)': st.column_config.NumberColumn('Median Lead Time (days)', help='Typical days between booking and visit (ignoring outliers)'),
                }
                st.dataframe(location_stats, use_container_width=True, column_config=location_stats_config)

            # Temperature Analysis
            if show_temperature:
                with st.spinner("Fetching temperature data..."):
                    data_with_temp = add_temperature_to_bookings(filtered_data)

                # Only show the section if we successfully got temperature data
                if 'temperature' in data_with_temp.columns and data_with_temp['temperature'].notna().any():
                    st.markdown("---")
                    st.markdown("### Temperature & Booking Behavior Analysis")

                    st.markdown("""
                    This analysis shows how weather conditions at the time of booking affect customer behavior.

                    **Key insights:**
                    - Does cold weather drive more sauna bookings?
                    - Do customers book further in advance during certain temperatures?
                    - Which temperature ranges have the highest booking volume?
                    """)
                    # Temperature vs Lead Time
                    temp_analysis = data_with_temp.groupby('temp_category', observed=True).agg({
                        'interval_days': 'mean',
                        'booking_id': 'count'
                    }).round(1)
                    temp_analysis.columns = ['Avg Lead Time (days)', 'Number of Bookings']
                    temp_analysis = temp_analysis.reset_index()

                    # Combine data for bubble plot
                    # temp_analysis already has temp_category, Avg Lead Time, Number of Bookings
                    # We need to merge with temp_stats to get Avg Temp and % of Total

                    # Calculate temp_stats early to get Avg Temp (°C)
                    temp_stats_early = data_with_temp.groupby('temp_category', observed=True).agg({
                        'temperature': 'mean'
                    }).round(1)
                    temp_stats_early.columns = ['Avg Temp (°C)']

                    # Merge temp_analysis with average temperature
                    bubble_data = temp_analysis.merge(
                        temp_stats_early,
                        left_on='temp_category',
                        right_index=True
                    )

                    # Calculate percentage of total
                    total_bookings = bubble_data['Number of Bookings'].sum()
                    bubble_data['% of Total'] = (bubble_data['Number of Bookings'] / total_bookings * 100).round(1)

                    # Create labels for bubbles
                    bubble_data['label'] = bubble_data.apply(
                        lambda row: f"{row['temp_category']}<br>({row['% of Total']}%)",
                        axis=1
                    )

                    # Create bubble plot
                    fig_temp = px.scatter(
                        bubble_data,
                        x='Avg Temp (°C)',
                        y='Avg Lead Time (days)',
                        size='% of Total',
                        text='label',
                        hover_data={
                            'temp_category': True,
                            'Avg Temp (°C)': ':.1f',
                            'Avg Lead Time (days)': ':.1f',
                            'Number of Bookings': ':,',
                            '% of Total': ':.1f',
                            'label': False
                        },
                        size_max=70,
                        title="Booking Behavior by Temperature"
                    )

                    # Update bubble appearance
                    fig_temp.update_traces(
                        marker=dict(
                            color='#1f77b4',  # Blue color matching app theme
                            opacity=0.7,
                            line=dict(width=1, color='white')
                        ),
                        textposition='top center',
                        textfont=dict(size=10)
                    )

                    # Update layout
                    fig_temp.update_layout(
                        xaxis=dict(
                            title='Average Temperature (°C)',
                            range=[0, bubble_data['Avg Temp (°C)'].max() * 1.1]
                        ),
                        yaxis=dict(
                            title='Average Lead Time (days)',
                            range=[0, bubble_data['Avg Lead Time (days)'].max() * 1.15]
                        ),
                        height=500,
                        showlegend=False,
                        hovermode='closest'
                    )

                    st.plotly_chart(fig_temp, use_container_width=True)

                    with st.expander("How to read the bubble plot"):
                        st.markdown("""
- **Horizontal position** (X-axis): Average temperature when booking was made
- **Vertical position** (Y-axis): How far in advance customers book
- **Bubble size**: Share of total bookings (larger = more bookings)
- **Pattern**: Warmer weather typically correlates with more advance planning

**Interpretation:**
- **Large bubbles high on chart**: Popular temperature range where customers plan ahead
- **Large bubbles low on chart**: Popular temperature range with spontaneous bookings
- **Small bubbles**: Temperature ranges with less sauna demand
                        """)

                    # Temperature statistics table
                    st.markdown("#### Temperature Breakdown")

                    # Add month column to data
                    data_with_temp['month'] = data_with_temp['booking_date'].dt.strftime('%B')

                    # Calculate main statistics
                    temp_stats = data_with_temp.groupby('temp_category', observed=True).agg({
                        'booking_id': 'count',
                        'interval_days': ['mean', 'median'],
                        'temperature': 'mean'
                    }).round(1)

                    temp_stats.columns = ['Bookings', 'Avg Lead Time', 'Median Lead Time', 'Avg Temp (°C)']

                    # Calculate common months separately
                    common_months = []
                    for category in temp_stats.index:
                        category_data = data_with_temp[data_with_temp['temp_category'] == category]
                        month_counts = category_data['month'].value_counts()
                        if len(month_counts) > 0:
                            # Get top 2 months
                            top_months = month_counts.head(2).index.tolist()
                            common_months.append(', '.join(top_months))
                        else:
                            common_months.append('N/A')

                    temp_stats['Common Months'] = common_months
                    temp_stats['% of Total'] = (temp_stats['Bookings'] / temp_stats['Bookings'].sum() * 100).round(1)
                    temp_stats = temp_stats[['Bookings', '% of Total', 'Common Months', 'Avg Temp (°C)', 'Avg Lead Time', 'Median Lead Time']]

                    temp_stats_config = {
                        'Bookings': st.column_config.NumberColumn('Bookings', help='Number of bookings in this temperature range'),
                        '% of Total': st.column_config.NumberColumn('% of Total', help='Percentage of all bookings'),
                        'Common Months': st.column_config.TextColumn('Common Months', help='Months when this temperature range is most common'),
                        'Avg Temp (°C)': st.column_config.NumberColumn('Avg Temp (°C)', help='Average temperature at time of booking'),
                        'Avg Lead Time': st.column_config.NumberColumn('Avg Lead Time', help='Average days between booking and visit'),
                        'Median Lead Time': st.column_config.NumberColumn('Median Lead Time', help='Typical days between booking and visit'),
                    }
                    st.dataframe(temp_stats, use_container_width=True, column_config=temp_stats_config)

            # Export functionality
            st.sidebar.markdown("---")
            st.sidebar.subheader("Export")

            export_data = filtered_data[['booking_id', 'booking_date', 'visit_date', 'interval_days']].copy()
            if location_col != "None":
                export_data['location'] = filtered_data['location']

            export_data.columns = ['Booking ID', 'Booking Date', 'Visit Date', 'Interval (Days)'] + (['Location'] if location_col != "None" else [])

            csv = export_data.to_csv(index=False)

            st.sidebar.download_button(
                label="Download Results (CSV)",
                data=csv,
                file_name=f"booking-analysis-{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

            # Data preview toggle
            show_preview = st.sidebar.checkbox("Show data preview", value=False)

            if show_preview:
                st.markdown("---")
                st.markdown("### Data Preview")

                with st.expander("File 1 Preview (Booking Creation)", expanded=False):
                    st.dataframe(df1.head(5), use_container_width=True)

                with st.expander("File 2 Preview (Visit Dates)", expanded=False):
                    st.dataframe(df2.head(5), use_container_width=True)

                with st.expander("Processed Data Preview", expanded=False):
                    st.dataframe(filtered_data.head(10), use_container_width=True)

        else:
            st.warning("No data matches the selected filters. Try adjusting your date range or location selection.")

    # Reset button
    st.sidebar.markdown("---")
    if st.sidebar.button("Clear All & Start Over"):
        st.session_state.df1 = None
        st.session_state.df2 = None
        st.rerun()

else:
    # Welcome message
    st.info("**Get started:** Upload both Excel files in the sidebar to begin analysis")

    st.markdown("### How it works")
    st.markdown("""
    1. **Upload** your two Excel files:
       - File 1: Contains booking creation dates
       - File 2: Contains visit dates

    2. **Map columns** to tell the app which columns contain:
       - Booking IDs (to match records)
       - Dates (booking creation and visit dates)
       - Locations (optional - supports 'Tour' or 'Activity' columns)
       - Email addresses (optional - for customer loyalty analysis)

    3. **Analyze booking patterns**:
       - **Booking Intervals**: Average and median lead times, same-day booking rates, distribution charts
       - **Location Performance**: Compare booking behavior and capacity across locations
       - **Temperature Impact**: Optional analysis showing how weather affects booking advance planning

    4. **Understand customer loyalty**:
       - **Customer Segmentation**: Five-tier system (One-time, Light, Regular, Frequent, VIP)
       - **Location Loyalty**: See which locations retain customers best
       - **Recurring Customer Metrics**: Track retention rates and multi-location behavior

    5. **Export** your results as CSV for further analysis or reporting
    """)
