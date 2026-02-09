import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import numpy as np
from io import BytesIO, StringIO
import time
import os
import glob
from data_loader import (
    init_session_state, get_location_column, get_available_locations,
    calculate_distribution_data, calculate_location_stats, calculate_heatmap_data,
    apply_demo_transform, DEMO_MODE
)

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
/* Hide "Press Enter to apply" tooltip on text inputs */
[data-testid="InputInstructions"] {
    display: none;
}
/* Invert logo to white when user's browser is in dark mode */
@media (prefers-color-scheme: dark) {
    [data-testid="stImage"] img {
        filter: invert(1);
    }
}
</style>
"""
st.markdown(hide_default_nav, unsafe_allow_html=True)

# Initialize session state using centralized function
init_session_state()

# Initialize authentication state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Google Drive functions (defined early so they can be used on login page)
@st.cache_resource
def get_drive_service():
    """Connect to Google Drive using service account credentials."""
    if not GOOGLE_DRIVE_AVAILABLE:
        return None, "Google Drive libraries not installed"

    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        service = build('drive', 'v3', credentials=credentials)
        return service, None
    except Exception as e:
        return None, f"Drive authentication failed: {e}"

@st.cache_data(ttl=3600)  # Cache for 1 hour (file names rarely change)
def list_drive_files(folder_id):
    """List all Excel and CSV files in a Google Drive folder."""
    service, err = get_drive_service()
    if not service:
        return []

    try:
        results = service.files().list(
            q=f"'{folder_id}' in parents and (mimeType='application/vnd.ms-excel' or mimeType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' or mimeType='text/csv')",
            fields="files(id, name, mimeType)"
        ).execute()
        return results.get('files', [])
    except Exception as e:
        return []

@st.cache_data(ttl=14400, show_spinner=False)  # Cache for 4 hours
def download_drive_file_bytes(file_id, file_name):
    """Download a file from Google Drive and return as bytes (cached)."""
    service, err = get_drive_service()
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

def load_files_from_drive(progress_callback=None):
    """Load booking, visit date, and marketing files from Google Drive.

    Args:
        progress_callback: Optional function(progress: int, text: str) to report loading progress
    """
    def update_progress(progress, text):
        if progress_callback:
            progress_callback(progress, text)

    if "google_drive" not in st.secrets:
        return None, None, None, None, "No Google Drive folder configured"

    update_progress(5, "Connecting to Google Drive...")

    service, auth_error = get_drive_service()
    if not service:
        return None, None, None, None, auth_error or "Could not connect to Google Drive"

    folder_id = st.secrets["google_drive"]["folder_id"]
    files = list_drive_files(folder_id)

    if not files:
        return None, None, None, None, f"No Excel/CSV files found in Drive folder ({folder_id})"

    update_progress(10, "Scanning files...")

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
            visit_files.append(f)
        elif 'day the booking was made' in name_lower or 'booking was made' in name_lower:
            booking_files.append(f)
        elif 'visit' in name_lower or 'start' in name_lower or 'datum' in name_lower:
            visit_files.append(f)
        elif 'booking' in name_lower or 'created' in name_lower or 'boeking' in name_lower:
            booking_files.append(f)

    # Check if any files were classified
    total_files = len(booking_files) + len(visit_files) + len(google_ads_files) + len(meta_ads_files)
    if total_files == 0:
        unclassified = [f['name'] for f in files]
        return None, None, None, None, f"Found {len(files)} file(s) but none matched expected naming patterns: {unclassified}"

    files_loaded = 0
    base_progress = 15  # Start at 15% after scanning
    progress_per_file = 70 / max(total_files, 1)  # Use 15-85% for file loading

    # Load and merge booking files
    df1 = None
    if booking_files:
        dfs = []
        for f in booking_files:
            file_name = f['name'][:40] + '...' if len(f['name']) > 40 else f['name']
            update_progress(int(base_progress + files_loaded * progress_per_file), f"Loading: {file_name}")
            file_buffer = download_drive_file(f['id'], f['name'])
            if file_buffer:
                try:
                    engine = 'xlrd' if f['name'].endswith('.xls') else 'openpyxl'
                    df = pd.read_excel(file_buffer, engine=engine)
                    dfs.append(df)
                except Exception:
                    pass
            files_loaded += 1
        if dfs:
            df1 = pd.concat(dfs, ignore_index=True)
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
            file_name = f['name'][:40] + '...' if len(f['name']) > 40 else f['name']
            update_progress(int(base_progress + files_loaded * progress_per_file), f"Loading: {file_name}")
            file_buffer = download_drive_file(f['id'], f['name'])
            if file_buffer:
                try:
                    engine = 'xlrd' if f['name'].endswith('.xls') else 'openpyxl'
                    df = pd.read_excel(file_buffer, engine=engine)
                    dfs.append(df)
                except Exception:
                    pass
            files_loaded += 1
        if dfs:
            df2 = pd.concat(dfs, ignore_index=True)
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
            file_name = f['name'][:40] + '...' if len(f['name']) > 40 else f['name']
            update_progress(int(base_progress + files_loaded * progress_per_file), f"Loading: {file_name}")
            file_buffer = download_drive_file(f['id'], f['name'])
            if file_buffer:
                try:
                    content = file_buffer.getvalue().decode('utf-8')
                    lines = content.split('\n')
                    csv_content = '\n'.join(lines[2:])
                    df = pd.read_csv(StringIO(csv_content))
                    df['Platform'] = 'Google Ads'
                    column_mapping = {
                        'Campaign': 'campaign_name', 'Cost': 'spend',
                        'Conversions': 'conversions', 'Conv. value': 'conversion_value',
                        'Impr.': 'impressions', 'Impressions': 'impressions',
                        'Clicks': 'clicks', 'CTR': 'ctr', 'Avg. CPC': 'cpc',
                        'CPC': 'cpc', 'Conv. Value': 'conversion_value'
                    }
                    df = df.rename(columns=column_mapping)
                    for col in ['spend', 'conversions', 'conversion_value', 'impressions', 'clicks']:
                        if col in df.columns:
                            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    dfs.append(df)
                except Exception:
                    pass
            files_loaded += 1
        if dfs:
            google_ads_df = pd.concat(dfs, ignore_index=True)

    # Load Meta Ads CSV files
    meta_ads_df = None
    if meta_ads_files:
        dfs = []
        for f in meta_ads_files:
            file_name = f['name'][:40] + '...' if len(f['name']) > 40 else f['name']
            update_progress(int(base_progress + files_loaded * progress_per_file), f"Loading: {file_name}")
            file_buffer = download_drive_file(f['id'], f['name'])
            if file_buffer:
                try:
                    df = pd.read_csv(file_buffer)
                    df['Platform'] = 'Meta Ads'
                    column_mapping = {
                        'Campaign name': 'campaign_name', 'Amount spent (EUR)': 'spend',
                        'Purchases': 'conversions', 'Purchases conversion value': 'conversion_value',
                        'Reach': 'reach', 'Link clicks': 'clicks',
                        'CTR (link click-through rate)': 'ctr',
                        'CPC (cost per link click) (EUR)': 'cpc',
                        'CPM (cost per 1,000 impressions) (EUR)': 'cpm',
                        'Results': 'results'
                    }
                    df = df.rename(columns=column_mapping)
                    for col in ['spend', 'conversions', 'conversion_value', 'reach', 'clicks', 'results']:
                        if col in df.columns:
                            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    dfs.append(df)
                except Exception:
                    pass
            files_loaded += 1
        if dfs:
            meta_ads_df = pd.concat(dfs, ignore_index=True)

    update_progress(90, "Finalizing...")

    return df1, df2, google_ads_df, meta_ads_df, None


def load_local_files():
    """Load files from local 'booking data' and 'marketing data' folders for development/testing."""
    # Use parent directory folders
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    booking_data_path = os.path.join(parent_dir, 'booking data')
    marketing_data_path = os.path.join(parent_dir, 'marketing data')

    # Load booking creation files (files with "day the booking was made" in name)
    df1 = None
    if os.path.exists(booking_data_path):
        all_files = glob.glob(os.path.join(booking_data_path, '*.xls')) + glob.glob(os.path.join(booking_data_path, '*.xlsx'))
        booking_files = [f for f in all_files if 'day the booking was made' in os.path.basename(f).lower() or 'booking was made' in os.path.basename(f).lower()]
        if booking_files:
            dfs = []
            for f in booking_files:
                try:
                    engine = 'xlrd' if f.endswith('.xls') else 'openpyxl'
                    df = pd.read_excel(f, engine=engine)
                    dfs.append(df)
                except Exception:
                    pass
            if dfs:
                df1 = pd.concat(dfs, ignore_index=True)
                if 'Activity' in df1.columns and 'Tour' in df1.columns:
                    df1['Location'] = df1['Activity'].fillna(df1['Tour'])
                elif 'Activity' in df1.columns:
                    df1['Location'] = df1['Activity']
                elif 'Tour' in df1.columns:
                    df1['Location'] = df1['Tour']

    # Load visit date files (files with "date booked" or "dated booked" in name)
    df2 = None
    if os.path.exists(booking_data_path):
        all_files = glob.glob(os.path.join(booking_data_path, '*.xls')) + glob.glob(os.path.join(booking_data_path, '*.xlsx'))
        visit_files = [f for f in all_files if 'date booked' in os.path.basename(f).lower() or 'dated booked' in os.path.basename(f).lower()]
        if visit_files:
            dfs = []
            for f in visit_files:
                try:
                    engine = 'xlrd' if f.endswith('.xls') else 'openpyxl'
                    df = pd.read_excel(f, engine=engine)
                    dfs.append(df)
                except Exception:
                    pass
            if dfs:
                df2 = pd.concat(dfs, ignore_index=True)
                if 'Activity' in df2.columns and 'Tour' in df2.columns:
                    df2['Location'] = df2['Activity'].fillna(df2['Tour'])
                elif 'Activity' in df2.columns:
                    df2['Location'] = df2['Activity']
                elif 'Tour' in df2.columns:
                    df2['Location'] = df2['Tour']

    # Load marketing files from 'marketing data' folder
    google_ads_df = None
    meta_ads_df = None
    if os.path.exists(marketing_data_path):
        csv_files = glob.glob(os.path.join(marketing_data_path, '*.csv'))

        google_dfs = []
        meta_dfs = []

        for f in csv_files:
            name_lower = os.path.basename(f).lower()
            try:
                if 'google' in name_lower:
                    with open(f, 'r', encoding='utf-8') as file:
                        content = file.read()
                    lines = content.split('\n')
                    csv_content = '\n'.join(lines[2:])  # Skip first 2 header rows
                    df = pd.read_csv(StringIO(csv_content))
                    df['Platform'] = 'Google Ads'
                    column_mapping = {
                        'Campaign': 'campaign_name', 'Cost': 'spend',
                        'Conversions': 'conversions', 'Conv. value': 'conversion_value',
                        'Clicks': 'clicks', 'Impr.': 'impressions', 'CTR': 'ctr',
                        'Avg. CPC': 'cpc', 'Avg. cost': 'avg_cost'
                    }
                    df = df.rename(columns=column_mapping)
                    for col in ['spend', 'conversions', 'conversion_value', 'clicks', 'impressions']:
                        if col in df.columns:
                            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    google_dfs.append(df)
                elif 'meta' in name_lower:
                    df = pd.read_csv(f)
                    df['Platform'] = 'Meta Ads'
                    column_mapping = {
                        'Campaign name': 'campaign_name', 'Amount spent (EUR)': 'spend',
                        'Purchases': 'conversions', 'Purchases conversion value': 'conversion_value',
                        'Reach': 'reach', 'Link clicks': 'clicks',
                        'CTR (link click-through rate)': 'ctr',
                        'CPC (cost per link click) (EUR)': 'cpc',
                        'CPM (cost per 1,000 impressions) (EUR)': 'cpm',
                        'Results': 'results'
                    }
                    df = df.rename(columns=column_mapping)
                    for col in ['spend', 'conversions', 'conversion_value', 'reach', 'clicks', 'results']:
                        if col in df.columns:
                            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    meta_dfs.append(df)
            except Exception:
                pass

        if google_dfs:
            google_ads_df = pd.concat(google_dfs, ignore_index=True)
        if meta_dfs:
            meta_ads_df = pd.concat(meta_dfs, ignore_index=True)

    # Check if any data was loaded
    if df1 is None and df2 is None and google_ads_df is None and meta_ads_df is None:
        return None, None, None, None, "No data files found in local_data folders"

    return df1, df2, google_ads_df, meta_ads_df, None


def has_local_data():
    """Check if local 'booking data' or 'marketing data' folders have data files."""
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    booking_data_path = os.path.join(parent_dir, 'booking data')
    marketing_data_path = os.path.join(parent_dir, 'marketing data')

    # Check for Excel files in booking data folder
    if os.path.exists(booking_data_path):
        files = glob.glob(os.path.join(booking_data_path, '*.xls')) + \
                glob.glob(os.path.join(booking_data_path, '*.xlsx'))
        if files:
            return True

    # Check for CSV files in marketing data folder
    if os.path.exists(marketing_data_path):
        files = glob.glob(os.path.join(marketing_data_path, '*.csv'))
        if files:
            return True

    return False


# Password protection - Show login page if not authenticated
if not st.session_state.authenticated:
    # Hide sidebar on login page and style login button
    st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none; }
    button[kind="primary"] {
        background-color: #3C3C3C !important;
        color: #fff !important;
        border: none !important;
    }
    button[kind="primary"]:hover {
        background-color: #2a2a2a !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Centered login container
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='height: 80px'></div>", unsafe_allow_html=True)

        # Kuuma logo centered using columns
        _, logo_col, _ = st.columns([1.5, 1, 1.5])
        with logo_col:
            st.image("assets/logo_black.svg", use_container_width=True)

        st.markdown("<h2 style='text-align: center; margin-bottom: 0.5rem;'>Kuuma Booking Analyzer</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #666; margin-bottom: 2rem;'>Customer insights & booking intelligence</p>", unsafe_allow_html=True)

        # Password input
        password = st.text_input("Enter password", type="password", key="login_password")

        # Login button
        if st.button("Login", use_container_width=True, type="primary"):
            if password == "Kuuma2026!":
                st.session_state.authenticated = True

                # Load data immediately after login
                # First check for local data (for development/testing)
                if has_local_data():
                    progress_bar = st.progress(0, text="Loading local data...")
                    progress_bar.progress(50, text="Loading local files...")

                    df1, df2, google_ads_df, meta_ads_df, error = load_local_files()

                    if error:
                        progress_bar.empty()
                        st.warning(f"Local data: {error}")
                    else:
                        progress_bar.progress(95, text="Processing data...")
                        if df1 is not None:
                            st.session_state.df1 = apply_demo_transform(df1)
                        if df2 is not None:
                            st.session_state.df2 = apply_demo_transform(df2)
                        if google_ads_df is not None:
                            st.session_state.google_ads_df = google_ads_df
                        if meta_ads_df is not None:
                            st.session_state.meta_ads_df = meta_ads_df
                        st.session_state.drive_loaded = True

                        progress_bar.progress(100, text="Complete!")
                        time.sleep(0.3)
                        progress_bar.empty()

                # Otherwise try Google Drive
                elif GOOGLE_DRIVE_AVAILABLE and "gcp_service_account" in st.secrets and "google_drive" in st.secrets:
                    progress_bar = st.progress(0, text="Connecting to Google Drive...")

                    # Progress callback that updates the progress bar
                    def update_login_progress(progress, text):
                        progress_bar.progress(min(progress, 100), text=text)

                    # Load data with progress callback
                    df1, df2, google_ads_df, meta_ads_df, error = load_files_from_drive(progress_callback=update_login_progress)

                    if error:
                        progress_bar.empty()
                        st.warning(f"Google Drive: {error}")
                    else:
                        # Store data in session state
                        update_login_progress(95, "Processing data...")
                        if df1 is not None:
                            st.session_state.df1 = apply_demo_transform(df1)
                        if df2 is not None:
                            st.session_state.df2 = apply_demo_transform(df2)
                        if google_ads_df is not None:
                            st.session_state.google_ads_df = google_ads_df
                        if meta_ads_df is not None:
                            st.session_state.meta_ads_df = meta_ads_df
                        st.session_state.drive_loaded = True

                        # Complete progress
                        update_login_progress(100, "Complete!")
                        time.sleep(0.3)
                        progress_bar.empty()

                if st.session_state.drive_loaded and (
                    st.session_state.df1 is not None or st.session_state.df2 is not None
                ):
                    st.switch_page("pages/1_Overview.py")
                else:
                    st.rerun()
            else:
                st.error("Incorrect password. Please try again.")

    st.stop()

# ============ AUTHENTICATED CONTENT BELOW ============

# Header with logo
col1, col2 = st.columns([1, 5])
with col1:
    st.image("assets/logo_black.svg", width=120)
with col2:
    st.title("Kuuma Booking Analyzer")
    st.markdown("**Customer insights & booking intelligence**")

if DEMO_MODE:
    st.info(":material/science: **Demo account** — All data shown is sample data for demonstration purposes")

# Reserve container for date range selector (filled after data loads)
date_range_container = st.container()

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

# Try to load data automatically (local first, then Google Drive)
if not st.session_state.drive_loaded:
    # First check for local data (for development/testing)
    if has_local_data():
        progress_bar = st.sidebar.progress(0, text="Loading local data...")
        progress_bar.progress(50, text="Loading local files...")

        df1, df2, google_ads_df, meta_ads_df, error = load_local_files()

        if error:
            progress_bar.empty()
            st.sidebar.warning(f"Local: {error}")
        else:
            progress_bar.progress(95, text="Processing data...")
            if df1 is not None:
                st.session_state.df1 = apply_demo_transform(df1)
            if df2 is not None:
                st.session_state.df2 = apply_demo_transform(df2)
            if google_ads_df is not None:
                st.session_state.google_ads_df = google_ads_df
            if meta_ads_df is not None:
                st.session_state.meta_ads_df = meta_ads_df

            progress_bar.progress(100, text="Complete!")
            time.sleep(0.3)
            progress_bar.empty()
            st.session_state.drive_loaded = True

    # Otherwise try Google Drive
    elif GOOGLE_DRIVE_AVAILABLE and "gcp_service_account" in st.secrets and "google_drive" in st.secrets:
        progress_bar = st.sidebar.progress(0, text="Connecting to Google Drive...")

        # Progress callback for sidebar
        def update_sidebar_progress(progress, text):
            progress_bar.progress(min(progress, 100), text=text)

        df1, df2, google_ads_df, meta_ads_df, error = load_files_from_drive(progress_callback=update_sidebar_progress)

        if error:
            progress_bar.empty()
            st.sidebar.warning(f"Drive: {error}")
        else:
            update_sidebar_progress(95, "Processing data...")
            if df1 is not None:
                st.session_state.df1 = apply_demo_transform(df1)
            if df2 is not None:
                st.session_state.df2 = apply_demo_transform(df2)
            if google_ads_df is not None:
                st.session_state.google_ads_df = google_ads_df
            if meta_ads_df is not None:
                st.session_state.meta_ads_df = meta_ads_df

            update_sidebar_progress(100, "Complete!")
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

# Manual upload option (in expander) - show when data is loaded from Drive or local
if st.session_state.drive_loaded:
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
                st.session_state.df1 = apply_demo_transform(df1)
                st.success(f"Loaded: {len(df1):,} rows")

        if uploaded_files2:
            df2, errors2, file_info2 = load_and_merge_files(uploaded_files2)
            if errors2:
                for error in errors2:
                    st.error(f"Error: {error}")
            elif df2 is not None:
                st.session_state.df2 = apply_demo_transform(df2)
                st.success(f"Loaded: {len(df2):,} rows")

if not st.session_state.drive_loaded:
    # No data loaded - show standard upload
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
            st.session_state.df1 = apply_demo_transform(df1)
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
            st.session_state.df2 = apply_demo_transform(df2)
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
        st.page_link("pages/1_Overview.py", label="Overview", icon=":material/home:")
        st.page_link("app.py", label="Booking Patterns", icon=":material/bar_chart:")
        st.page_link("pages/3_Customers.py", label="Recurring Customers", icon=":material/group:")
        st.page_link("pages/4_Revenue.py", label="Revenue & Value", icon=":material/payments:")
        st.page_link("pages/5_Promotions.py", label="Promotions", icon=":material/sell:")
        st.page_link("pages/6_Capacity.py", label="Capacity Analysis", icon=":material/analytics:")
        st.page_link("pages/7_Marketing.py", label="Marketing", icon=":material/campaign:")
        st.page_link("pages/8_Chart_Test.py", label="Chart Test", icon=":material/science:")
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

        # Pre-compute booking hour and day of week for heatmap (performance optimization)
        merged_clean['booking_hour'] = merged_clean['booking_date'].dt.hour
        merged_clean['booking_dow'] = merged_clean['booking_date'].dt.day_name()

        return merged_clean, unmatched_count, invalid_count, negative_count

    @st.cache_data
    def compute_heatmap_data(booking_hours, booking_dows):
        """Cached computation of heatmap pivot table and metrics."""
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        # Create DataFrame for groupby
        df = pd.DataFrame({'booking_hour': booking_hours, 'booking_dow': booking_dows})
        df['booking_dow'] = pd.Categorical(df['booking_dow'], categories=day_order, ordered=True)

        # Create pivot table
        heatmap_pivot = df.groupby(['booking_hour', 'booking_dow']).size().unstack(fill_value=0)
        heatmap_pivot = heatmap_pivot.reindex(columns=day_order, fill_value=0)

        # Calculate metrics
        peak_hour = int(df['booking_hour'].mode().iloc[0]) if len(df) > 0 else 0
        peak_day = str(df['booking_dow'].mode().iloc[0]) if len(df) > 0 else 'Unknown'
        evening_pct = (df['booking_hour'] >= 18).sum() / len(df) * 100 if len(df) > 0 else 0

        return heatmap_pivot, peak_hour, peak_day, evening_pct

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
            # Filter out NaN values and non-location entries (UTM test, etc.)
            all_locations = sorted([loc for loc in processed_data['location'].unique()
                                   if pd.notna(loc) and str(loc).lower().startswith('kuuma')])
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
                   f"{unmatched:,} unmatched (likely cancelled bookings) | {invalid:,} invalid dates | {negative:,} negative intervals")

            # Distribution chart
            st.markdown("### Booking Lead Time Distribution")

            # Use cached distribution calculation
            distribution, distribution_pct = calculate_distribution_data(
                tuple(filtered_data['interval_category'].tolist())
            )

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
                height=450,
                margin=dict(t=50),
                yaxis=dict(range=[0, 105])  # Give room for labels above bars
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

                with st.expander("Understanding Average vs Median", expanded=False):
                    st.markdown("""
                    - **Average** - Includes all bookings, including customers who book far in advance (e.g., 30 days ahead)
                    - **Median** - Shows what a typical customer actually does, ignoring extreme values

                    **Example:** If most customers book 0-2 days ahead, but a few book 30 days ahead:
                    - Average might be 5 days (pulled up by advance planners)
                    - Median would be 1 day (the typical customer)

                    **What to look for:** A large gap between Average and Median means you have mostly last-minute bookers
                    with some advance planners. A small gap means consistent booking behavior.
                    """)

                # Use cached location stats calculation
                location_stats = calculate_location_stats(
                    tuple(filtered_data.index.tolist()),
                    tuple(filtered_data['location'].tolist()),
                    tuple(filtered_data['interval_days'].tolist())
                )

                location_stats_config = {
                    'Total Bookings': st.column_config.NumberColumn('Total Bookings', help='Number of bookings for this location'),
                    'Avg Lead Time (days)': st.column_config.NumberColumn('Avg Lead Time (days)', help='Average days between booking and visit'),
                    'Median Lead Time (days)': st.column_config.NumberColumn('Median Lead Time (days)', help='Typical days between booking and visit (ignoring outliers)'),
                }
                st.dataframe(location_stats, use_container_width=True, column_config=location_stats_config)

                # Booking Decision Heatmap
                st.markdown("#### When Do Customers Book?")
                st.markdown("This heatmap shows when customers make their booking decisions (hour of day vs day of week).")

                # Location filter for heatmap
                heatmap_locations = ['All Locations'] + sorted(filtered_data['location'].dropna().unique().tolist())
                selected_heatmap_location = st.selectbox(
                    "Select Location",
                    options=heatmap_locations,
                    index=0,
                    key="booking_heatmap_location_filter"
                )

                # Filter data based on location selection (use pre-computed columns)
                if selected_heatmap_location == 'All Locations':
                    heatmap_hours = filtered_data['booking_hour'].tolist()
                    heatmap_dows = filtered_data['booking_dow'].tolist()
                else:
                    loc_mask = filtered_data['location'] == selected_heatmap_location
                    heatmap_hours = filtered_data.loc[loc_mask, 'booking_hour'].tolist()
                    heatmap_dows = filtered_data.loc[loc_mask, 'booking_dow'].tolist()

                # Use cached function for heatmap computation
                if len(heatmap_hours) > 0:
                    heatmap_pivot, peak_hour, peak_day, evening_pct = compute_heatmap_data(
                        tuple(heatmap_hours), tuple(heatmap_dows)
                    )

                    fig_heatmap = px.imshow(
                        heatmap_pivot,
                        labels=dict(x='Day of Week', y='Hour of Day', color='Bookings'),
                        aspect='auto',
                        color_continuous_scale='YlOrRd'
                    )
                    fig_heatmap.update_yaxes(tickvals=list(range(0, 24, 2)))
                    st.plotly_chart(fig_heatmap, use_container_width=True)

                    # Peak booking metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Peak Booking Hour", f"{peak_hour}:00")
                    with col2:
                        st.metric("Peak Booking Day", peak_day)
                    with col3:
                        st.metric("Evening Bookings (after 18:00)", f"{evening_pct:.1f}%")

                    with st.expander("Marketing Insight: How to use this data", expanded=False):
                        st.markdown("""
                        **This heatmap reveals when customers actually make booking decisions**, which directly impacts how you should schedule and allocate marketing spend.

                        | Booking Pattern | Marketing Action |
                        |-----------------|------------------|
                        | **Peak booking hours** | Increase ad bids during these hours, ensure ads are running |
                        | **Peak booking days** | Allocate more daily budget to these days |
                        | **Low activity periods** | Reduce ad spend, avoid wasting budget |
                        | **High evening bookings** | Schedule Meta/Google ads for evening delivery |

                        **Practical Applications:**
                        - **Google Ads**: Use ad scheduling to increase bids +20-30% during peak booking hours
                        - **Meta Ads**: Set dayparting to prioritize delivery when customers are actively booking
                        - **Email campaigns**: Send marketing emails to arrive 1-2 hours before peak booking times
                        - **Customer support**: Staff live chat during high-booking periods to capture conversions

                        **Example:** If peak booking is Sunday at 20:00 with 45% evening bookings → Schedule ads heavily for Sunday evenings, reduce weekday morning spend, and optimize checkout for mobile (evening = phone users).
                        """)

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

                    with st.expander("How to read the bubble plot", expanded=False):
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
