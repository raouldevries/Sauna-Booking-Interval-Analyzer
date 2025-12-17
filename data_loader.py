"""
Centralized data loader for Kuuma Booking Analyzer.
Provides shared data loading and caching functions used across all pages.
"""

import streamlit as st
import pandas as pd
from io import BytesIO


def init_session_state():
    """Initialize all session state variables for data storage."""
    defaults = {
        'df1': None,           # Booking creation dates
        'df2': None,           # Visit dates
        'google_ads_df': None, # Google Ads data
        'meta_ads_df': None,   # Meta Ads data
        'drive_loaded': False, # Flag for Drive loading
        'processed_data': None, # Cached processed data
        'data_hash': None,     # Hash to detect data changes
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
