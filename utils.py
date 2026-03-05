"""
Shared utilities for Kuuma Booking Analyzer
Contains common functions used across all pages
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta


# ============== Constants ==============

PAGES = {
    "Overview": "app",
    "Booking Patterns": "pages/1_Booking_Patterns",
    "Operations": "pages/2_Operations",
    "Customers": "pages/3_Customers",
    "Marketing": "pages/4_Marketing"
}

LOCATIONS = [
    "Kuuma Noord",
    "Kuuma Sloterplas",
    "Kuuma Marineterrein Bjørk",
    "Kuuma Marineterrein Matsu",
    "Kuuma Aan 't IJ (Centrum)"
]


# ============== Page Config ==============

def setup_page(page_title="Kuuma Booking Analyzer"):
    """Configure page settings - call at top of each page"""
    st.set_page_config(
        page_title=page_title,
        page_icon="🔥",
        layout="wide"
    )

    # Hide default Streamlit navigation
    hide_default_nav = """
    <style>
    [data-testid="stSidebarNav"] {
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


def render_header():
    """Render the app header with logo"""
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("assets/logo_black.svg", width=120)
    with col2:
        st.title("Kuuma Booking Analyzer")
        st.markdown("**Customer insights & booking intelligence**")


def render_sidebar_navigation():
    """Render navigation links at top of sidebar - only if data is loaded"""
    if not has_data():
        return

    st.sidebar.markdown("### Navigation")
    st.sidebar.page_link("app.py", label="Booking Patterns", icon=":material/bar_chart:")
    st.sidebar.page_link("pages/2_Operations.py", label="Operations", icon=":material/calendar_month:")
    st.sidebar.page_link("pages/3_Customers.py", label="Customers", icon=":material/group:")
    st.sidebar.markdown("---")


def render_sidebar_upload():
    """Render file upload section at bottom of sidebar"""
    st.sidebar.markdown("### Data Upload")

    uploaded_file1 = st.sidebar.file_uploader(
        "Booking Creation Dates (.xls/.xlsx)",
        type=["xls", "xlsx"],
        help="Upload the file containing when bookings were created",
        key="file1_uploader"
    )

    uploaded_file2 = st.sidebar.file_uploader(
        "Visit Dates (.xls/.xlsx)",
        type=["xls", "xlsx"],
        help="Upload the file containing when customers actually visited",
        key="file2_uploader"
    )

    return uploaded_file1, uploaded_file2


def render_sidebar_column_mapping():
    """Render column mapping configuration"""
    if st.session_state.df1 is None or st.session_state.df2 is None:
        return None

    df1 = st.session_state.df1
    df2 = st.session_state.df2

    st.sidebar.markdown("### Column Mapping")

    # Smart defaults
    default_id_col = "Booking number" if "Booking number" in df1.columns else df1.columns[0]
    default_created_col = "Created" if "Created" in df1.columns else df1.columns[0]
    default_start_col = "Start" if "Start" in df2.columns else df2.columns[0]

    if "Tour" in df1.columns:
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
        help="Column that uniquely identifies each booking",
        key="id_col_1"
    )

    date_col_1 = st.sidebar.selectbox(
        "Booking Creation Date (File 1)",
        options=df1.columns.tolist(),
        index=df1.columns.tolist().index(default_created_col) if default_created_col in df1.columns else 0,
        help="When the customer made the booking",
        key="date_col_1"
    )

    id_col_2 = st.sidebar.selectbox(
        "Booking ID (File 2)",
        options=df2.columns.tolist(),
        index=df2.columns.tolist().index(default_id_col) if default_id_col in df2.columns else 0,
        help="Column that uniquely identifies each booking",
        key="id_col_2"
    )

    visit_col_2 = st.sidebar.selectbox(
        "Visit Date (File 2)",
        options=df2.columns.tolist(),
        index=df2.columns.tolist().index(default_start_col) if default_start_col in df2.columns else 0,
        help="When the customer actually visited",
        key="visit_col_2"
    )

    # Location column (optional)
    location_options = ["None"] + df1.columns.tolist()
    location_default_index = location_options.index(default_location_col) if default_location_col in location_options else 0

    location_col = st.sidebar.selectbox(
        "Location (optional)",
        options=location_options,
        index=location_default_index,
        help="Column containing location/branch names for comparison",
        key="location_col"
    )

    # Email column
    default_email_col = "Email address" if "Email address" in df1.columns else None
    email_options = ["None"] + df1.columns.tolist()
    email_default_index = email_options.index(default_email_col) if default_email_col in email_options else 0

    email_col = st.sidebar.selectbox(
        "Email Address (optional)",
        options=email_options,
        index=email_default_index,
        help="Column containing customer email for recurring customer analysis",
        key="email_col"
    )

    # Source column for marketing
    default_source_col = "Source" if "Source" in df1.columns else None
    source_options = ["None"] + df1.columns.tolist()
    source_default_index = source_options.index(default_source_col) if default_source_col in source_options else 0

    source_col = st.sidebar.selectbox(
        "Marketing Source (optional)",
        options=source_options,
        index=source_default_index,
        help="Column containing marketing channel/source for attribution",
        key="source_col"
    )

    return {
        'id_col_1': id_col_1,
        'date_col_1': date_col_1,
        'id_col_2': id_col_2,
        'visit_col_2': visit_col_2,
        'location_col': location_col,
        'email_col': email_col,
        'source_col': source_col
    }


# ============== Data Loading ==============

@st.cache_data
def load_excel_file(uploaded_file):
    """Load an Excel file and return dataframe"""
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


def initialize_session_state():
    """Initialize session state variables"""
    if 'df1' not in st.session_state:
        st.session_state.df1 = None
    if 'df2' not in st.session_state:
        st.session_state.df2 = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'data_stats' not in st.session_state:
        st.session_state.data_stats = None


def load_uploaded_files(uploaded_file1, uploaded_file2):
    """Process uploaded files and store in session state"""
    if uploaded_file1 is not None:
        df1, error1 = load_excel_file(uploaded_file1)
        if error1:
            st.sidebar.error(f"Error reading File 1: {error1}")
        else:
            st.session_state.df1 = df1
            st.sidebar.success(f"File 1: {len(df1):,} rows")

    if uploaded_file2 is not None:
        df2, error2 = load_excel_file(uploaded_file2)
        if error2:
            st.sidebar.error(f"Error reading File 2: {error2}")
        else:
            st.session_state.df2 = df2
            st.sidebar.success(f"File 2: {len(df2):,} rows")


# ============== Data Processing ==============

@st.cache_data
def process_booking_data(df1, df2, id_col_1, date_col_1, id_col_2, visit_col_2, location_col, email_col=None, source_col=None):
    """
    Process and merge booking data from both files.
    Returns processed dataframe and statistics.
    """
    # Prepare dataframes with renamed columns
    cols_to_keep_1 = [id_col_1, date_col_1]
    df1_prep = df1[cols_to_keep_1].copy()
    df1_prep.columns = ['booking_id', 'booking_date']

    df2_prep = df2[[id_col_2, visit_col_2]].copy()
    df2_prep.columns = ['booking_id', 'visit_date']

    # Add optional columns from df1
    if location_col != "None":
        df1_prep['location'] = df1[location_col].values

    if email_col and email_col != "None":
        df1_prep['email'] = df1[email_col].values

    if source_col and source_col != "None":
        df1_prep['source'] = df1[source_col].values

    # Add additional columns for analysis
    if 'Participants' in df1.columns:
        df1_prep['participants'] = df1['Participants'].values
    if 'Adults' in df1.columns:
        df1_prep['adults'] = df1['Adults'].values
    if 'Total gross' in df1.columns:
        df1_prep['revenue'] = df1['Total gross'].values
    if 'Promotion' in df1.columns:
        df1_prep['promotion'] = df1['Promotion'].values

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

    stats = {
        'unmatched': unmatched_count,
        'invalid': invalid_count,
        'negative': negative_count,
        'total_matched': len(merged_clean)
    }

    return merged_clean, stats


def get_filtered_data(processed_data, date_range=None, selected_locations=None, location_col="None"):
    """Apply filters to processed data"""
    filtered = processed_data.copy()

    # Date filter
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered['visit_date'].dt.date >= start_date) &
            (filtered['visit_date'].dt.date <= end_date)
        ]

    # Location filter
    if selected_locations and location_col != "None" and 'location' in filtered.columns:
        filtered = filtered[filtered['location'].isin(selected_locations)]

    return filtered


# ============== Temperature Analysis ==============

@st.cache_data
def get_temperature_data(start_date, end_date, latitude=52.37, longitude=4.89):
    """Fetch daily average temperature data from Open-Meteo API"""
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
        pass

    # Fallback to Meteostat
    try:
        from meteostat import Point, Daily

        location = Point(latitude, longitude)
        data = Daily(location, start_date, end_date)
        meteo_df = data.fetch()

        if meteo_df is not None and not meteo_df.empty and 'tavg' in meteo_df.columns:
            meteo_df = meteo_df.reset_index()
            temp_df = pd.DataFrame({
                'date': pd.to_datetime(meteo_df['time']),
                'temperature': meteo_df['tavg']
            })
            temp_df = temp_df.dropna(subset=['temperature'])
            return temp_df
    except Exception:
        return None

    return None


def add_temperature_to_bookings(bookings_df):
    """Add daily average temperature data to bookings based on booking date"""
    if 'booking_date' not in bookings_df.columns:
        return bookings_df

    start_date = bookings_df['booking_date'].min() - pd.Timedelta(days=1)
    end_date = bookings_df['booking_date'].max() + pd.Timedelta(days=1)

    today = pd.Timestamp.now().normalize()
    if end_date > today:
        end_date = today

    temp_df = get_temperature_data(start_date, end_date)

    if temp_df is None:
        return bookings_df

    bookings_with_temp = bookings_df.copy()
    bookings_with_temp['booking_date_only'] = bookings_with_temp['booking_date'].dt.date
    temp_df['date_only'] = temp_df['date'].dt.date

    merged = bookings_with_temp.merge(
        temp_df,
        left_on='booking_date_only',
        right_on='date_only',
        how='left'
    )

    merged['temp_category'] = pd.cut(
        merged['temperature'],
        bins=[-np.inf, 5, 10, 13, 16, 20, np.inf],
        labels=['Below 5°C', '5-10°C', '10-13°C', '13-16°C', '16-20°C', 'Above 20°C']
    )

    return merged


# ============== Helper Functions ==============

def calculate_metrics(filtered_data):
    """Calculate key metrics from filtered data"""
    if len(filtered_data) == 0:
        return None

    return {
        'avg_interval': filtered_data['interval_days'].mean(),
        'median_interval': filtered_data['interval_days'].median(),
        'total_bookings': len(filtered_data),
        'same_day_pct': (filtered_data['interval_days'] == 0).sum() / len(filtered_data) * 100,
        'min_booking_date': filtered_data['booking_date'].min(),
        'max_booking_date': filtered_data['booking_date'].max()
    }


def has_data():
    """Check if data is loaded and processed"""
    return (
        st.session_state.get('df1') is not None and
        st.session_state.get('df2') is not None
    )


def show_no_data_message():
    """Show message when no data is loaded"""
    st.info("**No data loaded.** Please upload your Excel files using the sidebar to begin analysis.")
    st.markdown("""
    ### Getting Started
    1. Scroll down in the sidebar to find **Data Upload**
    2. Upload your booking creation dates file
    3. Upload your visit dates file
    4. Configure column mappings
    5. Explore insights across all pages!
    """)
