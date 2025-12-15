"""
Kuuma Booking Analyzer - Marketing Campaign Analysis Page
Analyze Google Ads and Meta Ads campaign performance with SEE-THINK-DO-CARE framework
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import sys
sys.path.insert(0, '..')
from data_loader import init_session_state

# Page configuration
st.set_page_config(
    page_title="Kuuma - Marketing Analysis",
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

# Location keyword mapping for filtering campaigns
# Maps campaign keywords to display names
LOCATION_KEYWORDS = {
    'Amsterdam Marine': ['marine', 'marineterrein', 'amsterdam m,', 'matsu', 'bjørk', 'bjork'],
    'Amsterdam Sloterplas': ['sloterplas', 'sloterpas'],
    'Amsterdam Noord': ['amsterdam n', '| amsterdam n'],
    'Amsterdam IJ': ['amsterdam ij', 'aan t ij', 'aan \'t ij', 'centrum'],
    'Nijmegen': ['nijmegen', 'nyma', 'lent'],
    'Rotterdam': ['rotterdam', 'delfshaven', 'rijnhaven'],
    'Scheveningen': ['scheveningen'],
    'Den Bosch': ['den bosch', 'denbosch'],
    'Katwijk': ['katwijk'],
    'Breda': ['breda'],
    'Wijk aan Zee': ['wijk aan zee'],
    'Bergen aan Zee': ['bergen aan zee'],
    'Bloemendaal': ['bloemendaal'],
}

# Capacity data from Bezettings analyse (weekly capacity per location)
# dal = off-peak (Mon-Thu 10:00-16:00), piek = peak (Mon-Thu other hours)
# weekday = dal + piek, weekend = Fri-Sun
LOCATION_CAPACITY = {
    'Kuuma Marineterrein Matsu': {'dal': 96, 'piek': 366, 'weekday': 462, 'weekend': 198, 'cluster': 'Flagship'},
    'Kuuma Marineterrein Bjørk': {'dal': 128, 'piek': 488, 'weekday': 616, 'weekend': 264, 'cluster': 'Flagship'},
    'Kuuma Noord': {'dal': 96, 'piek': 336, 'weekday': 432, 'weekend': 192, 'cluster': 'Groeier'},
    'Kuuma Sloterplas': {'dal': 112, 'piek': 392, 'weekday': 504, 'weekend': 224, 'cluster': 'Groeier'},
    'Kuuma Aan ´t IJ (Centrum)': {'dal': 96, 'piek': 324, 'weekday': 420, 'weekend': 180, 'cluster': 'Groeier'},
    'Kuuma Nijmegen Lent': {'dal': 80, 'piek': 300, 'weekday': 380, 'weekend': 160, 'cluster': 'Groeier'},
    'Kuuma Rotterdam Rijnhaven': {'dal': 80, 'piek': 300, 'weekday': 380, 'weekend': 160, 'cluster': 'Groeier'},
    'Kuuma Scheveningen': {'dal': 120, 'piek': 450, 'weekday': 570, 'weekend': 240, 'cluster': 'Flagship'},
    'Kuuma Den Bosch': {'dal': 80, 'piek': 300, 'weekday': 380, 'weekend': 160, 'cluster': 'Groeier'},
    'Kuuma Katwijk': {'dal': 80, 'piek': 300, 'weekday': 380, 'weekend': 160, 'cluster': 'Groeier'},
}

# Map marketing location names to booking data location names
LOCATION_NAME_MAP = {
    # Marketing name -> Booking data location names (list for multiple matches)
    'Amsterdam Marine': ['Kuuma Marineterrein Matsu', 'Kuuma Marineterrein Bjørk'],
    'Amsterdam Sloterplas': ['Kuuma Sloterplas'],
    'Amsterdam Noord': ['Kuuma Noord'],
    'Amsterdam IJ': ['Kuuma Aan ´t IJ (Centrum)'],
    'Nijmegen': ['Kuuma Nijmegen Lent'],
    'Rotterdam': ['Kuuma Rotterdam Rijnhaven', 'Kuuma Rotterdam Delfshaven'],
    'Scheveningen': ['Kuuma Scheveningen'],
    'Den Bosch': ['Kuuma Den Bosch'],
    'Katwijk': ['Kuuma Katwijk'],
}

# STDC phase colors (matching the plan)
STDC_COLORS = {
    'SEE': '#3498db',      # Blue
    'THINK': '#f39c12',    # Orange
    'DO': '#27ae60',       # Green
    'CARE': '#9b59b6',     # Purple
    'Untagged': '#9ca3af'  # Gray
}

# Default STDC suggestions based on campaign keywords
# Google Ads naming: NL | S | ... (Search), NL | PM | ... (Performance Max)
# Meta naming: Clicks | ..., Reach - ..., Conversions | ...
STDC_SUGGESTIONS = {
    'SEE': ['display', 'demand gen', 'reach', 'awareness', 'see'],
    'THINK': ['non-branded', 'non branded', 'think', 'consideration', 'clicks |'],
    'DO': ['| s |', '| pm |', 'branded', 'brand', 'conversions', 'conversion', 'do', 'purchase'],
    'CARE': ['retargeting', 'rm', 'remarketing', 'care', 'loyalty']
}


def suggest_stdc_phase(campaign_name):
    """Suggest STDC phase based on campaign name keywords."""
    if pd.isna(campaign_name) or not isinstance(campaign_name, str):
        return 'Untagged'
    name_lower = campaign_name.lower()

    # Priority 1: Check for Google Ads campaign types (S=Search, PM=Performance Max) → DO
    if '| s |' in name_lower or '| pm |' in name_lower:
        return 'DO'

    # Priority 2: Check for remarketing/retargeting → CARE
    for keyword in STDC_SUGGESTIONS['CARE']:
        if keyword in name_lower:
            return 'CARE'

    # Priority 3: Check for conversion campaigns → DO
    for keyword in ['conversions', 'conversion', 'purchase']:
        if keyword in name_lower:
            return 'DO'

    # Priority 4: Check for awareness campaigns → SEE
    for keyword in STDC_SUGGESTIONS['SEE']:
        if keyword in name_lower:
            return 'SEE'

    # Priority 5: Check for consideration campaigns → THINK
    for keyword in STDC_SUGGESTIONS['THINK']:
        if keyword in name_lower:
            return 'THINK'

    return 'Untagged'


def campaign_matches_location(campaign_name, locations=None):
    """Check if campaign name contains any location keywords."""
    if pd.isna(campaign_name) or not isinstance(campaign_name, str):
        return None

    name_lower = campaign_name.lower()

    # Check for "alle locaties" (all locations)
    if 'alle locaties' in name_lower or 'all locations' in name_lower:
        return 'all'

    # Search all location keywords
    for location, keywords in LOCATION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in name_lower:
                return location
    return None


def get_revenue_per_location(df1, location_col='Tour', revenue_col='Total gross', split_marine=False):
    """Calculate revenue and bookings per location from booking data."""
    if df1 is None or location_col not in df1.columns:
        return {}

    result = {}
    for marketing_loc, booking_locs in LOCATION_NAME_MAP.items():
        # Special handling for Amsterdam Marine - split into Matsu and Bjørk
        if split_marine and marketing_loc == 'Amsterdam Marine':
            for booking_loc in booking_locs:
                mask = df1[location_col] == booking_loc
                location_bookings = df1[mask]

                bookings = len(location_bookings)
                revenue = 0
                if revenue_col in df1.columns:
                    revenue = pd.to_numeric(location_bookings[revenue_col], errors='coerce').fillna(0).sum()

                # Use shorter display names
                if 'Matsu' in booking_loc:
                    display_name = 'Marineterrein Matsu'
                elif 'Bjørk' in booking_loc:
                    display_name = 'Marineterrein Bjørk'
                else:
                    display_name = booking_loc

                result[display_name] = {
                    'bookings': bookings,
                    'revenue': revenue,
                    'parent_marketing_loc': 'Amsterdam Marine'  # Track parent for ad spend split
                }
        else:
            # Filter bookings matching any of the booking location names
            mask = df1[location_col].isin(booking_locs)
            location_bookings = df1[mask]

            bookings = len(location_bookings)
            revenue = 0
            if revenue_col in df1.columns:
                revenue = pd.to_numeric(location_bookings[revenue_col], errors='coerce').fillna(0).sum()

            result[marketing_loc] = {
                'bookings': bookings,
                'revenue': revenue
            }

    return result


def get_capacity_per_location(marketing_loc, num_weeks=4):
    """Get weekly capacity for a marketing location."""
    if marketing_loc not in LOCATION_NAME_MAP:
        return None

    booking_locs = LOCATION_NAME_MAP[marketing_loc]
    total_weekly_capacity = 0
    total_weekend_capacity = 0

    for booking_loc in booking_locs:
        if booking_loc in LOCATION_CAPACITY:
            cap = LOCATION_CAPACITY[booking_loc]
            total_weekly_capacity += cap['weekday'] + cap['weekend']
            total_weekend_capacity += cap['weekend']

    # Return capacity for the period (num_weeks)
    return {
        'weekly_total': total_weekly_capacity,
        'period_total': total_weekly_capacity * num_weeks,
        'weekend_total': total_weekend_capacity * num_weeks
    }


@st.cache_data(show_spinner=False)
def calculate_marketing_metrics(_df_hash, df_json):
    """Calculate all marketing metrics with caching. Uses df_hash for cache key."""
    df = pd.read_json(StringIO(df_json))

    total_spend = df['spend'].sum()
    total_conversions = df['conversions'].sum()
    total_conv_value = df['conversion_value'].sum() if 'conversion_value' in df.columns else 0
    roas = (total_conv_value / total_spend * 100) if total_spend > 0 else 0
    cpa = (total_spend / total_conversions) if total_conversions > 0 else 0

    # Platform breakdown
    google_df = df[df['Platform'] == 'Google Ads']
    meta_df = df[df['Platform'] == 'Meta Ads']

    return {
        'total_spend': total_spend,
        'total_conversions': total_conversions,
        'total_conv_value': total_conv_value,
        'roas': roas,
        'cpa': cpa,
        'google_spend': google_df['spend'].sum(),
        'meta_spend': meta_df['spend'].sum(),
        'google_conv': google_df['conversions'].sum(),
        'meta_conv': meta_df['conversions'].sum(),
        'google_conv_value': google_df['conversion_value'].sum() if 'conversion_value' in google_df.columns else 0,
        'meta_conv_value': meta_df['conversion_value'].sum() if 'conversion_value' in meta_df.columns else 0,
    }


@st.cache_data(show_spinner=False)
def parse_google_ads_csv(uploaded_file):
    """Parse Google Ads CSV export (has 2-line header)."""
    try:
        content = uploaded_file.getvalue().decode('utf-8')
        lines = content.split('\n')

        # Skip first 2 lines (title and date range)
        csv_content = '\n'.join(lines[2:])

        df = pd.read_csv(StringIO(csv_content))
        df['Platform'] = 'Google Ads'

        # Standardize column names
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

        # Filter out summary/total rows (Total is in Campaign status column)
        if 'Campaign status' in df.columns:
            df = df[~df['Campaign status'].astype(str).str.startswith('Total:')]

        # Clean numeric columns
        for col in ['spend', 'conversions', 'conversion_value', 'impressions', 'clicks', 'cpc']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.replace(' --', '0'), errors='coerce').fillna(0)

        # Clean CTR
        if 'ctr' in df.columns:
            df['ctr'] = df['ctr'].astype(str).str.replace('%', '').str.replace(' --', '0')
            df['ctr'] = pd.to_numeric(df['ctr'], errors='coerce').fillna(0)

        return df, None
    except Exception as e:
        return None, str(e)


@st.cache_data(show_spinner=False)
def parse_meta_ads_csv(uploaded_file):
    """Parse Meta Ads CSV export."""
    try:
        df = pd.read_csv(uploaded_file)
        df['Platform'] = 'Meta Ads'

        # Standardize column names
        # Note: 'Purchases' is actual conversions, 'Results' is campaign objective result
        column_mapping = {
            'Campaign name': 'campaign_name',
            'Amount spent (EUR)': 'spend',
            'Purchases': 'conversions',  # Use Purchases for actual conversions
            'Purchases conversion value': 'conversion_value',
            'Reach': 'reach',
            'Link clicks': 'clicks',
            'CTR (link click-through rate)': 'ctr',
            'CPC (cost per link click) (EUR)': 'cpc',
            'CPM (cost per 1,000 impressions) (EUR)': 'cpm',
            'Results': 'results'  # Keep results separate (campaign objective)
        }

        df = df.rename(columns=column_mapping)

        # Clean numeric columns
        for col in ['spend', 'conversions', 'conversion_value', 'reach', 'clicks', 'cpc', 'cpm', 'purchases']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Clean CTR
        if 'ctr' in df.columns:
            df['ctr'] = pd.to_numeric(df['ctr'], errors='coerce').fillna(0)

        return df, None
    except Exception as e:
        return None, str(e)


# Header with logo
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://kuuma.nl/wp-content/themes/kuuma/images/logo.svg", width=120)
with col2:
    st.title("Kuuma Booking Analyzer")
    st.markdown("**Customer insights & booking intelligence**")

st.markdown("## Marketing Campaign Analysis")
st.markdown("Analyze Google Ads and Meta Ads performance using the SEE-THINK-DO-CARE framework")

# Reserve container for date range selector (filled after data loads)
date_range_container = st.container()

# Default file paths for auto-loading (only used for local development)
import os
DEFAULT_DATA_PATH = "/Users/raouldevries/Work/Kuuma/Booking analyzer"
DEFAULT_GOOGLE_ADS = f"{DEFAULT_DATA_PATH}/marketing data/google_ads.csv"
DEFAULT_META_ADS = f"{DEFAULT_DATA_PATH}/marketing data/meta_ads.csv"
DEFAULT_BOOKING_CREATION = f"{DEFAULT_DATA_PATH}/booking data/the day the booking was made.xls"
DEFAULT_VISIT_DATES = f"{DEFAULT_DATA_PATH}/booking data/the date booked.xls"

@st.cache_data(show_spinner=False)
def _load_local_excel(file_path, engine='xlrd'):
    """Load Excel file with caching."""
    if os.path.exists(file_path):
        return pd.read_excel(file_path, engine=engine)
    return None

@st.cache_data(show_spinner=False)
def _load_local_csv(file_path):
    """Load and process local CSV file with caching."""
    if not os.path.exists(file_path):
        return None
    return pd.read_csv(file_path)

def load_default_files():
    """Auto-load default data files if they exist (local development only)."""
    # Skip if already attempted loading
    if st.session_state.get('_local_files_checked', False):
        return
    st.session_state._local_files_checked = True

    # Load booking data (df1 - booking creation dates)
    if st.session_state.df1 is None:
        df = _load_local_excel(DEFAULT_BOOKING_CREATION)
        if df is not None:
            st.session_state.df1 = df

    # Load booking data (df2 - visit dates)
    if st.session_state.df2 is None:
        df = _load_local_excel(DEFAULT_VISIT_DATES)
        if df is not None:
            st.session_state.df2 = df

    # Load Google Ads data
    if st.session_state.google_ads_df is None:
        df = _load_local_csv(DEFAULT_GOOGLE_ADS)
        if df is not None:
            # Process Google Ads CSV (skip header rows already handled by pandas)
            df['Platform'] = 'Google Ads'
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
            if 'Campaign status' in df.columns:
                df = df[~df['Campaign status'].astype(str).str.startswith('Total:')]
            for col in ['spend', 'conversions', 'conversion_value', 'impressions', 'clicks', 'cpc']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.replace(' --', '0'), errors='coerce').fillna(0)
            if 'ctr' in df.columns:
                df['ctr'] = pd.to_numeric(df['ctr'].astype(str).str.replace('%', '').str.replace(' --', '0'), errors='coerce').fillna(0)
            st.session_state.google_ads_df = df

    # Load Meta Ads data
    if st.session_state.meta_ads_df is None:
        df = _load_local_csv(DEFAULT_META_ADS)
        if df is not None:
            df['Platform'] = 'Meta Ads'
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
            if 'campaign_name' in df.columns:
                df = df[df['campaign_name'].notna()]
                df = df[df['campaign_name'].astype(str).str.strip() != '']
            for col in ['spend', 'conversions', 'conversion_value', 'reach', 'clicks', 'cpc', 'cpm']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            if 'ctr' in df.columns:
                df['ctr'] = pd.to_numeric(df['ctr'], errors='coerce').fillna(0)
            st.session_state.meta_ads_df = df

# Initialize session state using centralized function
init_session_state()

# Auto-load default files
load_default_files()

# Always reset STDC tags to apply latest auto-tagging defaults
st.session_state.stdc_tags = {}

# Reserve container for navigation at top of sidebar
nav_container = st.sidebar.container()

# Sidebar - Upload section
st.sidebar.header("Marketing Data")

# Show loaded status
if st.session_state.google_ads_df is not None:
    st.sidebar.success(f"Google Ads: {len(st.session_state.google_ads_df)} campaigns")
if st.session_state.meta_ads_df is not None:
    st.sidebar.success(f"Meta Ads: {len(st.session_state.meta_ads_df)} campaigns")
if st.session_state.df1 is not None:
    st.sidebar.success(f"Booking data: {len(st.session_state.df1):,} rows")

# Manual upload option (in expander)
with st.sidebar.expander("Upload different files"):
    # Google Ads uploader (multiple files)
    uploaded_google_files = st.file_uploader(
        "Google Ads CSV",
        type=["csv"],
        help="Export from Google Ads campaign report. You can select multiple files.",
        key="google_ads_file",
        accept_multiple_files=True
    )

    if uploaded_google_files:
        dfs = []
        errors = []
        for uploaded_file in uploaded_google_files:
            df, error = parse_google_ads_csv(uploaded_file)
            if error:
                errors.append(f"{uploaded_file.name}: {error}")
            elif df is not None:
                dfs.append(df)

        if errors:
            for error in errors:
                st.error(f"Google Ads error: {error}")
        elif dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            st.session_state.google_ads_df = merged_df
            if len(dfs) == 1:
                st.success(f"Google Ads loaded: {len(merged_df)} campaigns")
            else:
                st.success(f"Google Ads merged {len(dfs)} files: {len(merged_df)} campaigns")

    # Meta Ads uploader (multiple files)
    uploaded_meta_files = st.file_uploader(
        "Meta Ads CSV",
        type=["csv"],
        help="Export from Meta Ads Manager. You can select multiple files.",
        key="meta_ads_file",
        accept_multiple_files=True
    )

    if uploaded_meta_files:
        dfs = []
        errors = []
        for uploaded_file in uploaded_meta_files:
            df, error = parse_meta_ads_csv(uploaded_file)
            if error:
                errors.append(f"{uploaded_file.name}: {error}")
            elif df is not None:
                dfs.append(df)

        if errors:
            for error in errors:
                st.error(f"Meta Ads error: {error}")
        elif dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            st.session_state.meta_ads_df = merged_df
            if len(dfs) == 1:
                st.success(f"Meta Ads loaded: {len(merged_df)} campaigns")
            else:
                st.success(f"Meta Ads merged {len(dfs)} files: {len(merged_df)} campaigns")

# Fill navigation container
if st.session_state.df2 is not None or st.session_state.google_ads_df is not None or st.session_state.meta_ads_df is not None:
    with nav_container:
        st.markdown("### Navigation")
        st.page_link("app.py", label="Booking Patterns", icon=":material/bar_chart:")
        st.page_link("pages/3_Customers.py", label="Recurring Customers", icon=":material/group:")
        st.page_link("pages/4_Revenue.py", label="Revenue & Value", icon=":material/payments:")
        st.page_link("pages/5_Promotions.py", label="Promotions", icon=":material/sell:")
        st.page_link("pages/6_Capacity.py", label="Capacity Analysis", icon=":material/analytics:")
        st.page_link("pages/7_Marketing.py", label="Marketing", icon=":material/campaign:")
        st.markdown("---")

# Main content
if st.session_state.google_ads_df is None and st.session_state.meta_ads_df is None:
    st.info("**No marketing data loaded.** Please upload Google Ads and/or Meta Ads CSV files using the sidebar.")
    st.markdown("""
    ### Getting Started
    1. Upload Google Ads CSV export
    2. Upload Meta Ads CSV export
    3. Tag campaigns with SEE-THINK-DO-CARE phases
    4. Analyze performance across the customer journey

    ### SEE-THINK-DO-CARE Framework
    - **SEE**: Awareness stage - reaching broad audiences (Display, Reach campaigns)
    - **THINK**: Consideration stage - engaging interested users (Non-branded, Clicks campaigns)
    - **DO**: Conversion stage - driving actions (Branded, Conversion campaigns)
    - **CARE**: Loyalty stage - retaining customers (Retargeting, Remarketing campaigns)
    """)
else:
    # Show loading message while processing
    loading_placeholder = st.empty()
    loading_placeholder.info("Loading marketing analysis...")

    # Combine data from both platforms
    dfs_to_combine = []

    if st.session_state.google_ads_df is not None:
        dfs_to_combine.append(st.session_state.google_ads_df)

    if st.session_state.meta_ads_df is not None:
        dfs_to_combine.append(st.session_state.meta_ads_df)

    combined_df = pd.concat(dfs_to_combine, ignore_index=True)

    # Extract date range from data
    data_min_date = None
    data_max_date = None

    # Try to get dates from Meta Ads data
    if st.session_state.meta_ads_df is not None:
        meta_df_dates = st.session_state.meta_ads_df
        if 'Reporting starts' in meta_df_dates.columns and 'Reporting ends' in meta_df_dates.columns:
            try:
                start_dates = pd.to_datetime(meta_df_dates['Reporting starts'], errors='coerce')
                end_dates = pd.to_datetime(meta_df_dates['Reporting ends'], errors='coerce')
                data_min_date = start_dates.min()
                data_max_date = end_dates.max()
            except:
                pass

    # Add date columns to combined_df for filtering
    if 'Reporting starts' in combined_df.columns:
        combined_df['report_start'] = pd.to_datetime(combined_df['Reporting starts'], errors='coerce')
    if 'Reporting ends' in combined_df.columns:
        combined_df['report_end'] = pd.to_datetime(combined_df['Reporting ends'], errors='coerce')

    # Filter out campaigns with 0 spend or invalid names
    if 'spend' in combined_df.columns:
        combined_df = combined_df[combined_df['spend'] > 0]
    if 'campaign_name' in combined_df.columns:
        combined_df = combined_df[combined_df['campaign_name'].notna()]
        combined_df = combined_df[combined_df['campaign_name'].astype(str).str.strip() != '']

    # Get locations from booking data
    available_locations = []
    if st.session_state.df2 is not None:
        if 'Location' in st.session_state.df2.columns:
            location_col = 'Location'
        elif 'Activity' in st.session_state.df2.columns:
            location_col = 'Activity'
        elif 'Tour' in st.session_state.df2.columns:
            location_col = 'Tour'
        else:
            location_col = None
        if location_col:
            available_locations = st.session_state.df2[location_col].unique().tolist()
            available_locations = [loc for loc in available_locations if loc in LOCATION_KEYWORDS]

    # Filter settings in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")

    show_all_campaigns = st.sidebar.checkbox(
        "Show all campaigns",
        value=True,
        help="Show all campaigns regardless of location"
    )

    # Filter campaigns by location (only if not showing all)
    if not show_all_campaigns and available_locations:
        filtered_campaigns = []
        for _, row in combined_df.iterrows():
            campaign_name = row.get('campaign_name', '')
            if pd.isna(campaign_name):
                continue
            match = campaign_matches_location(campaign_name, available_locations)
            if match:
                filtered_campaigns.append(row)

        if filtered_campaigns:
            combined_df = pd.DataFrame(filtered_campaigns)
            st.sidebar.info(f"Showing {len(combined_df)} campaigns matching your locations")
        else:
            st.warning("No campaigns match the locations in your booking data.")
            combined_df = pd.DataFrame()
    else:
        st.sidebar.info(f"Showing all {len(combined_df)} campaigns")

    if len(combined_df) > 0:
        # Check if campaign_name column exists
        if 'campaign_name' not in combined_df.columns:
            st.error("Marketing data is missing 'campaign_name' column. Please reboot the app in Streamlit Cloud (Manage app → Reboot) to reload data with correct column mapping.")
            st.write("**Available columns:**", list(combined_df.columns))
            st.stop()

        # Initialize STDC tags for new campaigns
        for campaign in combined_df['campaign_name'].unique():
            if campaign not in st.session_state.stdc_tags:
                st.session_state.stdc_tags[campaign] = suggest_stdc_phase(campaign)

        # Add STDC phase to dataframe
        combined_df['stdc_phase'] = combined_df['campaign_name'].map(st.session_state.stdc_tags)

        # Date range selector
        if pd.notna(data_min_date) and pd.notna(data_max_date):
            with date_range_container:
                date_col1, date_col2 = st.columns([2, 4])
                with date_col1:
                    date_range = st.date_input(
                        "Date Range",
                        value=(data_min_date.date(), data_max_date.date()),
                        min_value=data_min_date.date(),
                        max_value=data_max_date.date(),
                        help="Filter marketing data by reporting period",
                        key="mkt_date_range"
                    )

                # Apply date filter if dates are selected
                if len(date_range) == 2 and 'report_start' in combined_df.columns:
                    start_date, end_date = date_range
                    # Only filter rows that have date info (Meta Ads)
                    # Keep rows without dates (Google Ads) - they don't have reporting period columns
                    has_dates = combined_df['report_start'].notna() & combined_df['report_end'].notna()
                    date_filter = (
                        (combined_df['report_start'].dt.date <= end_date) &
                        (combined_df['report_end'].dt.date >= start_date)
                    )
                    # Keep rows without dates OR rows that pass the date filter
                    combined_df = combined_df[~has_dates | date_filter]

        # Clear loading message now that data is ready
        loading_placeholder.empty()

        # Key Metrics
        st.markdown("### Key Metrics")

        total_spend = combined_df['spend'].sum()
        total_conversions = combined_df['conversions'].sum()
        total_conv_value = combined_df['conversion_value'].sum() if 'conversion_value' in combined_df.columns else 0
        roas = (total_conv_value / total_spend * 100) if total_spend > 0 else 0
        cpa = (total_spend / total_conversions) if total_conversions > 0 else 0

        # Calculate platform distribution for all metrics
        google_df = combined_df[combined_df['Platform'] == 'Google Ads']
        meta_df = combined_df[combined_df['Platform'] == 'Meta Ads']

        google_spend = google_df['spend'].sum()
        meta_spend = meta_df['spend'].sum()
        google_conv = google_df['conversions'].sum()
        meta_conv = meta_df['conversions'].sum()
        google_conv_value = google_df['conversion_value'].sum() if 'conversion_value' in google_df.columns else 0
        meta_conv_value = meta_df['conversion_value'].sum() if 'conversion_value' in meta_df.columns else 0
        google_roas = (google_conv_value / google_spend * 100) if google_spend > 0 else 0
        meta_roas = (meta_conv_value / meta_spend * 100) if meta_spend > 0 else 0
        google_cpa = (google_spend / google_conv) if google_conv > 0 else 0
        meta_cpa = (meta_spend / meta_conv) if meta_conv > 0 else 0

        # Helper for platform split tooltip
        def platform_tooltip(g_val, m_val, fmt='currency'):
            if fmt == 'currency':
                g_str = f"€{g_val:,.0f}" if g_val > 0 else "-"
                m_str = f"€{m_val:,.0f}" if m_val > 0 else "-"
            elif fmt == 'percent':
                g_str = f"{g_val:.1f}%" if g_val > 0 else "-"
                m_str = f"{m_val:.1f}%" if m_val > 0 else "-"
            else:
                g_str = f"{g_val:,.0f}" if g_val > 0 else "-"
                m_str = f"{m_val:,.0f}" if m_val > 0 else "-"
            return f"G Ads: {g_str} · M Ads: {m_str}"

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Spend", f"€{total_spend:,.0f}", help=platform_tooltip(google_spend, meta_spend))
        with col2:
            st.metric("Conversions", f"{total_conversions:,.0f}", help=platform_tooltip(google_conv, meta_conv, 'number'))
        with col3:
            st.metric("Conv. Value", f"€{total_conv_value:,.0f}", help=platform_tooltip(google_conv_value, meta_conv_value))
        with col4:
            st.metric("ROAS", f"{roas:,.0f}%", help=platform_tooltip(google_roas, meta_roas, 'percent'))
        with col5:
            st.metric("Avg CPA", f"€{cpa:,.0f}", help=platform_tooltip(google_cpa, meta_cpa))

        # ===== VISUALIZATION B: Stage KPI Cards =====
        st.markdown("---")
        st.markdown("### STDC Performance")

        # Show loading spinner for STDC section
        stdc_loading = st.empty()
        with stdc_loading:
            st.markdown("""
            <div style="display: flex; align-items: center; padding: 1rem; background-color: #e3f2fd; border-radius: 8px; margin: 1rem 0;">
                <div style="width: 24px; height: 24px; border: 3px solid #1976d2; border-top-color: transparent; border-radius: 50%; animation: spin 1s linear infinite; margin-right: 12px;"></div>
                <span style="color: #1976d2; font-weight: 500;">Calculating STDC metrics...</span>
            </div>
            <style>
                @keyframes spin {
                    to { transform: rotate(360deg); }
                }
            </style>
            """, unsafe_allow_html=True)

        # Calculate metrics for each STDC phase with platform breakdown
        def get_phase_metrics(phase_name):
            phase_data = combined_df[combined_df['stdc_phase'] == phase_name]
            spend = phase_data['spend'].sum()
            impressions = phase_data['impressions'].sum() if 'impressions' in phase_data.columns else 0
            reach = phase_data['reach'].sum() if 'reach' in phase_data.columns else 0
            total_reach = impressions + reach
            clicks = phase_data['clicks'].sum() if 'clicks' in phase_data.columns else 0
            conversions = phase_data['conversions'].sum() if 'conversions' in phase_data.columns else 0

            # Calculate derived metrics
            cpm = (spend / total_reach * 1000) if total_reach > 0 else 0
            ctr = (clicks / total_reach * 100) if total_reach > 0 else 0
            cpa = (spend / conversions) if conversions > 0 else 0
            conv_rate = (conversions / clicks * 100) if clicks > 0 else 0

            # Platform breakdown
            google_data = phase_data[phase_data['Platform'] == 'Google Ads']
            meta_data = phase_data[phase_data['Platform'] == 'Meta Ads']

            g_spend = google_data['spend'].sum()
            m_spend = meta_data['spend'].sum()
            g_impr = google_data['impressions'].sum() if 'impressions' in google_data.columns else 0
            m_reach = meta_data['reach'].sum() if 'reach' in meta_data.columns else 0
            g_clicks = google_data['clicks'].sum() if 'clicks' in google_data.columns else 0
            m_clicks = meta_data['clicks'].sum() if 'clicks' in meta_data.columns else 0
            g_conv = google_data['conversions'].sum() if 'conversions' in google_data.columns else 0
            m_conv = meta_data['conversions'].sum() if 'conversions' in meta_data.columns else 0

            # Platform-specific derived metrics
            g_cpm = (g_spend / g_impr * 1000) if g_impr > 0 else 0
            m_cpm = (m_spend / m_reach * 1000) if m_reach > 0 else 0
            g_ctr = (g_clicks / g_impr * 100) if g_impr > 0 else 0
            m_ctr = (m_clicks / m_reach * 100) if m_reach > 0 else 0
            g_cpa = (g_spend / g_conv) if g_conv > 0 else 0
            m_cpa = (m_spend / m_conv) if m_conv > 0 else 0
            g_conv_rate = (g_conv / g_clicks * 100) if g_clicks > 0 else 0
            m_conv_rate = (m_conv / m_clicks * 100) if m_clicks > 0 else 0

            return {
                'spend': spend, 'reach': total_reach, 'clicks': clicks,
                'conversions': conversions, 'cpm': cpm, 'ctr': ctr,
                'cpa': cpa, 'conv_rate': conv_rate,
                # Platform breakdowns
                'google': {
                    'spend': g_spend, 'reach': g_impr, 'clicks': g_clicks,
                    'conversions': g_conv, 'cpm': g_cpm, 'ctr': g_ctr,
                    'cpa': g_cpa, 'conv_rate': g_conv_rate
                },
                'meta': {
                    'spend': m_spend, 'reach': m_reach, 'clicks': m_clicks,
                    'conversions': m_conv, 'cpm': m_cpm, 'ctr': m_ctr,
                    'cpa': m_cpa, 'conv_rate': m_conv_rate
                }
            }

        def format_platform_tooltip(google_val, meta_val, fmt='number'):
            """Format platform split as tooltip text."""
            if fmt == 'currency':
                g_str = f"€{google_val:,.0f}" if google_val > 0 else "-"
                m_str = f"€{meta_val:,.0f}" if meta_val > 0 else "-"
            elif fmt == 'percent':
                g_str = f"{google_val:.1f}%" if google_val > 0 else "-"
                m_str = f"{meta_val:.1f}%" if meta_val > 0 else "-"
            else:
                g_str = f"{google_val:,.0f}" if google_val > 0 else "-"
                m_str = f"{meta_val:,.0f}" if meta_val > 0 else "-"
            return f"G Ads: {g_str} · M Ads: {m_str}"

        see_metrics = get_phase_metrics('SEE')
        think_metrics = get_phase_metrics('THINK')
        do_metrics = get_phase_metrics('DO')
        care_metrics = get_phase_metrics('CARE')

        # Calculate rebooking metrics and CLV from booking data if available
        rebookings = 0
        rebook_rate = 0
        total_customers = 0
        clv = 0
        if st.session_state.df1 is not None:
            df1 = st.session_state.df1
            email_col = 'Email address' if 'Email address' in df1.columns else None
            revenue_col = 'Total gross' if 'Total gross' in df1.columns else None
            date_col = 'Created' if 'Created' in df1.columns else None

            if email_col:
                customer_bookings = df1.groupby(email_col).size()
                total_customers = len(customer_bookings)
                repeat_customers = (customer_bookings > 1).sum()
                rebookings = customer_bookings[customer_bookings > 1].sum()
                rebook_rate = (repeat_customers / total_customers * 100) if total_customers > 0 else 0

                # Calculate CLV if revenue and date columns exist
                if revenue_col and date_col:
                    # Filter valid data
                    clv_data = df1[[email_col, revenue_col, date_col]].copy()
                    clv_data.columns = ['email', 'revenue', 'booking_date']
                    clv_data['revenue'] = pd.to_numeric(clv_data['revenue'], errors='coerce').fillna(0)
                    clv_data['booking_date'] = pd.to_datetime(clv_data['booking_date'], errors='coerce')
                    clv_data = clv_data[clv_data['booking_date'].notna() & clv_data['email'].notna()]

                    if len(clv_data) > 0:
                        # Step 1: Average Order Value
                        aov = clv_data['revenue'].sum() / len(clv_data) if len(clv_data) > 0 else 0

                        # Step 2: Retention Rate (cohort-based)
                        min_date = clv_data['booking_date'].min()
                        max_date = clv_data['booking_date'].max()

                        first_month_start = min_date.replace(day=1)
                        if min_date.day > 1:
                            first_month_start = first_month_start + pd.DateOffset(months=1)
                        first_month_end = first_month_start + pd.DateOffset(months=1) - pd.Timedelta(days=1)
                        two_months_later = first_month_start + pd.DateOffset(months=3) - pd.Timedelta(days=1)

                        # Find customers whose first booking was in cohort month
                        customer_first = clv_data.groupby('email')['booking_date'].min().reset_index()
                        customer_first.columns = ['email', 'first_booking']
                        cohort_customers = customer_first[
                            (customer_first['first_booking'] >= first_month_start) &
                            (customer_first['first_booking'] <= first_month_end)
                        ]['email'].tolist()

                        cohort_size = len(cohort_customers)
                        if cohort_size > 0:
                            returning_customers = 0
                            for email in cohort_customers:
                                cust_bookings = clv_data[clv_data['email'] == email]['booking_date'].sort_values()
                                if len(cust_bookings) > 1:
                                    second_booking = cust_bookings.iloc[1]
                                    if second_booking <= two_months_later:
                                        returning_customers += 1
                            retention_rate = returning_customers / cohort_size
                        else:
                            retention_rate = 0.3  # Default fallback

                        # Step 3: Churn Rate
                        churn_rate = 1 - retention_rate

                        # Step 4: CLV = AOV × (1 + retention_rate / churn_rate)
                        if churn_rate > 0:
                            clv = aov * (1 + retention_rate / churn_rate)
                        else:
                            clv = aov * 10  # Cap at 10x AOV if no churn

        # Clear loading placeholder - metrics are ready
        stdc_loading.empty()

        # Stage KPI Cards with colored backgrounds
        st.markdown("""
        <style>
        .stdc-card {
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 0.5rem;
        }
        .stdc-see { background: linear-gradient(135deg, #3498db20 0%, #3498db10 100%); border-left: 4px solid #3498db; }
        .stdc-think { background: linear-gradient(135deg, #f39c1220 0%, #f39c1210 100%); border-left: 4px solid #f39c12; }
        .stdc-do { background: linear-gradient(135deg, #27ae6020 0%, #27ae6010 100%); border-left: 4px solid #27ae60; }
        .stdc-care { background: linear-gradient(135deg, #9b59b620 0%, #9b59b610 100%); border-left: 4px solid #9b59b6; }
        </style>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="stdc-card stdc-see">', unsafe_allow_html=True)
            st.markdown("**SEE** - Awareness")
            st.metric("CPM", f"€{see_metrics['cpm']:.2f}", help=format_platform_tooltip(see_metrics['google']['cpm'], see_metrics['meta']['cpm'], 'currency'))
            st.metric("Reach", f"{see_metrics['reach']:,.0f}", help=format_platform_tooltip(see_metrics['google']['reach'], see_metrics['meta']['reach']))
            st.metric("Spend", f"€{see_metrics['spend']:,.0f}", help=format_platform_tooltip(see_metrics['google']['spend'], see_metrics['meta']['spend'], 'currency'))
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="stdc-card stdc-think">', unsafe_allow_html=True)
            st.markdown("**THINK** - Consideration")
            st.metric("CTR", f"{think_metrics['ctr']:.2f}%", help=format_platform_tooltip(think_metrics['google']['ctr'], think_metrics['meta']['ctr'], 'percent'))
            st.metric("Clicks", f"{think_metrics['clicks']:,.0f}", help=format_platform_tooltip(think_metrics['google']['clicks'], think_metrics['meta']['clicks']))
            st.metric("Spend", f"€{think_metrics['spend']:,.0f}", help=format_platform_tooltip(think_metrics['google']['spend'], think_metrics['meta']['spend'], 'currency'))
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="stdc-card stdc-do">', unsafe_allow_html=True)
            st.markdown("**DO** - Conversion")
            st.metric("CPA", f"€{do_metrics['cpa']:.2f}" if do_metrics['cpa'] > 0 else "N/A", help=format_platform_tooltip(do_metrics['google']['cpa'], do_metrics['meta']['cpa'], 'currency'))
            st.metric("Conversions", f"{do_metrics['conversions']:,.0f} ({do_metrics['conv_rate']:.1f}%)", help=format_platform_tooltip(do_metrics['google']['conversions'], do_metrics['meta']['conversions']))
            st.metric("Spend", f"€{do_metrics['spend']:,.0f}", help=format_platform_tooltip(do_metrics['google']['spend'], do_metrics['meta']['spend'], 'currency'))
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="stdc-card stdc-care">', unsafe_allow_html=True)
            st.markdown("**CARE** - Loyalty")
            st.metric("Rebook Rate", f"{rebook_rate:.1f}%" if total_customers > 0 else "N/A", help="From booking data")
            st.metric("CLV", f"€{clv:,.0f}" if clv > 0 else "N/A", help="Customer Lifetime Value from Revenue & Value analysis")
            st.metric("Spend", f"€{care_metrics['spend']:,.0f}", help=format_platform_tooltip(care_metrics['google']['spend'], care_metrics['meta']['spend'], 'currency'))
            st.markdown('</div>', unsafe_allow_html=True)

        # Show untagged campaigns count if any
        untagged_count = len(combined_df[combined_df['stdc_phase'] == 'Untagged'])
        if untagged_count > 0:
            st.caption(f"Note: {untagged_count} campaigns are untagged. Tag them in the Campaign Performance table below.")

        # Platform comparison
        st.markdown("---")
        st.markdown("### Platform Comparison")

        # Better color contrast: Google = Blue, Meta = Teal
        platform_colors = {'Google Ads': '#4285f4', 'Meta Ads': '#00C4B4'}

        # Calculate platform metrics with more detail
        platform_metrics = combined_df.groupby('Platform').agg({
            'spend': 'sum',
            'conversions': 'sum',
            'clicks': 'sum',
            'conversion_value': 'sum'
        }).reset_index()

        # Add reach/impressions
        google_reach = combined_df[combined_df['Platform'] == 'Google Ads']['impressions'].sum() if 'impressions' in combined_df.columns else 0
        meta_reach = combined_df[combined_df['Platform'] == 'Meta Ads']['reach'].sum() if 'reach' in combined_df.columns else 0
        platform_metrics['reach'] = [google_reach, meta_reach] if len(platform_metrics) == 2 else [google_reach + meta_reach]

        # Calculate CPA per platform
        platform_metrics['cpa'] = platform_metrics.apply(
            lambda x: x['spend'] / x['conversions'] if x['conversions'] > 0 else 0, axis=1
        )

        # Row 1: Spend, Conversions, Conv Value
        col1, col2, col3 = st.columns(3)

        with col1:
            fig_plat_spend = px.pie(
                platform_metrics,
                values='spend',
                names='Platform',
                title='Spend',
                color='Platform',
                color_discrete_map=platform_colors
            )
            fig_plat_spend.update_layout(height=350, showlegend=True)
            fig_plat_spend.update_traces(textinfo='percent+value', texttemplate='%{percent:.1%}<br>€%{value:,.0f}')
            st.plotly_chart(fig_plat_spend, use_container_width=True)

        with col2:
            fig_plat_conv = px.pie(
                platform_metrics,
                values='conversions',
                names='Platform',
                title='Conversions',
                color='Platform',
                color_discrete_map=platform_colors
            )
            fig_plat_conv.update_layout(height=350, showlegend=True)
            fig_plat_conv.update_traces(textinfo='percent+value', texttemplate='%{percent:.1%}<br>%{value:,.0f}')
            st.plotly_chart(fig_plat_conv, use_container_width=True)

        with col3:
            fig_plat_value = px.pie(
                platform_metrics,
                values='conversion_value',
                names='Platform',
                title='Conversion Value',
                color='Platform',
                color_discrete_map=platform_colors
            )
            fig_plat_value.update_layout(height=350, showlegend=True)
            fig_plat_value.update_traces(textinfo='percent+value', texttemplate='%{percent:.1%}<br>€%{value:,.0f}')
            st.plotly_chart(fig_plat_value, use_container_width=True)

        # Row 2: Clicks, Reach, CPA comparison
        col1, col2, col3 = st.columns(3)

        with col1:
            fig_plat_clicks = px.pie(
                platform_metrics,
                values='clicks',
                names='Platform',
                title='Clicks',
                color='Platform',
                color_discrete_map=platform_colors
            )
            fig_plat_clicks.update_layout(height=350, showlegend=True)
            fig_plat_clicks.update_traces(textinfo='percent+value', texttemplate='%{percent:.1%}<br>%{value:,.0f}')
            st.plotly_chart(fig_plat_clicks, use_container_width=True)

        with col2:
            fig_plat_reach = px.pie(
                platform_metrics,
                values='reach',
                names='Platform',
                title='Reach / Impressions',
                color='Platform',
                color_discrete_map=platform_colors
            )
            fig_plat_reach.update_layout(height=350, showlegend=True)
            fig_plat_reach.update_traces(textinfo='percent+value', texttemplate='%{percent:.1%}<br>%{value:,.0f}')
            st.plotly_chart(fig_plat_reach, use_container_width=True)

        with col3:
            # CTR comparison as pie chart
            # Calculate CTR per platform
            google_ctr = (google_df['clicks'].sum() / google_df['impressions'].sum() * 100) if 'impressions' in google_df.columns and google_df['impressions'].sum() > 0 else 0
            meta_ctr = (meta_df['clicks'].sum() / meta_df['reach'].sum() * 100) if 'reach' in meta_df.columns and meta_df['reach'].sum() > 0 else 0

            ctr_data = pd.DataFrame({
                'Platform': ['Google Ads', 'Meta Ads'],
                'ctr': [google_ctr, meta_ctr]
            })

            fig_plat_ctr = px.pie(
                ctr_data,
                values='ctr',
                names='Platform',
                title='CTR Distribution',
                color='Platform',
                color_discrete_map=platform_colors
            )
            fig_plat_ctr.update_layout(height=350, showlegend=True)
            fig_plat_ctr.update_traces(textinfo='percent+value', texttemplate='%{percent:.1%}<br>%{value:.2f}%')
            st.plotly_chart(fig_plat_ctr, use_container_width=True)

        # Campaign Performance Table
        st.markdown("---")
        st.markdown("### Campaign Performance")

        # Prepare display table with more metrics
        display_df = combined_df.copy()

        # Filter out campaigns with invalid names (containing "--" or empty)
        display_df = display_df[~display_df['campaign_name'].astype(str).str.contains('^-+$|^ *$', regex=True, na=True)]
        display_df = display_df[display_df['campaign_name'].notna()]
        display_df = display_df[display_df['campaign_name'].astype(str).str.strip() != '']

        # Calculate additional metrics
        display_df['cpc'] = display_df.apply(
            lambda x: x['spend'] / x['clicks'] if x['clicks'] > 0 else 0, axis=1
        )
        display_df['cpa'] = display_df.apply(
            lambda x: x['spend'] / x['conversions'] if x['conversions'] > 0 else 0, axis=1
        )
        display_df['conv_rate'] = display_df.apply(
            lambda x: (x['conversions'] / x['clicks'] * 100) if x['clicks'] > 0 else 0, axis=1
        )

        # Select and rename columns
        display_cols = ['campaign_name', 'Platform', 'stdc_phase', 'spend', 'clicks', 'conversions', 'conversion_value', 'cpc', 'cpa', 'conv_rate']
        display_cols = [c for c in display_cols if c in display_df.columns]

        display_df = display_df[display_cols].copy()
        display_df = display_df.rename(columns={
            'campaign_name': 'Campaign',
            'stdc_phase': 'STDC',
            'spend': 'Spend',
            'clicks': 'Clicks',
            'conversions': 'Conv',
            'conversion_value': 'Value',
            'cpc': 'CPC',
            'cpa': 'CPA',
            'conv_rate': 'Conv %'
        })

        # Round all numeric columns to whole numbers
        numeric_cols = ['Spend', 'Clicks', 'Conv', 'Value', 'CPC', 'CPA']
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(0).astype(int)

        if 'Conv %' in display_df.columns:
            display_df['Conv %'] = display_df['Conv %'].round(0).astype(int)

        # Sort by Platform (Google Ads first) then by Spend descending
        display_df = display_df.sort_values(['Platform', 'Spend'], ascending=[True, False])

        # Color coding for STDC phase
        def style_stdc(val):
            if val == 'SEE':
                return 'background-color: #dbeafe; color: #1e40af'
            elif val == 'THINK':
                return 'background-color: #fef3c7; color: #92400e'
            elif val == 'DO':
                return 'background-color: #dcfce7; color: #166534'
            elif val == 'CARE':
                return 'background-color: #fce7f3; color: #9d174d'
            return 'background-color: #f3f4f6; color: #4b5563'

        styled_df = display_df.style.applymap(style_stdc, subset=['STDC'])

        campaign_config = {
            'Campaign': st.column_config.TextColumn('Campaign', help='Campaign name from ad platform'),
            'Platform': st.column_config.TextColumn('Platform', help='Google Ads or Meta Ads'),
            'STDC': st.column_config.TextColumn('STDC', help='SEE-THINK-DO-CARE funnel phase'),
            'Spend': st.column_config.NumberColumn('Spend', help='Total ad spend for this campaign'),
            'Clicks': st.column_config.NumberColumn('Clicks', help='Link clicks on ads'),
            'Conv': st.column_config.NumberColumn('Conv', help='Purchases/conversions tracked'),
            'Value': st.column_config.NumberColumn('Value', help='Conversion value reported'),
            'CPC': st.column_config.NumberColumn('CPC', help='Cost per Click = Spend / Clicks'),
            'CPA': st.column_config.NumberColumn('CPA', help='Cost per Acquisition = Spend / Conversions'),
            'Conv %': st.column_config.NumberColumn('Conv %', help='Conversion Rate = Conversions / Clicks'),
        }
        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=400, column_config=campaign_config)

        # STDC Tag Configuration
        with st.expander("Configure STDC Tags", expanded=False):
            st.markdown("""
            Assign each campaign to a SEE-THINK-DO-CARE phase.
            Default suggestions are based on campaign name keywords.
            """)

            campaigns = sorted(combined_df['campaign_name'].unique().tolist())

            for campaign in campaigns:
                current_tag = st.session_state.stdc_tags.get(campaign, 'Untagged')

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(campaign[:60] + '...' if len(campaign) > 60 else campaign)
                with col2:
                    new_tag = st.selectbox(
                        "Phase",
                        options=['SEE', 'THINK', 'DO', 'CARE', 'Untagged'],
                        index=['SEE', 'THINK', 'DO', 'CARE', 'Untagged'].index(current_tag),
                        key=f"stdc_{campaign}",
                        label_visibility="collapsed"
                    )
                    st.session_state.stdc_tags[campaign] = new_tag

            if st.button("Reset All to Suggestions", key="reset_stdc"):
                for campaign in campaigns:
                    st.session_state.stdc_tags[campaign] = suggest_stdc_phase(campaign)
                st.rerun()

        # Location Performance - analyze all campaigns by location
        st.markdown("---")
        st.markdown("### Performance by Location")

        # Map campaigns to locations with all metrics
        location_data = []
        for _, row in combined_df.iterrows():
            campaign_name = row.get('campaign_name', '')
            match = campaign_matches_location(campaign_name)
            if match and match != 'all':
                # Get reach (impressions for Google, reach for Meta)
                reach = row.get('impressions', 0) if row.get('Platform') == 'Google Ads' else row.get('reach', 0)
                location_data.append({
                    'Location': match,
                    'Spend': row.get('spend', 0),
                    'Reach': reach,
                    'Clicks': row.get('clicks', 0),
                    'Conversions': row.get('conversions', 0),
                    'Value': row.get('conversion_value', 0)
                })

        if location_data:
            loc_df = pd.DataFrame(location_data)

            # Aggregate by location (excluding "All Locations")
            loc_summary = loc_df.groupby('Location').agg({
                'Spend': 'sum',
                'Reach': 'sum',
                'Clicks': 'sum',
                'Conversions': 'sum',
                'Value': 'sum'
            }).reset_index()

            # Calculate efficiency metrics
            loc_summary['CTR'] = loc_summary.apply(
                lambda x: (x['Clicks'] / x['Reach'] * 100) if x['Reach'] > 0 else 0, axis=1
            )
            loc_summary['CPA'] = loc_summary.apply(
                lambda x: x['Spend'] / x['Conversions'] if x['Conversions'] > 0 else 0, axis=1
            )
            loc_summary['ROAS'] = loc_summary.apply(
                lambda x: x['Value'] / x['Spend'] if x['Spend'] > 0 else 0, axis=1
            )

            # Calculate totals for marketing
            total_spend = loc_summary['Spend'].sum()
            total_reach = loc_summary['Reach'].sum()
            total_clicks = loc_summary['Clicks'].sum()
            total_conv = loc_summary['Conversions'].sum()
            total_value = loc_summary['Value'].sum()
            avg_ctr = (total_clicks / total_reach * 100) if total_reach > 0 else 0
            avg_cpa = total_spend / total_conv if total_conv > 0 else 0
            avg_roas = total_value / total_spend if total_spend > 0 else 0

            # Get revenue data from booking files
            revenue_data = {}
            if st.session_state.df1 is not None:
                df1 = st.session_state.df1
                location_col = 'Location' if 'Location' in df1.columns else ('Tour' if 'Tour' in df1.columns else ('Activity' if 'Activity' in df1.columns else None))
                revenue_col = 'Total gross' if 'Total gross' in df1.columns else None
                if location_col:
                    revenue_data = get_revenue_per_location(df1, location_col, revenue_col)

            # ===== TABBED INTERFACE =====
            tab1, tab2, tab3 = st.tabs(["Marketing", "Revenue & Efficiency", "Capacity"])

            # ===== TAB 1: MARKETING =====
            with tab1:
                st.markdown("#### Marketing Performance Table")

                # Prepare display table
                table_df = loc_summary.copy()
                table_df = table_df.sort_values('CPA', ascending=True)

                table_display = table_df.copy()
                table_display['Spend'] = table_display['Spend'].apply(lambda x: f"€{x:,.0f}")
                table_display['Reach'] = table_display['Reach'].apply(lambda x: f"{x/1000:,.0f}K" if x >= 1000 else f"{x:,.0f}")
                table_display['Clicks'] = table_display['Clicks'].apply(lambda x: f"{x:,.0f}")
                table_display['Conv'] = table_display['Conversions'].apply(lambda x: f"{x:,.0f}")
                table_display['Value'] = table_display['Value'].apply(lambda x: f"€{x:,.0f}" if x > 0 else "-")
                table_display['CTR'] = table_display['CTR'].apply(lambda x: f"{x:.2f}%")
                table_display['CPA'] = table_display['CPA'].apply(lambda x: f"€{x:,.0f}" if x > 0 else "-")
                table_display['ROAS'] = table_display['ROAS'].apply(lambda x: f"{x:.1f}x" if x > 0 else "-")

                # Select and order columns
                table_display = table_display[['Location', 'Spend', 'Reach', 'Clicks', 'Conv', 'Value', 'CTR', 'CPA', 'ROAS']]

                total_row = pd.DataFrame({
                    'Location': ['TOTAL'],
                    'Spend': [f"€{total_spend:,.0f}"],
                    'Reach': [f"{total_reach/1000:,.0f}K" if total_reach >= 1000 else f"{total_reach:,.0f}"],
                    'Clicks': [f"{total_clicks:,.0f}"],
                    'Conv': [f"{total_conv:,.0f}"],
                    'Value': [f"€{total_value:,.0f}" if total_value > 0 else "-"],
                    'CTR': [f"{avg_ctr:.2f}%"],
                    'CPA': [f"€{avg_cpa:,.0f}" if avg_cpa > 0 else "-"],
                    'ROAS': [f"{avg_roas:.1f}x" if avg_roas > 0 else "-"]
                })
                table_display = pd.concat([table_display, total_row], ignore_index=True)

                # Column config with help tooltips
                marketing_column_config = {
                    'Location': st.column_config.TextColumn('Location'),
                    'Spend': st.column_config.TextColumn('Spend', help='Total ad spend per location'),
                    'Reach': st.column_config.TextColumn('Reach', help='People who saw ads (impressions for Google, reach for Meta)'),
                    'Clicks': st.column_config.TextColumn('Clicks', help='Link clicks on ads'),
                    'Conv': st.column_config.TextColumn('Conv', help='Purchases/conversions tracked by ad platforms'),
                    'Value': st.column_config.TextColumn('Value', help='Conversion value reported by ad platforms'),
                    'CTR': st.column_config.TextColumn('CTR', help='Click-through rate = Clicks / Reach'),
                    'CPA': st.column_config.TextColumn('CPA', help='Cost per Acquisition = Spend / Conversions'),
                    'ROAS': st.column_config.TextColumn('ROAS', help='Return on Ad Spend = Value / Spend'),
                }

                # Style TOTAL row
                def highlight_total(row):
                    if row['Location'] == 'TOTAL':
                        return ['font-weight: bold; background-color: #f3f4f6'] * len(row)
                    return [''] * len(row)

                styled_table = table_display.style.apply(highlight_total, axis=1)
                st.dataframe(styled_table, use_container_width=True, hide_index=True, column_config=marketing_column_config)

                # CPA Comparison Chart
                st.markdown("#### Cost per Conversion by Location")
                cpa_df = loc_summary[loc_summary['Conversions'] > 0].copy()
                cpa_df = cpa_df.sort_values('CPA', ascending=True)

                if len(cpa_df) > 0:
                    min_cpa = cpa_df['CPA'].min()
                    max_cpa = cpa_df['CPA'].max()

                    def get_cpa_color(cpa):
                        if max_cpa == min_cpa:
                            return '#22c55e'
                        ratio = (cpa - min_cpa) / (max_cpa - min_cpa)
                        if ratio < 0.5:
                            r = int(34 + (250 - 34) * (ratio * 2))
                            g = int(197 - (197 - 204) * (ratio * 2))
                            b = int(94 - 94 * (ratio * 2))
                        else:
                            r = int(250 - (250 - 239) * ((ratio - 0.5) * 2))
                            g = int(204 - (204 - 68) * ((ratio - 0.5) * 2))
                            b = int(0 + 68 * ((ratio - 0.5) * 2))
                        return f'rgb({r},{g},{b})'

                    cpa_df['color'] = cpa_df['CPA'].apply(get_cpa_color)
                    cpa_df['label'] = cpa_df.apply(
                        lambda x: f"€{x['CPA']:,.0f} (Best)" if x['CPA'] == min_cpa else f"€{x['CPA']:,.0f}",
                        axis=1
                    )

                    fig_cpa = go.Figure()
                    fig_cpa.add_trace(go.Bar(
                        y=cpa_df['Location'],
                        x=cpa_df['CPA'],
                        orientation='h',
                        marker_color=cpa_df['color'],
                        text=cpa_df['label'],
                        textposition='outside',
                        hovertemplate='%{y}: €%{x:,.2f}<extra></extra>'
                    ))
                    fig_cpa.add_vline(x=avg_cpa, line_dash="dash", line_color="#666",
                                      annotation_text=f"Avg: €{avg_cpa:,.0f}", annotation_position="top")
                    fig_cpa.update_layout(
                        height=max(300, len(cpa_df) * 50),
                        margin=dict(l=20, r=120, t=50, b=40),
                        xaxis_title='CPA (€)',
                        yaxis=dict(categoryorder='total ascending'),
                        showlegend=False
                    )
                    st.plotly_chart(fig_cpa, use_container_width=True)

            # ===== TAB 2: REVENUE & EFFICIENCY =====
            with tab2:
                # Get revenue data with Marine split into Matsu and Bjørk
                revenue_data_split = {}
                if st.session_state.df1 is not None:
                    revenue_data_split = get_revenue_per_location(df1, location_col, revenue_col, split_marine=True)

                if revenue_data_split:
                    st.markdown("#### Revenue & Marketing Efficiency")
                    st.caption("Showing correlation between marketing spend and actual revenue (not causation)")

                    # Get Amsterdam Marine ad spend/conv for proportional split
                    marine_spend = 0
                    marine_conv = 0
                    if 'Amsterdam Marine' in loc_summary['Location'].values:
                        marine_row = loc_summary[loc_summary['Location'] == 'Amsterdam Marine']
                        marine_spend = marine_row['Spend'].values[0]
                        marine_conv = marine_row['Conversions'].values[0]

                    # Calculate total revenue for Marine locations (for proportional split)
                    marine_total_revenue = 0
                    for loc, data in revenue_data_split.items():
                        if data.get('parent_marketing_loc') == 'Amsterdam Marine':
                            marine_total_revenue += data['revenue']

                    # Build combined table
                    efficiency_data = []
                    added_locations = set()

                    # First add non-Marine locations from loc_summary (marketing data)
                    for loc in loc_summary['Location'].tolist():
                        if loc == 'Amsterdam Marine':
                            continue  # Skip - will add Matsu and Bjørk separately
                        row_data = {
                            'Location': loc,
                            'Ad Spend': loc_summary[loc_summary['Location'] == loc]['Spend'].values[0],
                            'Ad Conv': loc_summary[loc_summary['Location'] == loc]['Conversions'].values[0],
                        }
                        if loc in revenue_data_split:
                            row_data['Bookings'] = revenue_data_split[loc]['bookings']
                            row_data['Revenue'] = revenue_data_split[loc]['revenue']
                        else:
                            row_data['Bookings'] = 0
                            row_data['Revenue'] = 0
                        efficiency_data.append(row_data)
                        added_locations.add(loc)

                    # Add Matsu and Bjørk with proportional ad spend split
                    for loc, data in revenue_data_split.items():
                        if data.get('parent_marketing_loc') == 'Amsterdam Marine':
                            # Split ad spend proportionally by revenue
                            if marine_total_revenue > 0:
                                revenue_share = data['revenue'] / marine_total_revenue
                            else:
                                revenue_share = 0.5  # Default to 50/50 if no revenue
                            row_data = {
                                'Location': loc,
                                'Ad Spend': marine_spend * revenue_share,
                                'Ad Conv': marine_conv * revenue_share,
                                'Bookings': data['bookings'],
                                'Revenue': data['revenue'],
                            }
                            efficiency_data.append(row_data)
                            added_locations.add(loc)

                    # Add locations with revenue data but no marketing data
                    for loc, data in revenue_data_split.items():
                        if loc not in added_locations and not data.get('parent_marketing_loc'):
                            row_data = {
                                'Location': loc,
                                'Ad Spend': 0,
                                'Ad Conv': 0,
                                'Bookings': data['bookings'],
                                'Revenue': data['revenue'],
                            }
                            efficiency_data.append(row_data)

                    eff_df = pd.DataFrame(efficiency_data)

                    # Calculate efficiency metrics
                    eff_df['Marketing %'] = eff_df.apply(
                        lambda x: (x['Ad Spend'] / x['Revenue'] * 100) if x['Revenue'] > 0 else 0, axis=1
                    )
                    eff_df['Ad Share'] = eff_df.apply(
                        lambda x: (x['Ad Conv'] / x['Bookings'] * 100) if x['Bookings'] > 0 else 0, axis=1
                    )
                    eff_df['Revenue/€ Spent'] = eff_df.apply(
                        lambda x: x['Revenue'] / x['Ad Spend'] if x['Ad Spend'] > 0 else 0, axis=1
                    )
                    eff_df['Blended CAC'] = eff_df.apply(
                        lambda x: x['Ad Spend'] / x['Bookings'] if x['Bookings'] > 0 else 0, axis=1
                    )

                    # Sort by Revenue/€ Spent (efficiency)
                    eff_df = eff_df.sort_values('Revenue/€ Spent', ascending=False)

                    # Calculate totals
                    total_ad_spend = eff_df['Ad Spend'].sum()
                    total_ad_conv = eff_df['Ad Conv'].sum()
                    total_bookings = eff_df['Bookings'].sum()
                    total_revenue = eff_df['Revenue'].sum()
                    total_marketing_pct = (total_ad_spend / total_revenue * 100) if total_revenue > 0 else 0
                    total_ad_share = (total_ad_conv / total_bookings * 100) if total_bookings > 0 else 0
                    total_rev_per_spend = total_revenue / total_ad_spend if total_ad_spend > 0 else 0
                    total_blended_cac = total_ad_spend / total_bookings if total_bookings > 0 else 0

                    # Format for display
                    eff_display = eff_df.copy()
                    eff_display['Ad Spend'] = eff_display['Ad Spend'].apply(lambda x: f"€{x:,.0f}")
                    eff_display['Ad Conv'] = eff_display['Ad Conv'].apply(lambda x: f"{x:,.0f}")
                    eff_display['Bookings'] = eff_display['Bookings'].apply(lambda x: f"{x:,.0f}")
                    eff_display['Revenue'] = eff_display['Revenue'].apply(lambda x: f"€{x:,.0f}")
                    eff_display['Marketing %'] = eff_display['Marketing %'].apply(lambda x: f"{x:.1f}%")
                    eff_display['Ad Share'] = eff_display['Ad Share'].apply(lambda x: f"{x:.1f}%")
                    eff_display['Rev/€ Spent'] = eff_display['Revenue/€ Spent'].apply(lambda x: f"€{x:.1f}")
                    eff_display['Blended CAC'] = eff_display['Blended CAC'].apply(lambda x: f"€{x:.1f}")

                    # Select columns
                    eff_display = eff_display[['Location', 'Ad Spend', 'Ad Conv', 'Bookings', 'Revenue', 'Marketing %', 'Ad Share', 'Rev/€ Spent', 'Blended CAC']]

                    # Add total row
                    total_row_eff = pd.DataFrame({
                        'Location': ['TOTAL'],
                        'Ad Spend': [f"€{total_ad_spend:,.0f}"],
                        'Ad Conv': [f"{total_ad_conv:,.0f}"],
                        'Bookings': [f"{total_bookings:,.0f}"],
                        'Revenue': [f"€{total_revenue:,.0f}"],
                        'Marketing %': [f"{total_marketing_pct:.1f}%"],
                        'Ad Share': [f"{total_ad_share:.1f}%"],
                        'Rev/€ Spent': [f"€{total_rev_per_spend:.1f}"],
                        'Blended CAC': [f"€{total_blended_cac:.1f}"]
                    })
                    eff_display = pd.concat([eff_display, total_row_eff], ignore_index=True)

                    # Column config with help tooltips
                    efficiency_column_config = {
                        'Location': st.column_config.TextColumn('Location'),
                        'Ad Spend': st.column_config.TextColumn('Ad Spend', help='Total marketing spend per location'),
                        'Ad Conv': st.column_config.TextColumn('Ad Conv', help='Conversions tracked by ad platforms'),
                        'Bookings': st.column_config.TextColumn('Bookings', help='Actual bookings from booking system'),
                        'Revenue': st.column_config.TextColumn('Revenue', help='Total revenue from booking system'),
                        'Marketing %': st.column_config.TextColumn('Marketing %', help='Ad Spend / Revenue (lower = more efficient)'),
                        'Ad Share': st.column_config.TextColumn('Ad Share', help='Ad conversions as % of total bookings'),
                        'Rev/€ Spent': st.column_config.TextColumn('Rev/€ Spent', help='Revenue per marketing euro (correlation, not ROI)'),
                        'Blended CAC': st.column_config.TextColumn('Blended CAC', help='Ad Spend / Total Bookings (cost to acquire any customer)'),
                    }

                    # Style TOTAL row
                    def highlight_total_eff(row):
                        if row['Location'] == 'TOTAL':
                            return ['font-weight: bold; background-color: #f3f4f6'] * len(row)
                        return [''] * len(row)

                    styled_eff = eff_display.style.apply(highlight_total_eff, axis=1)
                    st.dataframe(styled_eff, use_container_width=True, hide_index=True, column_config=efficiency_column_config)

                    # Scatter: Marketing Spend vs Revenue
                    st.markdown("#### Marketing Investment vs Revenue")
                    scatter_eff = eff_df[eff_df['Revenue'] > 0].copy()

                    if len(scatter_eff) > 1:
                        fig_eff = px.scatter(
                            scatter_eff,
                            x='Ad Spend',
                            y='Revenue',
                            color='Marketing %',
                            color_continuous_scale=['#22c55e', '#facc15', '#ef4444'],
                            hover_name='Location',
                            text='Location',
                            labels={
                                'Ad Spend': 'Marketing Spend (€)',
                                'Revenue': 'Revenue (€)',
                                'Marketing %': 'Marketing %'
                            }
                        )
                        fig_eff.update_traces(textposition='top center')
                        fig_eff.update_layout(
                            height=400,
                            coloraxis_colorbar_title='Marketing %'
                        )
                        st.plotly_chart(fig_eff, use_container_width=True)
                        st.caption("Points further from origin with low Marketing % = efficient locations")
                else:
                    st.info("Upload booking data to see revenue metrics. Revenue data comes from 'Total gross' column in booking files.")

            # ===== TAB 3: CAPACITY =====
            with tab3:
                st.markdown("#### Capacity & Marketing Alignment")
                st.caption("Compare marketing investment with location capacity utilization")

                # Build capacity data
                capacity_data = []
                num_weeks = 4  # Assume November = 4 weeks

                for loc in loc_summary['Location'].tolist():
                    cap_info = get_capacity_per_location(loc, num_weeks)
                    bookings = revenue_data.get(loc, {}).get('bookings', 0) if revenue_data else 0
                    ad_spend = loc_summary[loc_summary['Location'] == loc]['Spend'].values[0]

                    if cap_info and cap_info['period_total'] > 0:
                        occupancy = (bookings / cap_info['period_total'] * 100) if cap_info['period_total'] > 0 else 0
                        capacity_data.append({
                            'Location': loc,
                            'Capacity': cap_info['period_total'],
                            'Bookings': bookings,
                            'Occupancy': occupancy,
                            'Ad Spend': ad_spend,
                            'Available': cap_info['period_total'] - bookings,
                            'Cost/Slot': ad_spend / bookings if bookings > 0 else 0
                        })

                if capacity_data:
                    cap_df = pd.DataFrame(capacity_data)
                    cap_df = cap_df.sort_values('Occupancy', ascending=False)

                    # Identify opportunities
                    # Use absolute thresholds for meaningful status labels
                    # 65% occupancy is a reasonable target for "good" performance
                    occupancy_threshold = 65
                    # Calculate median spend from locations with actual bookings
                    locations_with_bookings = cap_df[cap_df['Bookings'] > 0]
                    spend_threshold = locations_with_bookings['Ad Spend'].median() if len(locations_with_bookings) > 0 else 0

                    def get_opportunity_label(row):
                        if row['Bookings'] == 0:
                            return 'No Data'  # No booking data for this location
                        elif row['Occupancy'] < occupancy_threshold and row['Ad Spend'] < spend_threshold:
                            return 'Opportunity'  # Low occupancy, low spend - room to grow
                        elif row['Occupancy'] >= occupancy_threshold and row['Ad Spend'] < spend_threshold:
                            return 'Efficient'  # High occupancy, low spend
                        elif row['Occupancy'] < occupancy_threshold and row['Ad Spend'] >= spend_threshold:
                            return 'Review'  # Low occupancy, high spend
                        else:
                            return 'Performing'  # High occupancy, high spend

                    cap_df['Status'] = cap_df.apply(get_opportunity_label, axis=1)

                    # Calculate totals
                    total_capacity = cap_df['Capacity'].sum()
                    total_bookings_cap = cap_df['Bookings'].sum()
                    total_available = cap_df['Available'].sum()
                    total_occupancy = (total_bookings_cap / total_capacity * 100) if total_capacity > 0 else 0
                    total_spend_cap = cap_df['Ad Spend'].sum()

                    # Format for display
                    cap_display = cap_df.copy()
                    cap_display['Capacity'] = cap_display['Capacity'].apply(lambda x: f"{x:,.0f}")
                    cap_display['Bookings'] = cap_display['Bookings'].apply(lambda x: f"{x:,.0f}")
                    cap_display['Occupancy'] = cap_display['Occupancy'].apply(lambda x: f"{x:.0f}%")
                    cap_display['Ad Spend'] = cap_display['Ad Spend'].apply(lambda x: f"€{x:,.0f}")
                    cap_display['Available'] = cap_display['Available'].apply(lambda x: f"{x:,.0f}")
                    cap_display['Cost/Slot'] = cap_display['Cost/Slot'].apply(lambda x: f"€{x:.1f}")

                    # Select columns
                    cap_display = cap_display[['Location', 'Capacity', 'Bookings', 'Occupancy', 'Ad Spend', 'Available', 'Cost/Slot', 'Status']]

                    total_row_cap = pd.DataFrame({
                        'Location': ['TOTAL'],
                        'Capacity': [f"{total_capacity:,.0f}"],
                        'Bookings': [f"{total_bookings_cap:,.0f}"],
                        'Occupancy': [f"{total_occupancy:.0f}%"],
                        'Ad Spend': [f"€{total_spend_cap:,.0f}"],
                        'Available': [f"{total_available:,.0f}"],
                        'Cost/Slot': ['-'],
                        'Status': ['-']
                    })
                    cap_display = pd.concat([cap_display, total_row_cap], ignore_index=True)

                    # Column config with help tooltips
                    capacity_column_config = {
                        'Location': st.column_config.TextColumn('Location'),
                        'Capacity': st.column_config.TextColumn('Capacity', help='Total available slots (weekly capacity × 4 weeks)'),
                        'Bookings': st.column_config.TextColumn('Bookings', help='Actual bookings from booking system'),
                        'Occupancy': st.column_config.TextColumn('Occupancy', help='Bookings / Capacity'),
                        'Ad Spend': st.column_config.TextColumn('Ad Spend', help='Marketing spend per location'),
                        'Available': st.column_config.TextColumn('Available', help='Remaining slots (Capacity - Bookings)'),
                        'Cost/Slot': st.column_config.TextColumn('Cost/Slot', help='Ad Spend / Bookings filled'),
                        'Status': st.column_config.TextColumn('Status', help='Based on 65% occupancy threshold. Efficient = ≥65% occ + below median spend | Performing = ≥65% occ + above median spend | Opportunity = <65% occ + below median spend | Review = <65% occ + above median spend'),
                    }

                    # Style TOTAL row
                    def highlight_total_cap(row):
                        if row['Location'] == 'TOTAL':
                            return ['font-weight: bold; background-color: #f3f4f6'] * len(row)
                        return [''] * len(row)

                    styled_cap = cap_display.style.apply(highlight_total_cap, axis=1)
                    st.dataframe(styled_cap, use_container_width=True, hide_index=True, column_config=capacity_column_config)

                    # Scatter: Occupancy vs Marketing Spend
                    st.markdown("#### Marketing Spend vs Occupancy")

                    # Use cap_df which already has Status column (need numeric values for chart)
                    cap_chart = cap_df[['Location', 'Capacity', 'Bookings', 'Occupancy', 'Ad Spend', 'Available', 'Cost/Slot', 'Status']].copy()

                    fig_cap = px.scatter(
                        cap_chart,
                        x='Ad Spend',
                        y='Occupancy',
                        size='Capacity',
                        color='Status',
                        hover_name='Location',
                        text='Location',
                        color_discrete_map={
                            'Opportunity': '#3b82f6',
                            'Efficient': '#22c55e',
                            'Review': '#ef4444',
                            'Performing': '#a855f7',
                            'No Data': '#9ca3af'
                        }
                    )

                    # Add quadrant lines using the thresholds
                    fig_cap.add_hline(y=occupancy_threshold, line_dash="dash", line_color="#999",
                                      annotation_text=f"Target: {occupancy_threshold}%")
                    fig_cap.add_vline(x=spend_threshold, line_dash="dash", line_color="#999",
                                      annotation_text=f"Median Spend: €{spend_threshold:,.0f}")

                    fig_cap.update_traces(textposition='top center')
                    fig_cap.update_layout(
                        height=450,
                        xaxis_title='Marketing Spend (€)',
                        yaxis_title='Occupancy (%)',
                        yaxis=dict(range=[0, 100])
                    )
                    st.plotly_chart(fig_cap, use_container_width=True)
                    st.caption("Bubble size = Total Capacity | Quadrants show investment opportunities")
                else:
                    st.info("No capacity data available for the matched locations. Check that location names in booking data match the capacity configuration.")
        else:
            st.info("No location-specific campaigns found. Campaigns need location names (e.g., 'Nijmegen', 'Rotterdam') to appear here.")

    # Reset button
    st.sidebar.markdown("---")
    if st.sidebar.button("Clear All & Start Over", key="mkt_reset"):
        st.session_state.google_ads_df = None
        st.session_state.meta_ads_df = None
        st.session_state.stdc_tags = {}
        st.rerun()
