"""
Kuuma Booking Analyzer - Overview Page
Landing page after login with app information and data status
"""

import streamlit as st
import pandas as pd
import sys
sys.path.insert(0, '..')
from data_loader import init_session_state, DEMO_MODE

# Page configuration
st.set_page_config(
    page_title="Kuuma - Overview",
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

# Initialize session state
init_session_state()

# Check authentication
if not st.session_state.get('authenticated', False):
    st.warning("Please log in to access this page.")
    st.page_link("app.py", label="Go to Login", icon=":material/login:")
    st.stop()

# Header with logo
col1, col2 = st.columns([1, 5])
with col1:
    st.image("assets/logo_black.svg", width=120)
with col2:
    st.title("Kuuma Booking Analyzer")
    st.markdown("**Customer insights & booking intelligence**")

if DEMO_MODE:
    st.info(":material/science: **Demo account** — All data shown is sample data for demonstration purposes")

# Navigation in sidebar
st.sidebar.header("Navigation")
st.sidebar.page_link("pages/1_Overview.py", label="Overview", icon=":material/home:")
st.sidebar.page_link("app.py", label="Booking Patterns", icon=":material/calendar_month:")
st.sidebar.page_link("pages/3_Customers.py", label="Recurring Customers", icon=":material/group:")
st.sidebar.page_link("pages/4_Revenue.py", label="Revenue & Value", icon=":material/payments:")
st.sidebar.page_link("pages/5_Promotions.py", label="Promotions", icon=":material/local_offer:")
st.sidebar.page_link("pages/6_Capacity.py", label="Capacity Analysis", icon=":material/analytics:")
st.sidebar.page_link("pages/7_Marketing.py", label="Marketing", icon=":material/campaign:")
st.sidebar.page_link("pages/8_Chart_Test.py", label="Chart Test", icon=":material/science:")

st.markdown("---")

# Welcome Section
st.markdown("## Welcome to the Kuuma Booking Analyzer")

st.markdown("""
This application provides comprehensive insights into your sauna booking data, helping you understand
customer behavior, optimize pricing, and make data-driven decisions.
""")

# What is this app section
st.markdown("### What does this app do?")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Analyze Booking Patterns**
    - Understand when customers book vs. when they visit
    - Identify lead time patterns by location
    - Discover seasonal trends

    **Track Customer Behavior**
    - Segment customers by visit frequency
    - Calculate recurring customer rates
    - Identify VIP customers
    """)

with col2:
    st.markdown("""
    **Monitor Revenue & Value**
    - Track revenue by location and time
    - Calculate Customer Lifetime Value (CLV)
    - Analyze promotion effectiveness

    **Optimize Operations**
    - Monitor capacity utilization
    - Track marketing campaign performance
    - Compare platform ROI (Google Ads vs Meta)
    """)

st.markdown("---")

# How it works section
st.markdown("### How does it work?")

st.markdown("""
The app automatically loads data from your connected Google Drive folder. The following data sources are used:

| Data Type | File Format | Description |
|-----------|-------------|-------------|
| **Booking Creation Dates** | Excel (.xls/.xlsx) | When bookings were made (Bookeo export) |
| **Visit Dates** | Excel (.xls/.xlsx) | When customers actually visited (Bookeo export) |
| **Google Ads** | CSV | Campaign performance data |
| **Meta Ads** | CSV | Facebook/Instagram campaign data |

Data is refreshed automatically every 4 hours from Google Drive.
""")

st.markdown("---")

# Data Status Section
st.markdown("### Data Status")

# Check what data is loaded
df1 = st.session_state.get('df1')
df2 = st.session_state.get('df2')
google_ads = st.session_state.get('google_ads_df')
meta_ads = st.session_state.get('meta_ads_df')

# Booking data metrics (3 columns)
col1, col2, col3 = st.columns(3)

with col1:
    if df1 is not None:
        st.metric("Booking Records", f"{len(df1):,}")
        st.caption("Booking creation data")
    else:
        st.metric("Booking Records", "Not loaded")
        st.caption("Upload or connect Drive")

with col2:
    if df2 is not None:
        st.metric("Visit Records", f"{len(df2):,}")
        st.caption("Visit date data")
    else:
        st.metric("Visit Records", "Not loaded")
        st.caption("Upload or connect Drive")

with col3:
    if df1 is not None and df2 is not None and len(df1) > 0:
        cancellations = len(df1) - len(df2)
        cancellation_rate = (cancellations / len(df1)) * 100
        st.metric("Cancellation Rate", f"{cancellation_rate:.1f}%")
        st.caption(f"{cancellations:,} bookings cancelled")
    else:
        st.metric("Cancellation Rate", "—")
        st.caption("Requires both files")

# Explanation for cancellation rate
if df1 is not None and df2 is not None:
    st.info(":material/info: The difference between Booking Records and Visit Records represents customers who booked but cancelled before their visit date.")

# Marketing data metrics (2 columns)
st.markdown("#### Marketing Data")
col1, col2 = st.columns(2)

with col1:
    if google_ads is not None:
        st.metric("Google Ads Campaigns", f"{len(google_ads):,}")
        st.caption("Campaign rows")
    else:
        st.metric("Google Ads", "Not loaded")
        st.caption("Optional")

with col2:
    if meta_ads is not None:
        st.metric("Meta Ads Campaigns", f"{len(meta_ads):,}")
        st.caption("Campaign rows")
    else:
        st.metric("Meta Ads", "Not loaded")
        st.caption("Optional")

# Date range info
if df2 is not None:
    date_col = None
    for col in ['Start date', 'Start Date', 'Visit Date', 'Date']:
        if col in df2.columns:
            date_col = col
            break

    if date_col:
        try:
            dates = pd.to_datetime(df2[date_col], errors='coerce')
            min_date = dates.min()
            max_date = dates.max()
            if pd.notna(min_date) and pd.notna(max_date):
                st.info(f"**Data covers:** {min_date.strftime('%d %b %Y')} to {max_date.strftime('%d %b %Y')}")
        except:
            pass

# Locations info
if df2 is not None:
    location_col = None
    if 'Location' in df2.columns:
        location_col = 'Location'
    elif 'Activity' in df2.columns:
        location_col = 'Activity'
    elif 'Tour' in df2.columns:
        location_col = 'Tour'

    if location_col:
        locations = df2[location_col].dropna().unique()
        # Filter out non-location entries (UTM test, etc.)
        locations = [loc for loc in locations if str(loc).lower().startswith('kuuma')]
        if len(locations) > 0:
            with st.expander(f"**{len(locations)} locations available**", expanded=False):
                for loc in sorted(locations):
                    st.write(f"- {loc}")

st.markdown("---")

# Quick navigation
st.markdown("### Get Started")

col1, col2, col3 = st.columns(3)

with col1:
    st.page_link("app.py", label="View Booking Patterns", icon=":material/calendar_month:", use_container_width=True)

with col2:
    st.page_link("pages/3_Customers.py", label="Analyze Customers", icon=":material/group:", use_container_width=True)

with col3:
    st.page_link("pages/4_Revenue.py", label="Check Revenue", icon=":material/payments:", use_container_width=True)
