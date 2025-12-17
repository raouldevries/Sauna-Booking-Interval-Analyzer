"""
Kuuma Booking Analyzer - Chart Test Page
Test page for evaluating new chart visualizations
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
sys.path.insert(0, '..')
from data_loader import init_session_state, get_location_column, get_available_locations

# Page configuration
st.set_page_config(
    page_title="Kuuma - Chart Test",
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

# Initialize session state
init_session_state()

# Check authentication
if not st.session_state.get('authenticated', False):
    st.warning("Please log in to access this page.")
    st.page_link("app.py", label="Go to Login", icon=":material/login:")
    st.stop()

# Header
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://kuuma.nl/wp-content/themes/kuuma/images/logo.svg", width=120)
with col2:
    st.title("Chart Test Page")
    st.markdown("**Evaluate new chart visualizations before adding to main app**")

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

# Check if data is loaded
df1 = st.session_state.get('df1')
df2 = st.session_state.get('df2')
google_ads_df = st.session_state.get('google_ads_df')
meta_ads_df = st.session_state.get('meta_ads_df')

if df1 is None or df2 is None:
    st.warning("Please load booking data to view charts.")
    st.page_link("app.py", label="Go to Booking Patterns to load data", icon=":material/upload:")
    st.stop()


# ============== DATA PREPARATION ==============

@st.cache_data
def prepare_booking_data(df1_json, df2_json):
    """Prepare booking data for analysis - uses LEFT join to keep ALL locations from df1."""
    df1 = pd.read_json(df1_json)
    df2 = pd.read_json(df2_json)

    # Get booking ID column
    id_col = 'Booking number' if 'Booking number' in df1.columns else df1.columns[0]

    # Start with df1 as base - this has all the booking info including ALL locations
    data = df1.copy()

    # Unify location column FIRST (before any filtering)
    if 'Location' in data.columns:
        data['location'] = data['Location']
    elif 'Tour' in data.columns:
        data['location'] = data['Tour']
    elif 'Activity' in data.columns:
        data['location'] = data['Activity']

    # Filter to Kuuma locations only
    if 'location' in data.columns:
        data = data[data['location'].str.lower().str.startswith('kuuma', na=False)]

    # Convert booking date
    if 'Created' in data.columns:
        data['booking_date'] = pd.to_datetime(data['Created'], errors='coerce')

    # Merge with df2 to get visit dates (LEFT join to keep all df1 records)
    if id_col in df2.columns:
        visit_col = 'Start' if 'Start' in df2.columns else df2.columns[1]
        df2_prep = df2[[id_col, visit_col]].copy()
        df2_prep.columns = [id_col, 'visit_date']
        df2_prep['visit_date'] = pd.to_datetime(df2_prep['visit_date'], errors='coerce')

        # LEFT join - keeps ALL records from df1
        data = data.merge(df2_prep, on=id_col, how='left')

    # Calculate lead time (only for records that have both dates)
    if 'booking_date' in data.columns and 'visit_date' in data.columns:
        data['lead_time'] = (data['visit_date'] - data['booking_date']).dt.days
        # Don't filter out negative lead times - just set them to NaN
        data.loc[data['lead_time'] < 0, 'lead_time'] = np.nan

    return data

# Prepare data
try:
    merged_data = prepare_booking_data(
        df1.to_json(date_format='iso'),
        df2.to_json(date_format='iso')
    )
except Exception as e:
    st.error(f"Error preparing data: {e}")
    st.stop()

st.success(f"Data loaded: {len(merged_data):,} bookings ready for analysis")

st.markdown("---")

# ============== CHART 1: REVENUE PER PARTICIPANT ==============
st.markdown("## 1. Revenue per Participant by Location")
st.markdown("**Question:** Which locations generate highest revenue efficiency?")

if 'Total gross' in merged_data.columns and 'Participants' in merged_data.columns and 'location' in merged_data.columns:
    # Location filter
    chart1_locations = sorted(merged_data['location'].dropna().unique())
    chart1_location_options = ['All Locations'] + list(chart1_locations)
    selected_chart1_location = st.selectbox(
        "Select Location",
        options=chart1_location_options,
        index=0,
        key="chart1_location_filter"
    )

    if selected_chart1_location == 'All Locations':
        chart1_data = merged_data.copy()
    else:
        chart1_data = merged_data[merged_data['location'] == selected_chart1_location].copy()

    rev_data = chart1_data[['location', 'Total gross', 'Participants']].dropna()
    rev_data = rev_data[(rev_data['Total gross'] > 0) & (rev_data['Participants'] > 0)]

    if len(rev_data) > 0:
        # Calculate per-booking and per-participant metrics
        location_metrics = rev_data.groupby('location').agg({
            'Total gross': ['mean', 'sum'],
            'Participants': ['mean', 'sum'],
        }).reset_index()
        location_metrics.columns = ['location', 'avg_revenue', 'total_revenue', 'avg_participants', 'total_participants']
        location_metrics['revenue_per_participant'] = location_metrics['total_revenue'] / location_metrics['total_participants']
        location_metrics['booking_count'] = rev_data.groupby('location').size().values

        # Bubble chart
        fig1 = px.scatter(
            location_metrics,
            x='avg_participants',
            y='avg_revenue',
            size='booking_count',
            color='location',
            hover_data=['revenue_per_participant', 'booking_count'],
            title='Revenue vs Group Size by Location',
            labels={
                'avg_participants': 'Avg Participants per Booking',
                'avg_revenue': 'Avg Revenue per Booking',
                'booking_count': 'Total Bookings'
            }
        )
        fig1.update_layout(showlegend=True)
        st.plotly_chart(fig1, use_container_width=True)

        # Metrics table
        display_df = location_metrics[['location', 'avg_revenue', 'avg_participants', 'revenue_per_participant', 'booking_count']].copy()
        display_df.columns = ['Location', 'Avg Revenue', 'Avg Group Size', 'Revenue/Participant', 'Bookings']
        display_df = display_df.sort_values('Revenue/Participant', ascending=False)
        display_df['Avg Revenue'] = display_df['Avg Revenue'].apply(lambda x: f"€{x:,.0f}")
        display_df['Revenue/Participant'] = display_df['Revenue/Participant'].apply(lambda x: f"€{x:,.1f}")
        display_df['Avg Group Size'] = display_df['Avg Group Size'].round(1)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("**Insight:** Higher revenue per participant indicates better pricing efficiency.")
    else:
        st.info("Not enough revenue data.")
else:
    st.info("Revenue, participants, or location data not available.")

st.markdown("---")

# ============== CHART 2: PRIVATE EVENT ANALYSIS ==============
st.markdown("## 2. Private Event Analysis")
st.markdown("**Question:** How do private events differ from regular bookings?")

if 'Private event' in merged_data.columns:
    # Location filter
    if 'location' in merged_data.columns:
        chart2_locations = sorted(merged_data['location'].dropna().unique())
        chart2_location_options = ['All Locations'] + list(chart2_locations)
        selected_chart2_location = st.selectbox(
            "Select Location",
            options=chart2_location_options,
            index=0,
            key="chart2_location_filter"
        )

        if selected_chart2_location == 'All Locations':
            pe_data = merged_data.copy()
        else:
            pe_data = merged_data[merged_data['location'] == selected_chart2_location].copy()
    else:
        pe_data = merged_data.copy()

    # Prepare private event data
    pe_data['is_private'] = pe_data['Private event'].fillna('No').astype(str).str.lower()
    pe_data['is_private'] = pe_data['is_private'].apply(lambda x: 'Private Event' if x in ['yes', 'true', '1', 'ja'] else 'Regular Booking')

    # Count by type
    pe_counts = pe_data['is_private'].value_counts()

    if 'Private Event' in pe_counts.index and pe_counts.get('Private Event', 0) > 0:
        col1, col2 = st.columns(2)

        with col1:
            # Pie chart
            fig2a = px.pie(
                values=pe_counts.values,
                names=pe_counts.index,
                title='Booking Type Distribution'
            )
            st.plotly_chart(fig2a, use_container_width=True)

        with col2:
            # Comparison metrics
            comparison = pe_data.groupby('is_private').agg({
                'Total gross': 'mean' if 'Total gross' in pe_data.columns else 'count',
                'Participants': 'mean' if 'Participants' in pe_data.columns else 'count',
                'lead_time': 'mean' if 'lead_time' in pe_data.columns else 'count'
            }).round(1)

            if 'Total gross' in pe_data.columns:
                comparison.columns = ['Avg Revenue', 'Avg Group Size', 'Avg Lead Time']
            st.dataframe(comparison, use_container_width=True)

        # Revenue comparison bar chart
        if 'Total gross' in pe_data.columns:
            rev_by_type = pe_data.groupby('is_private')['Total gross'].agg(['mean', 'sum', 'count']).reset_index()
            rev_by_type.columns = ['Type', 'Avg Revenue', 'Total Revenue', 'Count']

            fig2b = px.bar(
                rev_by_type,
                x='Type',
                y=['Avg Revenue', 'Total Revenue'],
                title='Revenue by Booking Type',
                barmode='group'
            )
            st.plotly_chart(fig2b, use_container_width=True)

        st.markdown("**Insight:** Private events often have higher revenue and group sizes.")
    else:
        st.info("No private events found in the data.")
else:
    st.info("Private event column not available in data.")

st.markdown("---")

# ============== CHART 3: PARTICIPANTS PER BOOKING ==============
st.markdown("## 3. Participants per Booking Analysis")
st.markdown("**Question:** What is the distribution of group sizes? How many participants typically book together?")

if 'Participants' in merged_data.columns:
    # Location filter
    if 'location' in merged_data.columns:
        chart3_locations = sorted(merged_data['location'].dropna().unique())
        chart3_location_options = ['All Locations'] + list(chart3_locations)
        selected_chart3_location = st.selectbox(
            "Select Location",
            options=chart3_location_options,
            index=0,
            key="chart3_location_filter"
        )

        if selected_chart3_location == 'All Locations':
            participants_data = merged_data.copy()
        else:
            participants_data = merged_data[merged_data['location'] == selected_chart3_location].copy()
    else:
        participants_data = merged_data.copy()

    # Clean participants data
    participants_data['Participants'] = pd.to_numeric(participants_data['Participants'], errors='coerce')
    participants_data = participants_data[participants_data['Participants'] > 0]

    if len(participants_data) > 0:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Group Size", f"{participants_data['Participants'].mean():.1f}")
        with col2:
            st.metric("Median Group Size", f"{participants_data['Participants'].median():.0f}")
        with col3:
            st.metric("Most Common Size", f"{participants_data['Participants'].mode().iloc[0]:.0f}")
        with col4:
            st.metric("Max Group Size", f"{participants_data['Participants'].max():.0f}")

        # Histogram of participants distribution
        fig7_hist = px.histogram(
            participants_data,
            x='Participants',
            nbins=20,
            title='Distribution of Group Sizes (Participants per Booking)',
            labels={'Participants': 'Number of Participants', 'count': 'Number of Bookings'},
            color_discrete_sequence=['#FF6B35']
        )
        fig7_hist.update_layout(
            xaxis_title='Number of Participants',
            yaxis_title='Number of Bookings',
            bargap=0.1
        )
        st.plotly_chart(fig7_hist, use_container_width=True)

        # Group size categories
        def size_category(p):
            if p == 1:
                return '1 (Solo)'
            elif p == 2:
                return '2 (Couple)'
            elif p <= 4:
                return '3-4 (Small group)'
            elif p <= 6:
                return '5-6 (Medium group)'
            elif p <= 10:
                return '7-10 (Large group)'
            else:
                return '11+ (Extra large)'

        participants_data['size_category'] = participants_data['Participants'].apply(size_category)

        # Order categories
        category_order = ['1 (Solo)', '2 (Couple)', '3-4 (Small group)', '5-6 (Medium group)', '7-10 (Large group)', '11+ (Extra large)']
        participants_data['size_category'] = pd.Categorical(participants_data['size_category'], categories=category_order, ordered=True)

        col1, col2 = st.columns(2)

        with col1:
            # Pie chart by category
            category_counts = participants_data['size_category'].value_counts().sort_index()
            fig7_pie = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title='Bookings by Group Size Category',
                color_discrete_sequence=px.colors.sequential.Oranges_r
            )
            st.plotly_chart(fig7_pie, use_container_width=True)

        with col2:
            # Stats table by category
            category_stats = participants_data.groupby('size_category').agg({
                'Participants': ['count', 'sum']
            }).reset_index()
            category_stats.columns = ['Category', 'Bookings', 'Total Participants']
            category_stats['% of Bookings'] = (category_stats['Bookings'] / category_stats['Bookings'].sum() * 100).round(1)
            category_stats['% of Participants'] = (category_stats['Total Participants'] / category_stats['Total Participants'].sum() * 100).round(1)
            st.dataframe(category_stats, use_container_width=True, hide_index=True)

        # Revenue by group size (if available)
        if 'Total gross' in participants_data.columns:
            st.markdown("### Revenue by Group Size")
            revenue_by_size = participants_data.groupby('size_category').agg({
                'Total gross': ['mean', 'sum'],
                'Participants': 'count'
            }).reset_index()
            revenue_by_size.columns = ['Category', 'Avg Revenue', 'Total Revenue', 'Bookings']
            revenue_by_size = revenue_by_size.sort_index()

            fig7_rev = px.bar(
                revenue_by_size,
                x='Category',
                y='Avg Revenue',
                title='Average Revenue by Group Size Category',
                color='Total Revenue',
                color_continuous_scale='Oranges',
                text='Bookings'
            )
            fig7_rev.update_traces(texttemplate='%{text} bookings', textposition='outside')
            st.plotly_chart(fig7_rev, use_container_width=True)

        st.markdown("**Insight:** Understanding group size distribution helps optimize capacity planning and pricing strategies.")
    else:
        st.info("No valid participants data available.")
else:
    st.info("Participants column not available in data.")

st.markdown("---")

# ============== SUMMARY ==============
st.markdown("## Summary: Chart Recommendations")

st.markdown("""
| # | Chart | Key Insight | Recommended Page |
|---|-------|-------------|------------------|
| 1 | Revenue per Participant | Pricing efficiency by location | Revenue & Value |
| 2 | Private Event Analysis | Private event value comparison | Revenue & Value |
| 3 | Participants per Booking | Group size distribution | Capacity Analysis |
""")
