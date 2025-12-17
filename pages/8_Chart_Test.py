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


@st.cache_data
def get_location_list(locations_series):
    """Cache sorted unique locations."""
    return sorted(locations_series.dropna().unique())


@st.cache_data
def compute_revenue_metrics(locations, revenues, participants):
    """Cached computation of revenue per participant metrics."""
    df = pd.DataFrame({
        'location': locations,
        'Total gross': revenues,
        'Participants': participants
    }).dropna()
    df = df[(df['Total gross'] > 0) & (df['Participants'] > 0)]

    if len(df) == 0:
        return None

    metrics = df.groupby('location').agg({
        'Total gross': ['mean', 'sum'],
        'Participants': ['mean', 'sum'],
    }).reset_index()
    metrics.columns = ['location', 'avg_revenue', 'total_revenue', 'avg_participants', 'total_participants']
    metrics['revenue_per_participant'] = metrics['total_revenue'] / metrics['total_participants']
    metrics['booking_count'] = df.groupby('location').size().values
    return metrics


@st.cache_data
def compute_private_event_metrics(is_private_series, revenues, participants, lead_times):
    """Cached computation of private event metrics."""
    df = pd.DataFrame({
        'is_private': is_private_series,
        'Total gross': revenues,
        'Participants': participants,
        'lead_time': lead_times
    })

    counts = df['is_private'].value_counts()
    comparison = df.groupby('is_private').agg({
        'Total gross': 'mean',
        'Participants': 'mean',
        'lead_time': 'mean'
    }).round(1).reset_index()
    comparison.columns = ['Type', 'Avg Revenue', 'Avg Group Size', 'Avg Lead Time']

    rev_by_type = df.groupby('is_private')['Total gross'].agg(['mean', 'count']).reset_index()
    rev_by_type.columns = ['Type', 'Avg Revenue', 'Bookings']

    return counts, comparison, rev_by_type


@st.cache_data
def compute_participants_metrics(participants_series, revenues=None):
    """Cached computation of participants distribution metrics."""
    df = pd.DataFrame({'Participants': participants_series})
    if revenues is not None:
        df['Total gross'] = revenues
    df = df[df['Participants'] > 0]

    if len(df) == 0:
        return None, None, None, None

    # Basic stats
    stats = {
        'mean': df['Participants'].mean(),
        'median': df['Participants'].median(),
        'mode': df['Participants'].mode().iloc[0] if len(df['Participants'].mode()) > 0 else 0,
        'max': df['Participants'].max()
    }

    # Vectorized size category assignment (faster than apply)
    p = df['Participants']
    conditions = [
        p == 1,
        p == 2,
        (p >= 3) & (p <= 4),
        (p >= 5) & (p <= 6),
        (p >= 7) & (p <= 10),
        p >= 11
    ]
    choices = ['1 (Solo)', '2 (Couple)', '3-4 (Small group)', '5-6 (Medium group)', '7-10 (Large group)', '11+ (Extra large)']
    df['size_category'] = np.select(conditions, choices, default='Unknown')

    category_order = ['1 (Solo)', '2 (Couple)', '3-4 (Small group)', '5-6 (Medium group)', '7-10 (Large group)', '11+ (Extra large)']
    df['size_category'] = pd.Categorical(df['size_category'], categories=category_order, ordered=True)

    # Category stats
    category_stats = df.groupby('size_category', observed=True).agg({
        'Participants': ['count', 'sum']
    }).reset_index()
    category_stats.columns = ['Category', 'Bookings', 'Total Participants']
    category_stats['% of Bookings'] = (category_stats['Bookings'] / category_stats['Bookings'].sum() * 100).round(1)
    category_stats['% of Participants'] = (category_stats['Total Participants'] / category_stats['Total Participants'].sum() * 100).round(1)

    # Revenue by size (if available)
    revenue_by_size = None
    if 'Total gross' in df.columns and df['Total gross'].notna().any():
        revenue_by_size = df.groupby('size_category', observed=True).agg({
            'Total gross': ['mean', 'sum'],
            'Participants': 'count'
        }).reset_index()
        revenue_by_size.columns = ['Category', 'Avg Revenue', 'Total Revenue', 'Bookings']

    return stats, df, category_stats, revenue_by_size


@st.cache_data
def compute_marketing_metrics(google_ads_json, meta_ads_json, booking_dates, revenues):
    """Cached computation of marketing spend vs bookings metrics."""
    # Parse marketing data
    dfs_to_combine = []
    if google_ads_json:
        dfs_to_combine.append(pd.read_json(google_ads_json))
    if meta_ads_json:
        dfs_to_combine.append(pd.read_json(meta_ads_json))

    if not dfs_to_combine:
        return None

    combined = pd.concat(dfs_to_combine, ignore_index=True)

    # Apply filters
    if 'spend' in combined.columns:
        combined = combined[combined['spend'] > 0]
    if 'campaign_name' in combined.columns:
        combined = combined[combined['campaign_name'].notna()]
        combined = combined[combined['campaign_name'].astype(str).str.strip() != '']
        combined = combined[~combined['campaign_name'].astype(str).str.strip().str.match(r'^-+$')]

    total_spend = combined['spend'].sum() if 'spend' in combined.columns else 0

    # Get marketing date range
    marketing_min_date = None
    marketing_max_date = None
    if 'Reporting starts' in combined.columns and 'Reporting ends' in combined.columns:
        start_dates = pd.to_datetime(combined['Reporting starts'], errors='coerce')
        end_dates = pd.to_datetime(combined['Reporting ends'], errors='coerce')
        if start_dates.notna().any():
            marketing_min_date = start_dates.min()
        if end_dates.notna().any():
            marketing_max_date = end_dates.max()

    # Calculate daily bookings
    df = pd.DataFrame({
        'booking_date': pd.to_datetime(booking_dates),
        'revenue': revenues
    })
    df['date'] = df['booking_date'].dt.date
    daily = df.groupby('date').agg(
        bookings=('date', 'count'),
        revenue=('revenue', 'sum')
    ).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])

    # Marketing period
    if marketing_min_date is not None and marketing_max_date is not None:
        marketing_days = (marketing_max_date - marketing_min_date).days + 1
    else:
        marketing_days = (daily['date'].max() - daily['date'].min()).days + 1

    daily_spend = total_spend / marketing_days if marketing_days > 0 else 0

    # Rolling averages
    daily['bookings_7d_avg'] = daily['bookings'].rolling(window=7, min_periods=1).mean()
    daily['revenue_7d_avg'] = daily['revenue'].rolling(window=7, min_periods=1).mean()

    return {
        'total_spend': total_spend,
        'daily_spend': daily_spend,
        'marketing_days': marketing_days,
        'marketing_min_date': marketing_min_date,
        'marketing_max_date': marketing_max_date,
        'daily_bookings': daily
    }

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
    # Cache location list
    all_locations = get_location_list(merged_data['location'])
    chart1_location_options = ['All Locations'] + list(all_locations)
    selected_chart1_location = st.selectbox(
        "Select Location",
        options=chart1_location_options,
        index=0,
        key="chart1_location_filter"
    )

    # Filter data
    if selected_chart1_location == 'All Locations':
        chart1_data = merged_data
    else:
        chart1_data = merged_data[merged_data['location'] == selected_chart1_location]

    # Use cached computation
    location_metrics = compute_revenue_metrics(
        chart1_data['location'].tolist(),
        chart1_data['Total gross'].tolist(),
        chart1_data['Participants'].tolist()
    )

    if location_metrics is not None and len(location_metrics) > 0:
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
    # Location filter - reuse cached location list
    if 'location' in merged_data.columns:
        chart2_location_options = ['All Locations'] + list(all_locations)
        selected_chart2_location = st.selectbox(
            "Select Location",
            options=chart2_location_options,
            index=0,
            key="chart2_location_filter"
        )

        if selected_chart2_location == 'All Locations':
            pe_data = merged_data
        else:
            pe_data = merged_data[merged_data['location'] == selected_chart2_location]
    else:
        pe_data = merged_data

    # Vectorized private event classification (faster than apply)
    pe_private_raw = pe_data['Private event'].fillna('No').astype(str).str.lower()
    is_private = np.where(pe_private_raw.isin(['yes', 'true', '1', 'ja']), 'Private Event', 'Regular Booking')

    # Use cached computation
    pe_counts, comparison, rev_by_type = compute_private_event_metrics(
        is_private.tolist(),
        pe_data['Total gross'].tolist() if 'Total gross' in pe_data.columns else [0] * len(pe_data),
        pe_data['Participants'].tolist() if 'Participants' in pe_data.columns else [0] * len(pe_data),
        pe_data['lead_time'].tolist() if 'lead_time' in pe_data.columns else [0] * len(pe_data)
    )

    if 'Private Event' in pe_counts.index and pe_counts.get('Private Event', 0) > 0:
        # Calculate key metrics
        private_count = pe_counts.get('Private Event', 0)
        regular_count = pe_counts.get('Regular Booking', 0)
        total_count = private_count + regular_count
        private_pct = private_count / total_count * 100 if total_count > 0 else 0

        # Get avg revenues from comparison table
        private_row = comparison[comparison['Type'] == 'Private Event']
        regular_row = comparison[comparison['Type'] == 'Regular Booking']
        avg_rev_private = private_row['Avg Revenue'].iloc[0] if len(private_row) > 0 else 0
        avg_rev_regular = regular_row['Avg Revenue'].iloc[0] if len(regular_row) > 0 else 0
        rev_diff_pct = ((avg_rev_private - avg_rev_regular) / avg_rev_regular * 100) if avg_rev_regular > 0 else 0

        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Private Events", f"{private_count:,}")
        with col2:
            st.metric("% of Total Bookings", f"{private_pct:.1f}%")
        with col3:
            st.metric("Avg Revenue (Private)", f"€{avg_rev_private:.0f}")
        with col4:
            st.metric("vs Regular", f"{rev_diff_pct:+.0f}%", delta_color="normal")

        # Two-column layout: Pie chart and comparison table
        col1, col2 = st.columns([1, 1])

        with col1:
            # Pie chart with better formatting
            fig2a = px.pie(
                values=pe_counts.values,
                names=pe_counts.index,
                title='Booking Type Distribution',
                color_discrete_sequence=['#3498db', '#2ecc71'],
                hole=0.4
            )
            fig2a.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='%{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )
            fig2a.update_layout(
                showlegend=False,
                height=300,
                margin=dict(t=40, b=20, l=20, r=20)
            )
            st.plotly_chart(fig2a, use_container_width=True)

        with col2:
            # Format comparison table
            comparison_display = comparison.copy()
            comparison_display['Avg Revenue'] = comparison_display['Avg Revenue'].apply(lambda x: f"€{x:,.0f}")
            comparison_display['Avg Group Size'] = comparison_display['Avg Group Size'].apply(lambda x: f"{x:.1f}")
            comparison_display['Avg Lead Time'] = comparison_display['Avg Lead Time'].apply(lambda x: f"{x:.1f} days")
            comparison_display = comparison_display.sort_values('Type', ascending=False)

            st.markdown("**Comparison by Type**")
            st.dataframe(
                comparison_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Type': st.column_config.TextColumn('Type', width='medium'),
                    'Avg Revenue': st.column_config.TextColumn('Avg Revenue', width='small'),
                    'Avg Group Size': st.column_config.TextColumn('Avg Group Size', width='small'),
                    'Avg Lead Time': st.column_config.TextColumn('Avg Lead Time', width='small')
                }
            )

        # Revenue comparison bar chart
        rev_by_type_sorted = rev_by_type.sort_values('Avg Revenue', ascending=True)
        fig2b = px.bar(
            rev_by_type_sorted,
            x='Avg Revenue',
            y='Type',
            orientation='h',
            title='Average Revenue Comparison',
            text=rev_by_type_sorted['Avg Revenue'].apply(lambda x: f"€{x:,.0f}"),
            color='Type',
            color_discrete_sequence=['#3498db', '#2ecc71']
        )
        fig2b.update_traces(textposition='outside')
        fig2b.update_layout(
            showlegend=False,
            height=200,
            margin=dict(t=40, b=20, l=20, r=80),
            xaxis_title='Average Revenue (€)',
            yaxis_title=''
        )
        st.plotly_chart(fig2b, use_container_width=True)

        st.markdown("**Insight:** Private events often have higher revenue and group sizes, making them valuable for revenue optimization.")
    else:
        st.info("No private events found in the data.")
else:
    st.info("Private event column not available in data.")

st.markdown("---")

# ============== CHART 3: PARTICIPANTS PER BOOKING ==============
st.markdown("## 3. Participants per Booking Analysis")
st.markdown("**Question:** What is the distribution of group sizes? How many participants typically book together?")

if 'Participants' in merged_data.columns:
    # Location filter - reuse cached location list
    if 'location' in merged_data.columns:
        chart3_location_options = ['All Locations'] + list(all_locations)
        selected_chart3_location = st.selectbox(
            "Select Location",
            options=chart3_location_options,
            index=0,
            key="chart3_location_filter"
        )

        if selected_chart3_location == 'All Locations':
            participants_data = merged_data
        else:
            participants_data = merged_data[merged_data['location'] == selected_chart3_location]
    else:
        participants_data = merged_data

    # Clean participants data
    participants_clean = pd.to_numeric(participants_data['Participants'], errors='coerce')
    revenues_clean = participants_data['Total gross'] if 'Total gross' in participants_data.columns else None

    # Use cached computation
    stats, df_with_categories, category_stats, revenue_by_size = compute_participants_metrics(
        participants_clean.tolist(),
        revenues_clean.tolist() if revenues_clean is not None else None
    )

    if stats is not None:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Group Size", f"{stats['mean']:.1f}")
        with col2:
            st.metric("Median Group Size", f"{stats['median']:.0f}")
        with col3:
            st.metric("Most Common Size", f"{stats['mode']:.0f}")
        with col4:
            st.metric("Max Group Size", f"{stats['max']:.0f}")

        # Histogram of participants distribution
        fig7_hist = px.histogram(
            df_with_categories,
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

        col1, col2 = st.columns(2)

        with col1:
            # Pie chart by category
            category_counts = df_with_categories['size_category'].value_counts().sort_index()
            fig7_pie = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title='Bookings by Group Size Category',
                color_discrete_sequence=px.colors.sequential.Oranges_r
            )
            st.plotly_chart(fig7_pie, use_container_width=True)

        with col2:
            st.dataframe(category_stats, use_container_width=True, hide_index=True)

        # Revenue by group size (if available)
        if revenue_by_size is not None:
            st.markdown("### Revenue by Group Size")
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

# ============== CHART 4: SPEND VS BOOKINGS TIMELINE ==============
st.markdown("## 4. Marketing Spend vs Bookings Timeline")
st.markdown("**Question:** Does increased marketing spend lead to more bookings? What's the correlation?")

# Check if marketing data is available
has_marketing_data = (google_ads_df is not None and len(google_ads_df) > 0) or (meta_ads_df is not None and len(meta_ads_df) > 0)

if has_marketing_data and 'booking_date' in merged_data.columns:
    # Use cached computation for marketing metrics
    google_json = google_ads_df.to_json() if google_ads_df is not None else None
    meta_json = meta_ads_df.to_json() if meta_ads_df is not None else None

    marketing_metrics = compute_marketing_metrics(
        google_json,
        meta_json,
        merged_data['booking_date'].tolist(),
        merged_data['Total gross'].tolist() if 'Total gross' in merged_data.columns else [0] * len(merged_data)
    )

    if marketing_metrics is not None:
        total_spend = marketing_metrics['total_spend']
        daily_spend_estimate = marketing_metrics['daily_spend']
        marketing_days = marketing_metrics['marketing_days']
        marketing_min_date = marketing_metrics['marketing_min_date']
        marketing_max_date = marketing_metrics['marketing_max_date']
        daily_bookings = marketing_metrics['daily_bookings']

        total_bookings = daily_bookings['bookings'].sum()
        total_revenue = daily_bookings['revenue'].sum()

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Marketing Spend", f"€{total_spend:,.0f}")
        with col2:
            st.metric("Total Bookings", f"{total_bookings:,}")
        with col3:
            cpa = total_spend / total_bookings if total_bookings > 0 else 0
            st.metric("Avg CPA", f"€{cpa:.2f}")
        with col4:
            roas = total_revenue / total_spend if total_spend > 0 else 0
            st.metric("ROAS", f"{roas:.1f}x")

        # Create dual-axis chart
        fig4 = go.Figure()

        # Add bookings line (primary y-axis)
        fig4.add_trace(go.Scatter(
            x=daily_bookings['date'],
            y=daily_bookings['bookings_7d_avg'],
            name='Bookings (7-day avg)',
            line=dict(color='#2ecc71', width=2),
            yaxis='y'
        ))

        # Add revenue line (secondary y-axis)
        fig4.add_trace(go.Scatter(
            x=daily_bookings['date'],
            y=daily_bookings['revenue_7d_avg'],
            name='Revenue (7-day avg)',
            line=dict(color='#3498db', width=2, dash='dot'),
            yaxis='y2'
        ))

        # Add daily spend reference line
        fig4.add_hline(
            y=daily_spend_estimate,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Avg Daily Spend: €{daily_spend_estimate:.0f}",
            annotation_position="top right"
        )

        # Update layout for dual axis
        fig4.update_layout(
            title='Daily Bookings & Revenue Over Time',
            xaxis=dict(
                title='Date',
                tickformat='%d %b %Y',
                dtick='M1',
                tickangle=-45
            ),
            yaxis=dict(title='Bookings (7-day avg)', side='left', showgrid=False),
            yaxis2=dict(title='Revenue (7-day avg)', side='right', overlaying='y', showgrid=False),
            hovermode='x unified',
            height=450,
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )

        st.plotly_chart(fig4, use_container_width=True)

        # Correlation analysis
        if len(daily_bookings) > 7:
            daily_bookings_copy = daily_bookings.copy()
            daily_bookings_copy['week'] = daily_bookings_copy['date'].dt.isocalendar().week
            daily_bookings_copy['year'] = daily_bookings_copy['date'].dt.year
            weekly_data = daily_bookings_copy.groupby(['year', 'week']).agg({
                'bookings': 'sum',
                'revenue': 'sum'
            }).reset_index()

            correlation = weekly_data['bookings'].corr(weekly_data['revenue'])

            # Format marketing period for display
            if marketing_min_date is not None and marketing_max_date is not None:
                marketing_period_str = f"{marketing_min_date.strftime('%Y-%m-%d')} to {marketing_max_date.strftime('%Y-%m-%d')} ({marketing_days} days)"
            else:
                marketing_period_str = "Unknown (using booking data range)"

            st.markdown(f"""
            **Analysis:**
            - **Booking-Revenue Correlation:** {correlation:.2f} ({"Strong" if abs(correlation) > 0.7 else "Moderate" if abs(correlation) > 0.4 else "Weak"})
            - **Average Daily Bookings:** {daily_bookings['bookings'].mean():.1f}
            - **Marketing Period:** {marketing_period_str}
            - **Booking Data Range:** {daily_bookings['date'].min().strftime('%Y-%m-%d')} to {daily_bookings['date'].max().strftime('%Y-%m-%d')}
            """)

        st.markdown("**Insight:** Track how marketing spend correlates with booking volume over time. Look for lag effects - bookings may increase days after spend increases.")
    else:
        st.info("Unable to compute marketing metrics.")
else:
    if not has_marketing_data:
        st.info("Marketing data not loaded. Load Google Ads or Meta Ads data on the Marketing page to see this chart.")
    else:
        st.info("Booking date data not available.")

st.markdown("---")

# ============== SUMMARY ==============
st.markdown("## Summary: Chart Recommendations")

st.markdown("""
| # | Chart | Key Insight | Recommended Page |
|---|-------|-------------|------------------|
| 1 | Revenue per Participant | Pricing efficiency by location | Revenue & Value |
| 2 | Private Event Analysis | Private event value comparison | Revenue & Value |
| 3 | Participants per Booking | Group size distribution | Capacity Analysis |
| 4 | Spend vs Bookings Timeline | Marketing-booking correlation | Marketing |
""")
