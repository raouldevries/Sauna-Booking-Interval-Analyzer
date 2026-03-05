"""
Kuuma Booking Analyzer - Capacity Analysis Page
Occupancy/utilization analysis per location and time period
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
sys.path.insert(0, '..')
from data_loader import init_session_state, apply_demo_transform

# Page configuration
st.set_page_config(
    page_title="Kuuma - Capacity Analysis",
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

# Capacity data from Bezettings analyse
# dal + piek = weekday (ma-do) capacity
# Note: Den Bosch, Nyma, Bloemendaal are new locations without capacity data yet
LOCATION_CAPACITY = {
    # Flagship
    'Kuuma Marineterrein Matsu': {'dal': 96, 'piek': 366, 'weekday': 462, 'weekend': 198, 'cluster': 'Flagship'},  # K1
    'Kuuma Marineterrein Bjørk': {'dal': 128, 'piek': 488, 'weekday': 616, 'weekend': 264, 'cluster': 'Flagship'},  # K8
    'Kuuma Scheveningen': {'dal': 96, 'piek': 520, 'weekday': 616, 'weekend': 264, 'cluster': 'Flagship'},  # K13
    # Groeier
    'Kuuma Noord': {'dal': 96, 'piek': 336, 'weekday': 432, 'weekend': 192, 'cluster': 'Groeier'},  # K2
    'Kuuma Sloterplas': {'dal': 112, 'piek': 392, 'weekday': 504, 'weekend': 224, 'cluster': 'Groeier'},  # K12
    'Kuuma Aan ´t IJ (Centrum)': {'dal': 96, 'piek': 324, 'weekday': 420, 'weekend': 180, 'cluster': 'Groeier'},  # K5
    'Kuuma Nijmegen Lent': {'dal': 120, 'piek': 342, 'weekday': 462, 'weekend': 198, 'cluster': 'Groeier'},  # K3
    'Kuuma Rotterdam Rijnhaven': {'dal': 96, 'piek': 324, 'weekday': 420, 'weekend': 180, 'cluster': 'Groeier'},  # K6
    # Onderbenut
    'Kuuma Rotterdam Delfshaven': {'dal': 120, 'piek': 342, 'weekday': 462, 'weekend': 198, 'cluster': 'Onderbenut'},  # K4
    'Kuuma Katwijk': {'dal': 96, 'piek': 282, 'weekday': 378, 'weekend': 162, 'cluster': 'Onderbenut'},  # K9
    'Kuuma Wijk aan Zee': {'dal': 96, 'piek': 336, 'weekday': 432, 'weekend': 192, 'cluster': 'Onderbenut'},  # K11
}

CLUSTER_TARGETS = {
    'Flagship': {'total': (75, 80), 'weekend': (90, 95), 'piek': (80, 85), 'dal': (55, 60)},
    'Groeier': {'total': (60, 65), 'weekend': (80, 85), 'piek': (70, 75), 'dal': (30, 40)},
    'Onderbenut': {'total': (50, 55), 'weekend': (70, 75), 'piek': (60, 65), 'dal': (25, 30)},
}


def find_capacity_match(location_name):
    """Find matching LOCATION_CAPACITY entry for a given location name.
    Returns the matching key from LOCATION_CAPACITY or None if no match found.
    Supports flexible matching (case-insensitive, partial matching)."""
    if not location_name:
        return None

    # Exact match first
    if location_name in LOCATION_CAPACITY:
        return location_name

    # Case-insensitive exact match
    location_lower = location_name.lower()
    for cap_key in LOCATION_CAPACITY.keys():
        if cap_key.lower() == location_lower:
            return cap_key

    # Partial match - check if location name is contained in capacity key or vice versa
    for cap_key in LOCATION_CAPACITY.keys():
        cap_key_lower = cap_key.lower()
        # Remove "Kuuma " prefix for matching
        cap_key_stripped = cap_key_lower.replace('kuuma ', '')

        if location_lower in cap_key_lower or cap_key_stripped in location_lower:
            return cap_key

        # Also try matching without "Kuuma" prefix in the location name
        location_stripped = location_lower.replace('kuuma ', '')
        if location_stripped in cap_key_stripped or cap_key_stripped in location_stripped:
            return cap_key

    return None


def get_capacity_for_location(location_name):
    """Get capacity data for a location, with flexible matching."""
    cap_key = find_capacity_match(location_name)
    if cap_key:
        return LOCATION_CAPACITY[cap_key]
    return None


# Header with logo
col1, col2 = st.columns([1, 5])
with col1:
    st.image("assets/logo_black.svg", width=120)
with col2:
    st.title("Kuuma Booking Analyzer")
    st.markdown("**Customer insights & booking intelligence**")

# Reserve container for date range selector (filled after data loads)
date_range_container = st.container()

st.markdown("## Capacity Analysis")
st.markdown("Occupancy rates and utilization per location and time period")

# Initialize session state using centralized function
init_session_state()

# Check authentication
if not st.session_state.get('authenticated', False):
    st.warning("Please log in to access this page.")
    st.page_link("app.py", label="Go to Login", icon=":material/login:")
    st.stop()

# Parse uploaded files
@st.cache_data
def load_excel_file(uploaded_file):
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

# Time period classification
def classify_time_period(dt):
    """
    Classify a datetime into time periods:
    - weekend: Fri, Sat, Sun (all hours)
    - dal: Mon-Thu 10:00-16:00
    - piek: Mon-Thu outside 10:00-16:00
    """
    day_of_week = dt.weekday()  # 0=Mon, 6=Sun
    hour = dt.hour
    if day_of_week >= 4:  # Fri (4), Sat (5), Sun (6)
        return 'weekend'
    elif 10 <= hour < 16:  # Mon-Thu 10:00-16:00
        return 'dal'
    else:
        return 'piek'

def is_weekday(dt):
    """Check if datetime is Monday-Thursday"""
    return dt.weekday() < 4

# Reserve container for navigation at top of sidebar
nav_container = st.sidebar.container()

# Sidebar - Upload section
st.sidebar.header("Upload & Configure")

# File uploaders with multiple file support
uploaded_files1 = st.sidebar.file_uploader(
    "Booking Creation Dates (.xls/.xlsx)",
    type=["xls", "xlsx"],
    help="Upload files containing when bookings were created. You can select multiple files from different Bookeo instances.",
    key="cap_file1",
    accept_multiple_files=True
)

uploaded_files2 = st.sidebar.file_uploader(
    "Visit Dates (.xls/.xlsx)",
    type=["xls", "xlsx"],
    help="Upload files containing when customers actually visited. You can select multiple files from different Bookeo instances.",
    key="cap_file2",
    accept_multiple_files=True
)

# Load and merge File 1
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

# Load and merge File 2
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

# Main content
if st.session_state.df2 is None:
    st.info("**No data loaded.** Please upload your Visit Dates file using the sidebar to begin capacity analysis.")
    st.markdown("""
    ### Getting Started
    1. Upload your visit dates file (contains visit times and locations)
    2. View occupancy rates by location and time period
    3. Compare actual performance vs targets

    ### Time Periods Analyzed
    - **Dal uren (Ma/Do 10:00-16:00)**: Off-peak weekday hours
    - **Piek uren (al het overige)**: Peak weekday hours (Mon-Thu outside 10-16)
    - **Weekend (vrij/za/zo)**: Friday, Saturday, Sunday
    - **Maandag - donderdag**: Combined weekday view
    """)
else:
    df2 = st.session_state.df2

    # Column mapping
    st.sidebar.markdown("---")
    st.sidebar.subheader("Column Mapping")

    # Smart defaults for visit dates file
    default_start_col = "Start" if "Start" in df2.columns else df2.columns[0]
    if "Location" in df2.columns:
        default_location_col = "Location"
    elif "Activity" in df2.columns:
        default_location_col = "Activity"
    elif "Tour" in df2.columns:
        default_location_col = "Tour"
    else:
        default_location_col = None
    default_participants_col = "Participants" if "Participants" in df2.columns else None

    visit_col = st.sidebar.selectbox(
        "Visit Date/Time",
        options=df2.columns.tolist(),
        index=df2.columns.tolist().index(default_start_col) if default_start_col in df2.columns else 0,
        key="cap_visit_col"
    )

    location_options = ["None"] + df2.columns.tolist()
    location_default_index = location_options.index(default_location_col) if default_location_col in location_options else 0

    location_col = st.sidebar.selectbox(
        "Location",
        options=location_options,
        index=location_default_index,
        key="cap_location_col"
    )

    participants_options = ["None"] + df2.columns.tolist()
    participants_default_index = participants_options.index(default_participants_col) if default_participants_col in participants_options else 0

    participants_col = st.sidebar.selectbox(
        "Participants (optional)",
        options=participants_options,
        index=participants_default_index,
        help="Number of people per booking. If not selected, each booking counts as 1.",
        key="cap_participants_col"
    )

    # Check if location column is configured
    if location_col == "None":
        st.warning("""
        **Location column not configured.**

        To use capacity analysis, please select a Location column in the sidebar.
        This allows us to calculate occupancy rates per location.
        """)
    else:
        # Prepare data
        capacity_data = df2[[visit_col, location_col]].copy()
        capacity_data.columns = ['visit_datetime', 'location']

        # Add participants if available
        if participants_col != "None":
            capacity_data['participants'] = df2[participants_col]
        else:
            capacity_data['participants'] = 1

        # Convert to datetime
        capacity_data['visit_datetime'] = pd.to_datetime(capacity_data['visit_datetime'], errors='coerce')

        # Remove invalid dates
        capacity_data = capacity_data[capacity_data['visit_datetime'].notna()]

        # Show all locations from the data file (no filtering against predefined list)
        if len(capacity_data) == 0:
            st.warning("**No valid data found.** Please check your data file has valid dates.")
        else:
            # Classify time periods
            capacity_data['time_period'] = capacity_data['visit_datetime'].apply(classify_time_period)
            capacity_data['is_weekday'] = capacity_data['visit_datetime'].apply(is_weekday)

            # Extract week for weekly analysis
            capacity_data['week'] = capacity_data['visit_datetime'].dt.isocalendar().week
            capacity_data['year'] = capacity_data['visit_datetime'].dt.year
            capacity_data['year_week'] = capacity_data['year'].astype(str) + '-W' + capacity_data['week'].astype(str).str.zfill(2)

            # Date range selector in reserved container (under header)
            min_date = capacity_data['visit_datetime'].min().date()
            max_date = capacity_data['visit_datetime'].max().date()

            with date_range_container:
                date_col1, date_col2 = st.columns([2, 4])
                with date_col1:
                    date_range = st.date_input(
                        "Date Range (Visit Date)",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                        help="Filter capacity data by visit date",
                        key="cap_date_range"
                    )

            # Apply date filter
            if len(date_range) == 2:
                start_date, end_date = date_range
                capacity_data = capacity_data[
                    (capacity_data['visit_datetime'].dt.date >= start_date) &
                    (capacity_data['visit_datetime'].dt.date <= end_date)
                ]

            # Location filter in sidebar
            st.sidebar.markdown("---")
            st.sidebar.subheader("Filters")

            # Filter to only show Kuuma locations (exclude test data like "UTM test")
            all_locations = capacity_data['location'].unique().tolist()
            kuuma_locations = [loc for loc in all_locations if loc.lower().startswith('kuuma')]
            capacity_data = capacity_data[capacity_data['location'].isin(kuuma_locations)]
            available_locations = sorted(kuuma_locations)
            selected_locations = st.sidebar.multiselect(
                "Locations",
                options=available_locations,
                default=available_locations,
                key="cap_locations"
            )

            if selected_locations:
                capacity_data = capacity_data[capacity_data['location'].isin(selected_locations)]

            if len(capacity_data) == 0:
                st.warning("No data matches the selected filters.")
            else:
                # Calculate date range string
                data_min = capacity_data['visit_datetime'].min()
                data_max = capacity_data['visit_datetime'].max()
                if data_min.year == data_max.year:
                    date_range_str = f"{data_min.strftime('%b %d')} - {data_max.strftime('%b %d, %Y')}"
                else:
                    date_range_str = f"{data_min.strftime('%b %y')} - {data_max.strftime('%b %y')}"

                # Calculate occupancy per location per time period
                # Group by location and time period
                period_stats = capacity_data.groupby(['location', 'time_period']).agg({
                    'participants': 'sum'
                }).reset_index()
                period_stats.columns = ['location', 'time_period', 'bookings']

                # Calculate weekday stats (dal + piek combined)
                weekday_stats = capacity_data[capacity_data['is_weekday']].groupby('location').agg({
                    'participants': 'sum'
                }).reset_index()
                weekday_stats.columns = ['location', 'weekday_bookings']

                # Calculate number of weeks in the data for averaging
                num_weeks = capacity_data['year_week'].nunique()

                # Create summary table
                summary_data = []
                for location in selected_locations:
                    cap = get_capacity_for_location(location)
                    if cap is None:
                        continue

                    cluster = cap['cluster']

                    # Get bookings for each period
                    dal_bookings = period_stats[(period_stats['location'] == location) & (period_stats['time_period'] == 'dal')]['bookings'].sum()
                    piek_bookings = period_stats[(period_stats['location'] == location) & (period_stats['time_period'] == 'piek')]['bookings'].sum()
                    weekend_bookings = period_stats[(period_stats['location'] == location) & (period_stats['time_period'] == 'weekend')]['bookings'].sum()

                    weekday_row = weekday_stats[weekday_stats['location'] == location]
                    weekday_bookings = weekday_row['weekday_bookings'].values[0] if len(weekday_row) > 0 else 0

                    # Calculate weekly averages
                    dal_weekly = dal_bookings / num_weeks if num_weeks > 0 else 0
                    piek_weekly = piek_bookings / num_weeks if num_weeks > 0 else 0
                    weekend_weekly = weekend_bookings / num_weeks if num_weeks > 0 else 0
                    weekday_weekly = weekday_bookings / num_weeks if num_weeks > 0 else 0

                    # Calculate occupancy percentages
                    dal_occupancy = (dal_weekly / cap['dal'] * 100) if cap['dal'] > 0 else 0
                    piek_occupancy = (piek_weekly / cap['piek'] * 100) if cap['piek'] > 0 else 0
                    weekend_occupancy = (weekend_weekly / cap['weekend'] * 100) if cap['weekend'] > 0 else 0
                    weekday_occupancy = (weekday_weekly / cap['weekday'] * 100) if cap['weekday'] > 0 else 0

                    summary_data.append({
                        'Location': location,
                        'Cluster': cluster,
                        'Dal uren (%)': round(dal_occupancy),
                        'Piek uren (%)': round(piek_occupancy),
                        'Ma-Do (%)': round(weekday_occupancy),
                        'Weekend (%)': round(weekend_occupancy),
                        'Dal (weekly avg)': round(dal_weekly),
                        'Piek (weekly avg)': round(piek_weekly),
                        'Weekend (weekly avg)': round(weekend_weekly),
                    })

                summary_df = pd.DataFrame(summary_data)

                if len(summary_df) == 0:
                    st.warning("No capacity data available for the selected locations.")
                else:
                    # Key metrics
                    st.markdown("### Key Metrics")

                    total_visits = capacity_data['participants'].sum()
                    avg_dal = summary_df['Dal uren (%)'].mean()
                    avg_piek = summary_df['Piek uren (%)'].mean()
                    avg_weekend = summary_df['Weekend (%)'].mean()

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(
                            "Total Visits",
                            f"{total_visits:,}",
                            help="Total number of visitors in the selected period"
                        )
                    with col2:
                        st.metric(
                            "Avg Dal Occupancy",
                            f"{avg_dal:.1f}%",
                            help="Average off-peak (Mon-Thu 10-16) occupancy across locations"
                        )
                    with col3:
                        st.metric(
                            "Avg Piek Occupancy",
                            f"{avg_piek:.1f}%",
                            help="Average peak hour occupancy across locations"
                        )
                    with col4:
                        st.metric(
                            "Avg Weekend Occupancy",
                            f"{avg_weekend:.1f}%",
                            help="Average weekend occupancy across locations"
                        )

                    st.info(f"**Analysis Period:** {num_weeks} weeks | Occupancy calculated as weekly average bookings / weekly capacity")

                    # Sauna Visit Heatmap
                    st.markdown("---")
                    st.markdown("### Sauna Visit Heatmap")

                    st.markdown("""
                    Visualize visit patterns by day of week and time of day. Darker colors indicate higher visit volumes.
                    Select a location to see its specific pattern, or choose "All Locations" for aggregate view.
                    """)

                    # Prepare heatmap data
                    heatmap_data = capacity_data.copy()
                    heatmap_data['visit_hour'] = heatmap_data['visit_datetime'].dt.hour
                    heatmap_data['day_of_week'] = heatmap_data['visit_datetime'].dt.day_name()

                    # Order days of week
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    heatmap_data['day_of_week'] = pd.Categorical(heatmap_data['day_of_week'], categories=day_order, ordered=True)

                    # Location filter dropdown for heatmap
                    heatmap_location_options = ['All Locations'] + selected_locations
                    selected_heatmap_location = st.selectbox(
                        'Select Location',
                        options=heatmap_location_options,
                        index=0,
                        key='cap_heatmap_location_selector'
                    )

                    # Filter data by selected location
                    if selected_heatmap_location == 'All Locations':
                        heatmap_filtered = heatmap_data
                    else:
                        heatmap_filtered = heatmap_data[heatmap_data['location'] == selected_heatmap_location]

                    # Group by day and hour
                    visit_counts = heatmap_filtered.groupby(['day_of_week', 'visit_hour'], observed=True).agg({
                        'participants': 'sum'
                    }).reset_index()
                    visit_counts.columns = ['Day', 'Hour', 'Visits']

                    # Create complete grid (all day/hour combinations)
                    all_combinations = pd.MultiIndex.from_product(
                        [day_order, range(24)],
                        names=['Day', 'Hour']
                    ).to_frame(index=False)

                    heatmap_grid = all_combinations.merge(visit_counts, on=['Day', 'Hour'], how='left').fillna(0)
                    heatmap_grid['Visits'] = heatmap_grid['Visits'].astype(int)

                    # Pivot to create heatmap matrix
                    heatmap_matrix = heatmap_grid.pivot(index='Hour', columns='Day', values='Visits')
                    heatmap_matrix = heatmap_matrix[day_order]  # Ensure correct day order

                    # Create heatmap
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=heatmap_matrix.values,
                        x=day_order,
                        y=[f"{h:02d}:00" for h in range(24)],
                        colorscale='Purples',
                        hovertemplate='%{x}<br>%{y}<br>%{z} visits<extra></extra>',
                        showscale=True,
                        colorbar=dict(title='Visits')
                    ))

                    fig_heatmap.update_layout(
                        title=f'Visit Heatmap - {selected_heatmap_location}',
                        xaxis_title='Day of Week',
                        yaxis_title='Hour of Day',
                        height=600,
                        xaxis=dict(side='bottom'),
                        yaxis=dict(autorange='reversed')  # Hours from top to bottom
                    )

                    st.plotly_chart(fig_heatmap, use_container_width=True)

                    # Peak time insights
                    if len(visit_counts) > 0:
                        peak_row = visit_counts.loc[visit_counts['Visits'].idxmax()]
                        peak_day = peak_row['Day']
                        peak_hour = int(peak_row['Hour'])
                        peak_count = int(peak_row['Visits'])

                        st.info(f"""
**Peak Time for {selected_heatmap_location}**: {peak_day} at {peak_hour:02d}:00 - {(peak_hour+1):02d}:00 with {peak_count} visits.
                        """)

                    with st.expander("How to Read the Heatmap", expanded=False):
                        st.markdown("""
**Understanding the Visualization:**
- **X-axis**: Days of the week (Monday through Sunday)
- **Y-axis**: Hours of the day (00:00 to 23:00 in 24-hour format)
- **Color intensity**: Darker purple = more visits, lighter = fewer visits
- **Hover**: Move your mouse over any cell to see exact visit counts

**Operational Insights:**
- **Dark clusters** indicate peak times requiring maximum staffing and capacity
- **Light areas** show off-peak times suitable for maintenance or promotions
- **Vertical patterns** (dark columns) reveal busy days regardless of time
- **Horizontal patterns** (dark rows) show popular time slots across the week
- **Compare locations** using the dropdown to identify unique patterns per branch

**Staffing Strategy:**
Use this heatmap to optimize your team schedule - ensure adequate coverage during dark (high-traffic) periods and consider reduced staffing during light (low-traffic) times.
                        """)

                    # Initialize custom targets in session state if not exists
                    if 'custom_targets' not in st.session_state:
                        st.session_state.custom_targets = {}

                    # Function to get target for a location/period
                    def get_target(location, period):
                        # Check for custom target first
                        if location in st.session_state.custom_targets:
                            custom = st.session_state.custom_targets[location].get(period)
                            if custom is not None and custom > 0:
                                return custom

                        # Fall back to cluster default
                        cluster = LOCATION_CAPACITY.get(location, {}).get('cluster', 'Groeier')
                        cluster_targets = CLUSTER_TARGETS.get(cluster, {'dal': (30, 40), 'piek': (60, 70), 'weekend': (70, 80)})
                        return cluster_targets[period][0]  # Use lower bound as target

                    # Location comparison table
                    st.markdown("---")
                    st.markdown("### Occupancy by Location")

                    # Week selector
                    available_weeks = sorted(capacity_data['year_week'].unique().tolist())
                    week_options = ['All weeks (average)'] + available_weeks

                    selected_week = st.selectbox(
                        "Select Week",
                        options=week_options,
                        index=0,
                        key="cap_week_selector",
                        help="View occupancy for a specific week or average across all weeks"
                    )

                    # Filter data if specific week selected and recalculate
                    if selected_week != 'All weeks (average)':
                        week_data = capacity_data[capacity_data['year_week'] == selected_week]
                        num_weeks_display = 1

                        # Recalculate period_stats for selected week
                        period_stats = week_data.groupby(['location', 'time_period']).agg({
                            'participants': 'sum'
                        }).reset_index()
                        period_stats.columns = ['location', 'time_period', 'bookings']

                        weekday_stats = week_data[week_data['is_weekday']].groupby('location').agg({
                            'participants': 'sum'
                        }).reset_index()
                        weekday_stats.columns = ['location', 'weekday_bookings']

                        # Rebuild summary_df for selected week
                        summary_data = []
                        for location in selected_locations:
                            cap = get_capacity_for_location(location)
                            if cap is None:
                                continue

                            cluster = cap['cluster']

                            dal_bookings = period_stats[(period_stats['location'] == location) & (period_stats['time_period'] == 'dal')]['bookings'].sum()
                            piek_bookings = period_stats[(period_stats['location'] == location) & (period_stats['time_period'] == 'piek')]['bookings'].sum()
                            weekend_bookings = period_stats[(period_stats['location'] == location) & (period_stats['time_period'] == 'weekend')]['bookings'].sum()

                            weekday_row = weekday_stats[weekday_stats['location'] == location]
                            weekday_bookings = weekday_row['weekday_bookings'].values[0] if len(weekday_row) > 0 else 0

                            dal_occupancy = (dal_bookings / cap['dal'] * 100) if cap['dal'] > 0 else 0
                            piek_occupancy = (piek_bookings / cap['piek'] * 100) if cap['piek'] > 0 else 0
                            weekend_occupancy = (weekend_bookings / cap['weekend'] * 100) if cap['weekend'] > 0 else 0
                            weekday_occupancy = (weekday_bookings / cap['weekday'] * 100) if cap['weekday'] > 0 else 0

                            summary_data.append({
                                'Location': location,
                                'Cluster': cluster,
                                'Dal uren (%)': round(dal_occupancy),
                                'Piek uren (%)': round(piek_occupancy),
                                'Ma-Do (%)': round(weekday_occupancy),
                                'Weekend (%)': round(weekend_occupancy),
                            })

                        summary_df = pd.DataFrame(summary_data)
                        st.caption(f"Showing data for week: **{selected_week}**")
                    else:
                        st.caption(f"Showing average across **{num_weeks} weeks**")

                    # Build display table with targets
                    display_data = []
                    for _, row in summary_df.iterrows():
                        location = row['Location']
                        display_data.append({
                            'Location': location,
                            'Cluster': row['Cluster'],
                            'Dal uren (%)': int(row['Dal uren (%)']),
                            'Dal Target (%)': int(get_target(location, 'dal')),
                            'Piek uren (%)': int(row['Piek uren (%)']),
                            'Piek Target (%)': int(get_target(location, 'piek')),
                            'Weekend (%)': int(row['Weekend (%)']),
                            'Weekend Target (%)': int(get_target(location, 'weekend')),
                        })

                    display_df = pd.DataFrame(display_data)
                    display_df = display_df.sort_values('Weekend (%)', ascending=False)

                    # Color coding function for actual vs target
                    def color_vs_target(row):
                        styles = [''] * len(row)
                        col_pairs = [
                            ('Dal uren (%)', 'Dal Target (%)'),
                            ('Piek uren (%)', 'Piek Target (%)'),
                            ('Weekend (%)', 'Weekend Target (%)')
                        ]
                        for actual_col, target_col in col_pairs:
                            actual = row[actual_col]
                            target = row[target_col]
                            idx = row.index.get_loc(actual_col)
                            if actual >= target:
                                styles[idx] = 'background-color: #dcfce7'  # Green
                            elif actual >= target * 0.9:
                                styles[idx] = 'background-color: #fef3c7'  # Yellow
                            else:
                                styles[idx] = 'background-color: #fee2e2'  # Red
                        return styles

                    styled_df = display_df.style.apply(color_vs_target, axis=1)

                    capacity_config = {
                        'Location': st.column_config.TextColumn('Location'),
                        'Cluster': st.column_config.TextColumn('Cluster', help='Location category (Flagship, Groeier, Onderbenut)'),
                        'Dal uren (%)': st.column_config.NumberColumn('Dal uren (%)', help='Occupancy during off-peak hours (Mon-Thu 10:00-16:00)'),
                        'Dal Target (%)': st.column_config.NumberColumn('Dal Target (%)', help='Target occupancy for off-peak hours'),
                        'Piek uren (%)': st.column_config.NumberColumn('Piek uren (%)', help='Occupancy during peak hours (Mon-Thu outside 10:00-16:00)'),
                        'Piek Target (%)': st.column_config.NumberColumn('Piek Target (%)', help='Target occupancy for peak hours'),
                        'Weekend (%)': st.column_config.NumberColumn('Weekend (%)', help='Occupancy during weekend (Fri-Sun)'),
                        'Weekend Target (%)': st.column_config.NumberColumn('Weekend Target (%)', help='Target occupancy for weekend'),
                    }

                    st.dataframe(styled_df, use_container_width=True, hide_index=True, column_config=capacity_config)
                    st.caption("Green = At/above target | Yellow = Within 10% | Red = More than 10% below")

                    # Editable targets table
                    with st.expander("Edit Target Values", expanded=False):
                        target_df = display_df[['Location', 'Dal Target (%)', 'Piek Target (%)', 'Weekend Target (%)']].copy()

                        target_config = {
                            'Location': st.column_config.TextColumn('Location'),
                            'Dal Target (%)': st.column_config.NumberColumn('Dal Target (%)', min_value=0, max_value=100),
                            'Piek Target (%)': st.column_config.NumberColumn('Piek Target (%)', min_value=0, max_value=100),
                            'Weekend Target (%)': st.column_config.NumberColumn('Weekend Target (%)', min_value=0, max_value=100),
                        }

                        edited_targets = st.data_editor(
                            target_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config=target_config,
                            disabled=['Location'],
                            key='target_editor'
                        )

                        # Check if any values changed and save to session state
                        targets_changed = False
                        for _, row in edited_targets.iterrows():
                            location = row['Location']
                            new_targets = {
                                'dal': int(row['Dal Target (%)']),
                                'piek': int(row['Piek Target (%)']),
                                'weekend': int(row['Weekend Target (%)'])
                            }
                            # Compare with displayed values
                            old_dal = int(display_df[display_df['Location'] == location]['Dal Target (%)'].values[0])
                            old_piek = int(display_df[display_df['Location'] == location]['Piek Target (%)'].values[0])
                            old_weekend = int(display_df[display_df['Location'] == location]['Weekend Target (%)'].values[0])

                            if new_targets['dal'] != old_dal or new_targets['piek'] != old_piek or new_targets['weekend'] != old_weekend:
                                targets_changed = True
                                st.session_state.custom_targets[location] = new_targets

                        # Rerun to update the colored table if targets changed
                        if targets_changed:
                            st.rerun()

                        if st.button("Reset to Cluster Defaults", key="reset_targets"):
                            st.session_state.custom_targets = {}
                            st.rerun()

                    # Weekly trend chart
                    st.markdown("---")
                    st.markdown("### Weekly Trend")

                    # Calculate weekly occupancy
                    weekly_data = capacity_data.groupby(['year_week', 'location', 'time_period']).agg({
                        'participants': 'sum'
                    }).reset_index()

                    # Pivot to get periods as columns
                    weekly_pivot = weekly_data.pivot_table(
                        index=['year_week', 'location'],
                        columns='time_period',
                        values='participants',
                        fill_value=0
                    ).reset_index()

                    # Calculate occupancy for each week
                    trend_data = []
                    for _, row in weekly_pivot.iterrows():
                        location = row['location']
                        cap = get_capacity_for_location(location)
                        if cap is None:
                            continue

                        dal_occ = (row.get('dal', 0) / cap['dal'] * 100) if cap['dal'] > 0 else 0
                        piek_occ = (row.get('piek', 0) / cap['piek'] * 100) if cap['piek'] > 0 else 0
                        weekend_occ = (row.get('weekend', 0) / cap['weekend'] * 100) if cap['weekend'] > 0 else 0

                        trend_data.append({
                            'Week': row['year_week'],
                            'Location': location,
                            'Dal (%)': dal_occ,
                            'Piek (%)': piek_occ,
                            'Weekend (%)': weekend_occ
                        })

                    trend_df = pd.DataFrame(trend_data)

                    if len(trend_df) > 0:
                        # Location selector for trend
                        trend_location = st.selectbox(
                            "Select Location for Trend",
                            options=['All Locations (Average)'] + list(summary_df['Location'].unique()),
                            key="trend_location"
                        )

                        if trend_location == 'All Locations (Average)':
                            trend_plot = trend_df.groupby('Week')[['Dal (%)', 'Piek (%)', 'Weekend (%)']].mean().reset_index()
                        else:
                            trend_plot = trend_df[trend_df['Location'] == trend_location][['Week', 'Dal (%)', 'Piek (%)', 'Weekend (%)']]

                        # Melt for plotting
                        trend_melted = trend_plot.melt(id_vars=['Week'], var_name='Period', value_name='Occupancy (%)')

                        fig_trend = px.line(
                            trend_melted,
                            x='Week',
                            y='Occupancy (%)',
                            color='Period',
                            title=f'Weekly Occupancy Trend - {trend_location}',
                            markers=True
                        )

                        fig_trend.update_layout(
                            height=400,
                            yaxis=dict(range=[0, 100]),
                            xaxis_tickangle=-45
                        )

                        st.plotly_chart(fig_trend, use_container_width=True)

                    # Explanation
                    st.markdown("---")
                    with st.expander("Understanding the Metrics", expanded=False):
                        st.markdown("""
                        ### Time Periods
                        - **Dal uren (Ma/Do 10:00-16:00)**: Off-peak hours on weekdays. Lower traffic, opportunity for promotions.
                        - **Piek uren (al het overige)**: Peak hours on weekdays (before 10:00 and after 16:00). Higher demand.
                        - **Weekend (vrij/za/zo)**: All hours on Friday, Saturday, and Sunday. Typically highest demand.
                        - **Ma-Do**: Combined Monday-Thursday view (dal + piek together).

                        ### Capacity
                        Weekly capacity is based on the number of available seats per time period:
                        - Capacity varies by location and time period
                        - Occupancy % = (Weekly bookings / Weekly capacity) × 100

                        ### Cluster Targets
                        Locations are grouped into clusters with different performance targets:
                        - **Flagship** (K1, K8): Highest targets (75-80% overall, 90-95% weekend)
                        - **Groeier** (Growth): Medium targets (60-65% overall, 80-85% weekend)
                        - **Onderbenut** (Underutilized): Lower targets with growth potential (50-55% overall)

                        ### Strategic Insights
                        - Focus marketing on dal uren to improve off-peak utilization
                        - Weekend capacity is typically well-utilized - consider price optimization
                        - Compare locations within the same cluster for fair benchmarking
                        """)

    # Reset button
    st.sidebar.markdown("---")
    if st.sidebar.button("Clear All & Start Over", key="cap_reset"):
        st.session_state.df1 = None
        st.session_state.df2 = None
        st.rerun()
