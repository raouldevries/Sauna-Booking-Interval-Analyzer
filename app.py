import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Kuuma Booking Analyzer",
    page_icon="🔥",
    layout="wide"
)

# Header with logo
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://kuuma.nl/wp-content/themes/kuuma/images/logo.svg", width=120)
with col2:
    st.title("Kuuma Booking Analyzer")
    st.markdown("**Customer insights & booking intelligence**")

# Sidebar
st.sidebar.header("Upload & Configure")

# File uploaders
uploaded_file1 = st.sidebar.file_uploader(
    "Booking Creation Dates (.xls/.xlsx)",
    type=["xls", "xlsx"],
    help="Upload the file containing when bookings were created"
)

uploaded_file2 = st.sidebar.file_uploader(
    "Visit Dates (.xls/.xlsx)",
    type=["xls", "xlsx"],
    help="Upload the file containing when customers actually visited"
)

# Initialize session state
if 'df1' not in st.session_state:
    st.session_state.df1 = None
if 'df2' not in st.session_state:
    st.session_state.df2 = None

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

# Load File 1
if uploaded_file1 is not None:
    df1, error1 = load_excel_file(uploaded_file1)
    if error1:
        st.sidebar.error(f"Error reading File 1: {error1}")
    else:
        st.session_state.df1 = df1
        st.sidebar.success(f"File 1 loaded: {len(df1):,} rows")

# Load File 2
if uploaded_file2 is not None:
    df2, error2 = load_excel_file(uploaded_file2)
    if error2:
        st.sidebar.error(f"Error reading File 2: {error2}")
    else:
        st.session_state.df2 = df2
        st.sidebar.success(f"File 2 loaded: {len(df2):,} rows")

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
    default_location_col = "Activity" if "Activity" in df1.columns else None

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
        # Filters section
        st.sidebar.markdown("---")
        st.sidebar.subheader("Filters")

        # Date range filter
        min_date = processed_data['visit_date'].min().date()
        max_date = processed_data['visit_date'].max().date()

        date_range = st.sidebar.date_input(
            "Visit Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            help="Filter bookings by visit date"
        )

        # Location filter (if location column selected)
        if location_col != "None":
            all_locations = sorted(processed_data['location'].unique().tolist())
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

            # Calculate date range
            min_booking_date = filtered_data['booking_date'].min()
            max_booking_date = filtered_data['booking_date'].max()
            # Use compact format: if same year, show year only once
            if min_booking_date.year == max_booking_date.year:
                date_range_str = f"{min_booking_date.strftime('%b %d')} - {max_booking_date.strftime('%b %d, %Y')}"
            else:
                date_range_str = f"{min_booking_date.strftime('%b %y')} - {max_booking_date.strftime('%b %y')}"

            # Display data range
            st.markdown(f"**Data Range:** {date_range_str}")
            st.markdown("")

            # Display metrics
            st.markdown("### Key Metrics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Average Lead Time", f"{avg_interval:.1f} days")

            with col2:
                st.metric("Median Lead Time", f"{median_interval:.1f} days")

            with col3:
                st.metric("Total Bookings", f"{total_bookings:,}")

            with col4:
                st.metric("Same-Day Bookings", f"{same_day_pct:.1f}%")

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

                st.markdown("""
                Compare booking behavior across different locations.

                **Understanding Average vs Median:**
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

                st.dataframe(location_stats, use_container_width=True)

                # Sauna Visit Time Analysis (Day of Week & Time of Day)
                # Check if we have time data (not just dates)
                if (filtered_data['visit_date'].dt.hour.max() > 0 or
                    filtered_data['visit_date'].dt.minute.max() > 0):

                    st.markdown("---")
                    st.markdown("### Sauna Visit Heatmap")

                    st.markdown("""
                    Visualize visit patterns by day of week and time of day. Darker colors indicate higher visit volumes.
                    Select a location to see its specific pattern, or choose "All Locations" for aggregate view.
                    """)

                    # Prepare data
                    time_data = filtered_data.copy()
                    time_data['visit_hour'] = time_data['visit_date'].dt.hour
                    time_data['day_of_week'] = time_data['visit_date'].dt.day_name()

                    # Order days of week
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    time_data['day_of_week'] = pd.Categorical(time_data['day_of_week'], categories=day_order, ordered=True)

                    # Location filter dropdown
                    available_locations = sorted(time_data['location'].unique().tolist())
                    location_options = ['All Locations'] + available_locations

                    selected_heatmap_location = st.selectbox(
                        'Select Location',
                        options=location_options,
                        index=0,
                        key='heatmap_location_selector'
                    )

                    # Filter data by selected location
                    if selected_heatmap_location == 'All Locations':
                        heatmap_data = time_data
                    else:
                        heatmap_data = time_data[time_data['location'] == selected_heatmap_location]

                    # Group by day and hour
                    visit_counts = heatmap_data.groupby(['day_of_week', 'visit_hour'], observed=True).agg({
                        'booking_id': 'count'
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

                    with st.expander("How to Read the Heatmap"):
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

                    st.dataframe(temp_stats, use_container_width=True)

                    # Insights
                    st.info("""
                    **How to read the bubble plot:**
                    - **Horizontal position** (X-axis): Average temperature when booking was made
                    - **Vertical position** (Y-axis): How far in advance customers book
                    - **Bubble size**: Share of total bookings (larger = more bookings)
                    - **Pattern**: Warmer weather typically correlates with more advance planning

                    **Interpretation:**
                    - **Large bubbles high on chart**: Popular temperature range where customers plan ahead
                    - **Large bubbles low on chart**: Popular temperature range with spontaneous bookings
                    - **Small bubbles**: Temperature ranges with less sauna demand
                    """)

            # Recurring Customer Analysis
            if email_col != "None":
                st.markdown("---")
                st.markdown("### Recurring Customer Analysis")

                st.markdown("""
                Analyze customer loyalty and repeat visit patterns using email addresses.
                Understand which customers return and how frequently they book.
                """)

                # Prepare customer data
                customer_data = df1[[id_col_1, email_col]].copy()
                customer_data.columns = ['booking_id', 'email']

                # Add location if available
                if location_col != "None":
                    customer_data['location'] = df1[location_col]

                # Remove rows with missing emails
                customer_data = customer_data[customer_data['email'].notna() & (customer_data['email'] != '')]

                # Count bookings per customer
                customer_frequency = customer_data.groupby('email').agg({
                    'booking_id': 'count'
                }).reset_index()
                customer_frequency.columns = ['email', 'bookings']

                # Categorize into tiers
                def categorize_customer(bookings):
                    if bookings == 1:
                        return "One-time"
                    elif 2 <= bookings <= 3:
                        return "Light (2-3)"
                    elif 4 <= bookings <= 6:
                        return "Regular (4-6)"
                    elif 7 <= bookings <= 10:
                        return "Frequent (7-10)"
                    else:
                        return "VIP (11+)"

                customer_frequency['tier'] = customer_frequency['bookings'].apply(categorize_customer)

                # Tier order for consistent display
                tier_order = ['One-time', 'Light (2-3)', 'Regular (4-6)', 'Frequent (7-10)', 'VIP (11+)']
                customer_frequency['tier'] = pd.Categorical(customer_frequency['tier'], categories=tier_order, ordered=True)

                # Customer tier distribution
                tier_distribution = customer_frequency['tier'].value_counts().reindex(tier_order, fill_value=0).reset_index()
                tier_distribution.columns = ['Tier', 'Customers']

                # Calculate metrics
                total_customers = len(customer_frequency)
                recurring_customers = len(customer_frequency[customer_frequency['bookings'] > 1])
                recurring_pct = (recurring_customers / total_customers * 100) if total_customers > 0 else 0

                # Calculate percentage of total for each tier
                tier_distribution['Percentage'] = (tier_distribution['Customers'] / total_customers * 100).round(1)

                # Create text labels with percentage
                tier_distribution['label'] = tier_distribution['Percentage'].apply(lambda x: f"{x}%")

                # Display key metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Unique Customers", f"{total_customers:,}")
                with col2:
                    st.metric("Recurring Customers", f"{recurring_customers:,}")
                with col3:
                    st.metric("Recurring Rate", f"{recurring_pct:.1f}%")

                # Customer tier chart
                fig_tiers = px.bar(
                    tier_distribution,
                    x='Tier',
                    y='Percentage',
                    title='Customer Distribution by Tier',
                    labels={'Tier': 'Customer Tier', 'Percentage': 'Percentage of Total Customers'},
                    text='label'
                )

                fig_tiers.update_traces(
                    marker_color='#1f77b4',
                    textposition='outside'
                )

                fig_tiers.update_layout(
                    height=500,
                    showlegend=False,
                    xaxis=dict(tickangle=0),
                    yaxis=dict(range=[0, 100])
                )

                st.plotly_chart(fig_tiers, use_container_width=True)

                # Location loyalty analysis (only if location is selected)
                if location_col != "None":
                    st.markdown("#### Location Loyalty Among Recurring Customers")

                    # For recurring customers, count how many locations they visit
                    recurring_customer_data = customer_data[customer_data['email'].isin(
                        customer_frequency[customer_frequency['bookings'] > 1]['email']
                    )]

                    locations_per_customer = recurring_customer_data.groupby('email')['location'].nunique().reset_index()
                    locations_per_customer.columns = ['email', 'locations_visited']

                    # Categorize loyalty
                    def categorize_loyalty(num_locations):
                        if num_locations == 1:
                            return "Single location"
                        elif num_locations == 2:
                            return "2 locations"
                        else:
                            return "3+ locations"

                    locations_per_customer['loyalty_type'] = locations_per_customer['locations_visited'].apply(categorize_loyalty)

                    loyalty_distribution = locations_per_customer['loyalty_type'].value_counts().reset_index()
                    loyalty_distribution.columns = ['Loyalty Type', 'Customers']

                    # Reorder for consistent display
                    loyalty_order = ['Single location', '2 locations', '3+ locations']
                    loyalty_distribution['Loyalty Type'] = pd.Categorical(
                        loyalty_distribution['Loyalty Type'],
                        categories=loyalty_order,
                        ordered=True
                    )
                    loyalty_distribution = loyalty_distribution.sort_values('Loyalty Type')

                    # Side-by-side layout: 60% chart, 40% table
                    col1, col2 = st.columns([3, 2])

                    with col1:
                        # Loyalty donut chart
                        fig_loyalty = px.pie(
                            loyalty_distribution,
                            values='Customers',
                            names='Loyalty Type',
                            hole=0.4
                        )

                        fig_loyalty.update_layout(
                            height=400,
                            showlegend=True,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=-0.2,
                                xanchor="center",
                                x=0.5
                            ),
                            margin=dict(t=20, b=80)
                        )
                        st.plotly_chart(fig_loyalty, use_container_width=True)

                    with col2:
                        st.dataframe(loyalty_distribution, use_container_width=True, hide_index=True)

                    # Recurring customers per location
                    st.markdown("#### Recurring Customers by Location")

                    # Get total customers per location (all customers, not just recurring)
                    location_total = customer_data.groupby('location')['email'].nunique().reset_index()
                    location_total.columns = ['Location', 'Total Customers']

                    # Get recurring customers per location
                    location_recurring = recurring_customer_data.groupby('location')['email'].nunique().reset_index()
                    location_recurring.columns = ['Location', 'Recurring Customers']

                    # Merge and calculate recurring rate
                    location_stats = location_total.merge(location_recurring, on='Location')
                    location_stats['Recurring Rate (%)'] = (location_stats['Recurring Customers'] / location_stats['Total Customers'] * 100).round(1)

                    # Keep only the columns we want to display
                    location_display = location_stats[['Location', 'Recurring Customers', 'Recurring Rate (%)']].copy()
                    location_display = location_display.sort_values('Recurring Customers', ascending=False)

                    # Add total row with average recurring rate
                    total_recurring = location_display['Recurring Customers'].sum()
                    avg_recurring_rate = location_stats['Recurring Rate (%)'].mean().round(1)

                    total_row = pd.DataFrame({
                        'Location': ['Total'],
                        'Recurring Customers': [total_recurring],
                        'Recurring Rate (%)': [avg_recurring_rate]
                    })
                    location_display = pd.concat([location_display, total_row], ignore_index=True)

                    st.dataframe(location_display, use_container_width=True, hide_index=True)

                # Insights
                st.info("""
                **Understanding Customer Tiers:**
                - **One-time**: Customers who have made only 1 booking (potential for conversion)
                - **Light (2-3 bookings)**: Customers returning occasionally
                - **Regular (4-6 bookings)**: Customers with established booking patterns
                - **Frequent (7-10 bookings)**: Loyal customers who visit regularly
                - **VIP (11+ bookings)**: Your most loyal customer base

                **Location Loyalty Insights:**
                - **Single location**: Brand loyal to specific location (consider location-specific retention programs)
                - **2 locations**: Cross-location customers (appreciate variety)
                - **3+ locations**: True Kuuma fans exploring all offerings

                **Strategic Actions:**
                - Target one-time customers with re-engagement campaigns
                - Reward VIP and Frequent tiers with loyalty benefits
                - Encourage single-location customers to try other branches
                """)

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
       - Locations (optional)

    3. **View insights**:
       - Average and median booking lead times
       - Distribution charts
       - Trends over time
       - Location comparisons

    4. **Export** your results as CSV for further analysis
    """)
