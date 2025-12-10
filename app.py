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
    st.markdown("**Analyze the time between booking creation and customer visit**")

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
            fig_dist.update_layout(showlegend=False, height=400)

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
                    st.markdown("### Sauna Visit Patterns by Day & Time")

                    st.markdown("""
                    Understand when customers actually visit the sauna throughout the week. This helps with:
                    - **Staffing optimization** during peak visit times
                    - **Capacity planning** for high-demand periods
                    - **Day-of-week patterns** to inform operational scheduling
                    """)

                    # Create copy for time analysis to avoid modifying filtered_data
                    time_data = filtered_data.copy()
                    time_data['visit_hour'] = time_data['visit_date'].dt.hour
                    time_data['visit_minute'] = time_data['visit_date'].dt.minute
                    time_data['day_of_week'] = time_data['visit_date'].dt.day_name()

                    # Order days of week properly
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    time_data['day_of_week'] = pd.Categorical(time_data['day_of_week'], categories=day_order, ordered=True)

                    # Calculate visits by day of week and location
                    day_stats = time_data.groupby(['location', 'day_of_week'], observed=True).agg({
                        'booking_id': 'count'
                    }).reset_index()
                    day_stats.columns = ['Location', 'Day of Week', 'Visits']

                    # Create grouped bar chart for day of week patterns
                    fig_weekly = px.bar(
                        day_stats,
                        x='Day of Week',
                        y='Visits',
                        color='Location',
                        barmode='group',
                        title='Sauna Visits by Day of Week',
                        labels={'Day of Week': 'Day of Week', 'Visits': 'Number of Visits'},
                        category_orders={'Day of Week': day_order}
                    )

                    fig_weekly.update_layout(
                        height=450,
                        hovermode='x unified',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )

                    st.plotly_chart(fig_weekly, use_container_width=True)

                    # Calculate average visit time per location per day of week
                    st.markdown("#### Average Visit Time by Location and Day")

                    visit_time_summary = []
                    for location in time_data['location'].unique():
                        for day in day_order:
                            loc_day_data = time_data[(time_data['location'] == location) & (time_data['day_of_week'] == day)]
                            if len(loc_day_data) > 0:
                                total_minutes = (loc_day_data['visit_hour'] * 60 + loc_day_data['visit_minute']).mean()
                                avg_hour = int(total_minutes // 60)
                                avg_min = int(total_minutes % 60)
                                avg_time = f"{avg_hour:02d}:{avg_min:02d}"

                                visit_time_summary.append({
                                    'Location': location,
                                    'Day': day,
                                    'Avg Visit Time': avg_time,
                                    'Total Visits': len(loc_day_data)
                                })

                    visit_time_df = pd.DataFrame(visit_time_summary)

                    # Pivot for better readability
                    visit_time_pivot = visit_time_df.pivot(
                        index='Location',
                        columns='Day',
                        values='Avg Visit Time'
                    )
                    # Reorder columns by day of week
                    visit_time_pivot = visit_time_pivot[day_order]

                    st.dataframe(visit_time_pivot, use_container_width=True)

                    # Hourly distribution across all days
                    st.markdown("#### Visit Volume by Hour of Day")

                    hourly_dist = time_data.groupby(['location', 'visit_hour']).agg({
                        'booking_id': 'count'
                    }).reset_index()
                    hourly_dist.columns = ['Location', 'Hour', 'Visits']

                    # Ensure all hours 0-23 are represented
                    all_combinations = pd.MultiIndex.from_product(
                        [time_data['location'].unique(), range(24)],
                        names=['Location', 'Hour']
                    ).to_frame(index=False)
                    hourly_dist = all_combinations.merge(hourly_dist, on=['Location', 'Hour'], how='left').fillna(0)
                    hourly_dist['Visits'] = hourly_dist['Visits'].astype(int)

                    # Create grouped bar chart
                    fig_hourly = px.bar(
                        hourly_dist,
                        x='Hour',
                        y='Visits',
                        color='Location',
                        barmode='group',
                        title='Visit Volume Throughout the Day',
                        labels={'Hour': 'Hour of Day (24-hour format)', 'Visits': 'Number of Visits'}
                    )

                    fig_hourly.update_layout(
                        height=450,
                        xaxis=dict(
                            tickmode='linear',
                            tick0=0,
                            dtick=2,
                            ticksuffix=':00'
                        ),
                        hovermode='x unified',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )

                    st.plotly_chart(fig_hourly, use_container_width=True)

                    # Calculate peak visit times
                    overall_peak_day = day_stats.groupby('Day of Week', observed=True)['Visits'].sum().idxmax()
                    overall_peak_hour = hourly_dist.groupby('Hour')['Visits'].sum().idxmax()
                    peak_hour_count = int(hourly_dist.groupby('Hour')['Visits'].sum().max())

                    st.info(f"""
**Busiest Day**: {overall_peak_day} has the most sauna visits across all locations.
**Peak Visit Hour**: {overall_peak_hour:02d}:00 - {(overall_peak_hour+1):02d}:00 with {peak_hour_count} visits.
Consider ensuring adequate staffing and capacity during these peak periods.
                    """)

                    with st.expander("Understanding Visit Patterns"):
                        st.markdown("""
**Day of Week Chart** shows which days are busiest for actual sauna visits at each location. This helps identify weekend vs. weekday patterns.

**Average Visit Time Table** displays the typical time of day people visit on each day of the week, per location. Use this to identify patterns like "Saturday evening rush" or "Monday morning crowd".

**Hourly Distribution** shows visit volume across all 24 hours, helping identify peak times for staffing needs.

**Operational Insights:**
- Peak days and hours indicate when to maximize staff presence and sauna capacity
- Off-peak times might be good for maintenance or special promotions
- Location differences reveal customer demographics (e.g., business district vs. residential area patterns)
- Day-of-week patterns help optimize weekly scheduling and resource allocation
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

                    # Create dual-axis chart
                    fig_temp = go.Figure()

                    # Add bar chart for number of bookings
                    fig_temp.add_trace(go.Bar(
                        x=temp_analysis['temp_category'],
                        y=temp_analysis['Number of Bookings'],
                        name='Number of Bookings',
                        marker_color='lightblue',
                        yaxis='y'
                    ))

                    # Add line chart for average lead time
                    fig_temp.add_trace(go.Scatter(
                        x=temp_analysis['temp_category'],
                        y=temp_analysis['Avg Lead Time (days)'],
                        name='Avg Lead Time',
                        mode='lines+markers',
                        marker=dict(size=10, color='red'),
                        line=dict(width=3, color='red'),
                        yaxis='y2'
                    ))

                    # Update layout with dual y-axes
                    fig_temp.update_layout(
                        title="Booking Volume & Lead Time by Temperature",
                        xaxis=dict(title='Temperature at Booking Time'),
                        yaxis=dict(
                            title='Number of Bookings',
                            side='left'
                        ),
                        yaxis2=dict(
                            title='Average Lead Time (days)',
                            side='right',
                            overlaying='y'
                        ),
                        hovermode='x unified',
                        height=500,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
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
                    **How to interpret:**
                    - **High bookings + short lead time**: Customers book spontaneously in this temperature range
                    - **High bookings + long lead time**: Customers plan ahead for sauna visits in this weather
                    - **Low bookings**: This temperature range sees less sauna demand
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
