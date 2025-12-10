import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

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
    st.title("Kuuma Booking Interval Analyzer")
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
        st.sidebar.success(f"✓ File 1 loaded: {len(df1):,} rows")

# Load File 2
if uploaded_file2 is not None:
    df2, error2 = load_excel_file(uploaded_file2)
    if error2:
        st.sidebar.error(f"Error reading File 2: {error2}")
    else:
        st.session_state.df2 = df2
        st.sidebar.success(f"✓ File 2 loaded: {len(df2):,} rows")

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

    # Process the data
    with st.spinner("Processing data..."):
        processed_data, unmatched, invalid, negative = process_data(
            df1, df2, id_col_1, date_col_1, id_col_2, visit_col_2, location_col
        )

    # Check if we have data to display
    if len(processed_data) == 0:
        st.error("⚠️ No matching booking IDs found between files. Please check your column selections.")
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
                st.metric("Average Lead Time", f"{avg_interval:.1f} days")

            with col2:
                st.metric("Median Lead Time", f"{median_interval:.1f} days")

            with col3:
                st.metric("Total Bookings", f"{total_bookings:,}")

            with col4:
                st.metric("Same-Day Bookings", f"{same_day_pct:.1f}%")

            # Explanation box
            with st.expander("ℹ️ What does 'Lead Time' mean?", expanded=False):
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

            # Data quality summary
            st.info(f"📊 **Data Quality:** {len(processed_data):,} matched records | "
                   f"{unmatched:,} unmatched | {invalid:,} invalid dates | {negative:,} negative intervals")

            # Distribution chart
            st.markdown("### Booking Lead Time Distribution")

            category_order = ["Same day", "1-3 days", "4-7 days", "1-2 weeks", "2+ weeks"]
            distribution = filtered_data['interval_category'].value_counts()
            distribution = distribution.reindex(category_order, fill_value=0)

            fig_dist = px.bar(
                x=distribution.index,
                y=distribution.values,
                labels={'x': 'Lead Time', 'y': 'Number of Bookings'},
                title="How far in advance do customers book?"
            )
            fig_dist.update_traces(marker_color='#1f77b4', text=distribution.values, textposition='outside')
            fig_dist.update_layout(showlegend=False, height=400)

            st.plotly_chart(fig_dist, use_container_width=True)

            # Trend chart
            st.markdown("### Average Lead Time Over Time")

            st.markdown("""
            This chart shows how booking behavior changes over time. Each point represents the average lead time for bookings in that week.

            **What to look for:**
            - **Upward trends**: Customers are booking further in advance
            - **Downward trends**: Customers are booking more spontaneously (last-minute)
            - **Seasonal patterns**: Do customers plan more during certain periods?
            - **Sudden changes**: Could indicate marketing campaigns, events, or external factors
            """)

            # Group by week
            filtered_data_sorted = filtered_data.copy()
            filtered_data_sorted['week'] = filtered_data_sorted['visit_date'].dt.to_period('W').dt.start_time

            trend_data = filtered_data_sorted.groupby('week')['interval_days'].mean().reset_index()
            trend_data.columns = ['Week', 'Average Interval']

            if len(trend_data) > 1:
                fig_trend = px.line(
                    trend_data,
                    x='Week',
                    y='Average Interval',
                    labels={'Week': 'Week', 'Average Interval': 'Average Lead Time (days)'},
                    title="Booking lead time trends"
                )
                fig_trend.update_traces(mode='lines+markers', marker=dict(size=6))
                fig_trend.update_layout(height=400)

                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("Not enough data to show trend over time. Upload data spanning multiple weeks.")

            # Location breakdown
            if location_col != "None" and len(selected_locations) > 0:
                st.markdown("### Breakdown by Location")

                location_stats = filtered_data.groupby('location').agg({
                    'booking_id': 'count',
                    'interval_days': ['mean', 'median']
                }).round(1)

                location_stats.columns = ['Total Bookings', 'Avg Lead Time (days)', 'Median Lead Time (days)']
                location_stats = location_stats.sort_values('Total Bookings', ascending=False)
                location_stats['Total Bookings'] = location_stats['Total Bookings'].astype(int)

                st.dataframe(location_stats, use_container_width=True)

            # Export functionality
            st.sidebar.markdown("---")
            st.sidebar.subheader("Export")

            export_data = filtered_data[['booking_id', 'booking_date', 'visit_date', 'interval_days']].copy()
            if location_col != "None":
                export_data['location'] = filtered_data['location']

            export_data.columns = ['Booking ID', 'Booking Date', 'Visit Date', 'Interval (Days)'] + (['Location'] if location_col != "None" else [])

            csv = export_data.to_csv(index=False)

            st.sidebar.download_button(
                label="📥 Download Results (CSV)",
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
            st.warning("⚠️ No data matches the selected filters. Try adjusting your date range or location selection.")

    # Reset button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Clear All & Start Over"):
        st.session_state.df1 = None
        st.session_state.df2 = None
        st.rerun()

else:
    # Welcome message
    st.info("👆 **Get started:** Upload both Excel files in the sidebar to begin analysis")

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
