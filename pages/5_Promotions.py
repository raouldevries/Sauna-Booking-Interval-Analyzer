"""
Kuuma Booking Analyzer - Promotions Page
Promotion effectiveness and discount analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Kuuma - Promotions",
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

# Header with logo
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://kuuma.nl/wp-content/themes/kuuma/images/logo.svg", width=120)
with col2:
    st.title("Kuuma Booking Analyzer")
    st.markdown("**Customer insights & booking intelligence**")

# Reserve container for date range selector (filled after data loads)
date_range_container = st.container()

st.markdown("## Promotion Effectiveness")
st.markdown("Analyze promotion usage, revenue impact, and conversion rates")

# Initialize session state
if 'df1' not in st.session_state:
    st.session_state.df1 = None
if 'df2' not in st.session_state:
    st.session_state.df2 = None

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

# Reserve container for navigation at top of sidebar
nav_container = st.sidebar.container()

# Sidebar - Upload section
st.sidebar.header("Upload & Configure")

# File uploaders with multiple file support
uploaded_files1 = st.sidebar.file_uploader(
    "Booking Creation Dates (.xls/.xlsx)",
    type=["xls", "xlsx"],
    help="Upload files containing when bookings were created. You can select multiple files from different Bookeo instances.",
    key="promo_file1",
    accept_multiple_files=True
)

uploaded_files2 = st.sidebar.file_uploader(
    "Visit Dates (.xls/.xlsx)",
    type=["xls", "xlsx"],
    help="Upload files containing when customers actually visited. You can select multiple files from different Bookeo instances.",
    key="promo_file2",
    accept_multiple_files=True
)

# Load and merge File 1
if uploaded_files1:
    df1, errors1, file_info1 = load_and_merge_files(uploaded_files1)
    if errors1:
        for error in errors1:
            st.sidebar.error(f"Error: {error}")
    elif df1 is not None:
        st.session_state.df1 = df1
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
        st.session_state.df2 = df2
        if len(file_info2) == 1:
            st.sidebar.success(f"Loaded: {len(df2):,} rows")
        else:
            st.sidebar.success(f"Merged {len(file_info2)} files: {len(df2):,} total rows")

# Fill navigation container (now that files are loaded)
if st.session_state.df1 is not None and st.session_state.df2 is not None:
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
if st.session_state.df1 is None or st.session_state.df2 is None:
    st.info("**No data loaded.** Please upload your Excel files using the sidebar to begin analysis.")
    st.markdown("""
    ### Getting Started
    1. Upload your booking creation dates file
    2. Upload your visit dates file
    3. Configure column mappings
    4. Explore promotion insights!
    """)
else:
    df1 = st.session_state.df1
    df2 = st.session_state.df2

    # Column mapping
    st.sidebar.markdown("---")
    st.sidebar.subheader("Column Mapping")

    # Smart defaults
    default_id_col = "Booking number" if "Booking number" in df1.columns else df1.columns[0]
    default_created_col = "Created" if "Created" in df1.columns else df1.columns[0]
    default_start_col = "Start" if "Start" in df2.columns else df2.columns[0]

    if "Location" in df1.columns:
        default_location_col = "Location"
    elif "Tour" in df1.columns:
        default_location_col = "Tour"
    elif "Activity" in df1.columns:
        default_location_col = "Activity"
    else:
        default_location_col = None

    id_col_1 = st.sidebar.selectbox(
        "Booking ID (File 1)",
        options=df1.columns.tolist(),
        index=df1.columns.tolist().index(default_id_col) if default_id_col in df1.columns else 0,
        key="promo_id_col_1"
    )

    date_col_1 = st.sidebar.selectbox(
        "Booking Creation Date (File 1)",
        options=df1.columns.tolist(),
        index=df1.columns.tolist().index(default_created_col) if default_created_col in df1.columns else 0,
        key="promo_date_col_1"
    )

    id_col_2 = st.sidebar.selectbox(
        "Booking ID (File 2)",
        options=df2.columns.tolist(),
        index=df2.columns.tolist().index(default_id_col) if default_id_col in df2.columns else 0,
        key="promo_id_col_2"
    )

    visit_col_2 = st.sidebar.selectbox(
        "Visit Date (File 2)",
        options=df2.columns.tolist(),
        index=df2.columns.tolist().index(default_start_col) if default_start_col in df2.columns else 0,
        key="promo_visit_col_2"
    )

    # Location column (optional)
    location_options = ["None"] + df1.columns.tolist()
    location_default_index = location_options.index(default_location_col) if default_location_col in location_options else 0

    location_col = st.sidebar.selectbox(
        "Location (optional)",
        options=location_options,
        index=location_default_index,
        key="promo_location_col"
    )

    # Revenue column
    default_revenue_col = "Total gross" if "Total gross" in df1.columns else None
    revenue_options = ["None"] + df1.columns.tolist()
    revenue_default_index = revenue_options.index(default_revenue_col) if default_revenue_col in revenue_options else 0

    revenue_col = st.sidebar.selectbox(
        "Revenue Column",
        options=revenue_options,
        index=revenue_default_index,
        help="Column containing booking revenue/price",
        key="promo_revenue_col"
    )

    # Promotion column
    default_promo_col = "Promotion" if "Promotion" in df1.columns else None
    promo_options = ["None"] + df1.columns.tolist()
    promo_default_index = promo_options.index(default_promo_col) if default_promo_col in promo_options else 0

    promotion_col = st.sidebar.selectbox(
        "Promotion Column",
        options=promo_options,
        index=promo_default_index,
        help="Column containing promotion/discount info",
        key="promo_promo_col"
    )

    # Check required columns
    if promotion_col == "None":
        st.warning("""
        **Promotion column not configured.**

        To use promotion analysis, please select a Promotion Column in the sidebar (e.g., "Promotion").
        """)
    elif revenue_col == "None":
        st.warning("""
        **Revenue column not configured.**

        To analyze promotion effectiveness, please select a Revenue Column in the sidebar (e.g., "Total gross").
        """)
    else:
        # Process data
        @st.cache_data
        def process_promo_data(df1, df2, id_col_1, date_col_1, id_col_2, visit_col_2,
                               location_col, revenue_col, promotion_col):
            # Prepare dataframes
            cols_to_use = [id_col_1, date_col_1]
            df1_prep = df1[cols_to_use].copy()
            df1_prep.columns = ['booking_id', 'booking_date']

            df2_prep = df2[[id_col_2, visit_col_2]].copy()
            df2_prep.columns = ['booking_id', 'visit_date']

            # Add optional columns
            if location_col != "None":
                df1_prep['location'] = df1[location_col].values

            if revenue_col != "None":
                df1_prep['revenue'] = pd.to_numeric(df1[revenue_col], errors='coerce').fillna(0)

            if promotion_col != "None":
                df1_prep['promotion'] = df1[promotion_col].values

            # Merge on booking ID
            merged = df1_prep.merge(df2_prep, on='booking_id', how='inner')

            # Convert dates
            merged['booking_date'] = pd.to_datetime(merged['booking_date'], errors='coerce')
            merged['visit_date'] = pd.to_datetime(merged['visit_date'], errors='coerce')

            # Filter invalid records
            invalid_dates = merged['booking_date'].isna() | merged['visit_date'].isna()
            merged_clean = merged[~invalid_dates].copy()

            return merged_clean

        with st.spinner("Processing data..."):
            processed_data = process_promo_data(
                df1, df2, id_col_1, date_col_1, id_col_2, visit_col_2,
                location_col, revenue_col, promotion_col
            )

        if len(processed_data) == 0:
            st.error("No matching booking IDs found between files.")
        else:
            # Date range selector in reserved container (under header)
            min_date = processed_data['visit_date'].min().date()
            max_date = processed_data['visit_date'].max().date()

            with date_range_container:
                date_col1, date_col2 = st.columns([2, 4])
                with date_col1:
                    date_range = st.date_input(
                        "Date Range (Visit Date)",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                        help="Filter promotions by visit date",
                        key="promo_date_range"
                    )

            # Apply date filter
            if len(date_range) == 2:
                start_date, end_date = date_range
                processed_data = processed_data[
                    (processed_data['visit_date'].dt.date >= start_date) &
                    (processed_data['visit_date'].dt.date <= end_date)
                ]

            # Clean promotion data
            promo_data = processed_data.copy()
            promo_data['has_promotion'] = promo_data['promotion'].notna() & (promo_data['promotion'] != '')

            # Summary metrics
            with_promo = promo_data[promo_data['has_promotion']]
            without_promo = promo_data[~promo_data['has_promotion']]
            total_revenue = promo_data['revenue'].sum()

            # Key metrics
            st.markdown("### Overview")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                promo_pct = len(with_promo) / len(promo_data) * 100
                st.metric(
                    "Bookings with Promotion",
                    f"{len(with_promo):,}",
                    delta=f"{promo_pct:.1f}% of total",
                    help="Number of bookings that used a promotion code."
                )
            with col2:
                avg_with = with_promo['revenue'].mean() if len(with_promo) > 0 else 0
                avg_without = without_promo['revenue'].mean() if len(without_promo) > 0 else 0
                diff = avg_with - avg_without
                st.metric(
                    "Avg Booking (with promo)",
                    f"€{avg_with:.2f}",
                    delta=f"€{diff:.2f} vs no promo",
                    help="Average revenue for bookings that used a promotion."
                )
            with col3:
                st.metric(
                    "Avg Booking (no promo)",
                    f"€{avg_without:.2f}",
                    help="Average revenue for bookings without a promotion."
                )
            with col4:
                promo_revenue = with_promo['revenue'].sum()
                promo_rev_pct = promo_revenue / total_revenue * 100 if total_revenue > 0 else 0
                st.metric(
                    "Revenue from Promotions",
                    f"€{promo_revenue:,.0f}",
                    delta=f"{promo_rev_pct:.1f}% of total",
                    help="Total revenue generated from bookings with promotions."
                )

            # Promotion breakdown
            if len(with_promo) > 0:
                st.markdown("---")
                st.markdown("### Performance by Promotion")

                promo_stats = with_promo.groupby('promotion').agg({
                    'booking_id': 'count',
                    'revenue': ['sum', 'mean']
                }).round(2)
                promo_stats.columns = ['Bookings', 'Total Revenue', 'Avg Booking']
                promo_stats = promo_stats.sort_values('Bookings', ascending=False)
                promo_stats['% of Promo Bookings'] = (promo_stats['Bookings'] / len(with_promo) * 100).round(1)
                promo_stats['% of Promo Revenue'] = (promo_stats['Total Revenue'] / promo_revenue * 100).round(1)

                # Format for display
                promo_display = promo_stats.copy()
                promo_display['Total Revenue'] = promo_display['Total Revenue'].apply(lambda x: f"€{x:,.0f}")
                promo_display['Avg Booking'] = promo_display['Avg Booking'].apply(lambda x: f"€{x:.2f}")

                promo_config = {
                    'Bookings': st.column_config.NumberColumn('Bookings', help='Number of bookings using this promotion'),
                    '% of Promo Bookings': st.column_config.NumberColumn('% of Promo Bookings', help='Share of all promotional bookings'),
                    'Total Revenue': st.column_config.TextColumn('Total Revenue', help='Revenue from this promotion'),
                    '% of Promo Revenue': st.column_config.NumberColumn('% of Promo Revenue', help='Share of all promotional revenue'),
                    'Avg Booking': st.column_config.TextColumn('Avg Booking', help='Average booking value with this promotion'),
                }
                st.dataframe(
                    promo_display[['Bookings', '% of Promo Bookings', 'Total Revenue', '% of Promo Revenue', 'Avg Booking']],
                    use_container_width=True, column_config=promo_config
                )

                # Top promotions chart
                st.markdown("### Top Promotions by Usage")

                top_promos = promo_stats.head(10).reset_index()

                fig_promo = px.bar(
                    top_promos,
                    x='promotion',
                    y='Bookings',
                    title="Top 10 Promotions by Number of Bookings",
                    labels={'promotion': 'Promotion', 'Bookings': 'Number of Bookings'},
                    text='Bookings'
                )
                fig_promo.update_traces(marker_color='#e74c3c', textposition='outside')
                fig_promo.update_layout(height=450, xaxis_tickangle=-45)
                st.plotly_chart(fig_promo, use_container_width=True)

                # Top promotions by revenue
                st.markdown("### Top Promotions by Revenue")

                top_promos_rev = promo_stats.sort_values('Total Revenue', ascending=False).head(10).reset_index()

                fig_promo_rev = px.bar(
                    top_promos_rev,
                    x='promotion',
                    y='Total Revenue',
                    title="Top 10 Promotions by Revenue Generated",
                    labels={'promotion': 'Promotion', 'Total Revenue': 'Revenue (€)'},
                    text=top_promos_rev['Total Revenue'].apply(lambda x: f"€{x:,.0f}")
                )
                fig_promo_rev.update_traces(marker_color='#2ecc71', textposition='outside')
                fig_promo_rev.update_layout(height=450, xaxis_tickangle=-45)
                st.plotly_chart(fig_promo_rev, use_container_width=True)

                # Comparison charts
                st.markdown("---")
                st.markdown("### Promo vs No Promo Comparison")

                comparison_data = pd.DataFrame({
                    'Category': ['With Promotion', 'Without Promotion'],
                    'Avg Booking Value': [avg_with, avg_without],
                    'Bookings': [len(with_promo), len(without_promo)],
                    'Total Revenue': [promo_revenue, without_promo['revenue'].sum()]
                })

                col1, col2 = st.columns(2)

                with col1:
                    fig_comp_val = px.bar(
                        comparison_data,
                        x='Category',
                        y='Avg Booking Value',
                        title="Average Booking Value",
                        labels={'Avg Booking Value': 'Avg Value (€)'},
                        text=comparison_data['Avg Booking Value'].apply(lambda x: f"€{x:.2f}")
                    )
                    fig_comp_val.update_traces(marker_color=['#e74c3c', '#3498db'], textposition='outside')
                    fig_comp_val.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_comp_val, use_container_width=True)

                with col2:
                    fig_comp_book = px.pie(
                        comparison_data,
                        values='Bookings',
                        names='Category',
                        title="Booking Distribution",
                        color_discrete_sequence=['#e74c3c', '#3498db']
                    )
                    fig_comp_book.update_layout(height=400)
                    st.plotly_chart(fig_comp_book, use_container_width=True)

                # Revenue comparison
                col1, col2 = st.columns(2)

                with col1:
                    fig_rev_comp = px.pie(
                        comparison_data,
                        values='Total Revenue',
                        names='Category',
                        title="Revenue Distribution",
                        color_discrete_sequence=['#e74c3c', '#3498db']
                    )
                    fig_rev_comp.update_layout(height=400)
                    st.plotly_chart(fig_rev_comp, use_container_width=True)

                with col2:
                    fig_rev_bar = px.bar(
                        comparison_data,
                        x='Category',
                        y='Total Revenue',
                        title="Total Revenue",
                        labels={'Total Revenue': 'Revenue (€)'},
                        text=comparison_data['Total Revenue'].apply(lambda x: f"€{x:,.0f}")
                    )
                    fig_rev_bar.update_traces(marker_color=['#e74c3c', '#3498db'], textposition='outside')
                    fig_rev_bar.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_rev_bar, use_container_width=True)

                # Promotion by location (if available)
                if location_col != "None" and 'location' in with_promo.columns:
                    st.markdown("---")
                    st.markdown("### Promotion Usage by Location")

                    location_promo = promo_data.groupby('location').agg({
                        'booking_id': 'count',
                        'has_promotion': 'sum',
                        'revenue': 'sum'
                    }).reset_index()
                    location_promo.columns = ['Location', 'Total Bookings', 'Promo Bookings', 'Total Revenue']
                    location_promo['Promo Rate (%)'] = (location_promo['Promo Bookings'] / location_promo['Total Bookings'] * 100).round(1)
                    location_promo = location_promo.sort_values('Promo Bookings', ascending=False)

                    loc_promo_config = {
                        'Location': st.column_config.TextColumn('Location'),
                        'Total Bookings': st.column_config.NumberColumn('Total Bookings', help='All bookings at this location'),
                        'Promo Bookings': st.column_config.NumberColumn('Promo Bookings', help='Bookings that used a promotion'),
                        'Total Revenue': st.column_config.NumberColumn('Total Revenue', help='Revenue from all bookings'),
                        'Promo Rate (%)': st.column_config.NumberColumn('Promo Rate (%)', help='Percentage of bookings using promotions'),
                    }
                    st.dataframe(location_promo, use_container_width=True, hide_index=True, column_config=loc_promo_config)

                    fig_loc_promo = px.bar(
                        location_promo,
                        x='Location',
                        y='Promo Rate (%)',
                        title="Promotion Usage Rate by Location",
                        labels={'Promo Rate (%)': 'Promo Rate (%)'},
                        text='Promo Rate (%)'
                    )
                    fig_loc_promo.update_traces(marker_color='#9b59b6', textposition='outside')
                    fig_loc_promo.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig_loc_promo, use_container_width=True)

                # Insights
                st.markdown("---")
                top_promo = promo_stats.index[0]
                top_promo_bookings = promo_stats.iloc[0]['Bookings']
                top_promo_pct = promo_stats.iloc[0]['% of Promo Bookings']

                st.info(f"""
**Promotion Insights:**

- **Most Popular Promotion:** "{top_promo}" with {top_promo_bookings:,.0f} bookings ({top_promo_pct:.1f}% of all promo bookings)
- **Average Booking Difference:** Promo bookings average €{diff:.2f} {'less' if diff < 0 else 'more'} than non-promo bookings
- **Revenue Impact:** {promo_rev_pct:.1f}% of total revenue comes from promotional bookings

**Strategic Considerations:**
- {'Promotions are driving lower-value bookings - consider if volume compensates for lower margins' if diff < 0 else 'Promotions are associated with higher-value bookings - effective upselling'}
- Monitor which promotions drive repeat customers vs one-time visits
- Consider location-specific promotions based on usage patterns
                """)

            else:
                st.info("No bookings with promotions found in the selected data.")

    # Reset button
    st.sidebar.markdown("---")
    if st.sidebar.button("Clear All & Start Over", key="promo_reset"):
        st.session_state.df1 = None
        st.session_state.df2 = None
        st.rerun()
