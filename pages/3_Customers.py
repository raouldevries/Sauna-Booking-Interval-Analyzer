"""
Kuuma Booking Analyzer - Customers Page
Recurring customer analysis, tiers, and loyalty
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Kuuma - Customers",
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

st.markdown("## Recurring Customer Analysis")
st.markdown("Customer segmentation, loyalty tiers, and retention insights")

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
    key="cust_file1",
    accept_multiple_files=True
)

uploaded_files2 = st.sidebar.file_uploader(
    "Visit Dates (.xls/.xlsx)",
    type=["xls", "xlsx"],
    help="Upload files containing when customers actually visited. You can select multiple files from different Bookeo instances.",
    key="cust_file2",
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
    4. Explore customer insights!
    """)
else:
    df1 = st.session_state.df1
    df2 = st.session_state.df2

    # Column mapping
    st.sidebar.markdown("---")
    st.sidebar.subheader("Column Mapping")

    # Smart defaults
    default_id_col = "Booking number" if "Booking number" in df1.columns else df1.columns[0]
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
        key="cust_id_col_1"
    )

    # Location column (optional)
    location_options = ["None"] + df1.columns.tolist()
    location_default_index = location_options.index(default_location_col) if default_location_col in location_options else 0

    location_col = st.sidebar.selectbox(
        "Location (optional)",
        options=location_options,
        index=location_default_index,
        key="cust_location_col"
    )

    # Email column (required for this page)
    default_email_col = "Email address" if "Email address" in df1.columns else None
    email_options = ["None"] + df1.columns.tolist()
    email_default_index = email_options.index(default_email_col) if default_email_col in email_options else 0

    email_col = st.sidebar.selectbox(
        "Email Address",
        options=email_options,
        index=email_default_index,
        help="Required for customer analysis",
        key="cust_email_col"
    )

    # Get visit date column mapping
    default_start_col = "Start" if "Start" in df2.columns else df2.columns[0]
    id_col_2 = st.sidebar.selectbox(
        "Booking ID (File 2)",
        options=df2.columns.tolist(),
        index=df2.columns.tolist().index("Booking number") if "Booking number" in df2.columns else 0,
        key="cust_id_col_2"
    )
    visit_col_2 = st.sidebar.selectbox(
        "Visit Date (File 2)",
        options=df2.columns.tolist(),
        index=df2.columns.tolist().index(default_start_col) if default_start_col in df2.columns else 0,
        key="cust_visit_col_2"
    )

    # Check if email column is configured
    if email_col == "None":
        st.warning("""
        **Email column not configured.**

        To use customer analysis, please select an Email Address column in the sidebar.
        This allows us to identify unique customers and track recurring visits.
        """)
    else:
        # Prepare customer data with visit dates
        customer_data = df1[[id_col_1, email_col]].copy()
        customer_data.columns = ['booking_id', 'email']

        # Add location if available
        if location_col != "None":
            customer_data['location'] = df1[location_col]

        # Merge with visit dates from df2
        df2_prep = df2[[id_col_2, visit_col_2]].copy()
        df2_prep.columns = ['booking_id', 'visit_date']
        df2_prep['visit_date'] = pd.to_datetime(df2_prep['visit_date'], errors='coerce')
        customer_data = customer_data.merge(df2_prep, on='booking_id', how='inner')

        # Remove rows with missing emails
        customer_data = customer_data[customer_data['email'].notna() & (customer_data['email'] != '')]

        if len(customer_data) == 0:
            st.warning("No valid email addresses found in the data.")
        else:
            # Date range selector in reserved container (under header)
            min_date = customer_data['visit_date'].min().date()
            max_date = customer_data['visit_date'].max().date()

            with date_range_container:
                date_col1, date_col2 = st.columns([2, 4])
                with date_col1:
                    date_range = st.date_input(
                        "Date Range (Visit Date)",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                        help="Filter customers by visit date",
                        key="cust_date_range"
                    )

            # Apply date filter
            if len(date_range) == 2:
                start_date, end_date = date_range
                customer_data = customer_data[
                    (customer_data['visit_date'].dt.date >= start_date) &
                    (customer_data['visit_date'].dt.date <= end_date)
                ]

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
            tier_distribution['label'] = tier_distribution['Percentage'].apply(lambda x: f"{x}%")

            # Display key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Total Unique Customers",
                    f"{total_customers:,}",
                    help="Number of unique email addresses in the dataset."
                )
            with col2:
                st.metric(
                    "Recurring Customers",
                    f"{recurring_customers:,}",
                    help="Customers who have made more than one booking."
                )
            with col3:
                st.metric(
                    "Recurring Rate",
                    f"{recurring_pct:.1f}%",
                    help="Percentage of customers who returned for at least one more booking."
                )

            # Customer tier chart
            st.markdown("### Customer Tier Distribution")

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
                st.markdown("---")
                st.markdown("### Location Loyalty Among Recurring Customers")

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

                # Side-by-side layout
                col1, col2 = st.columns([3, 2])

                with col1:
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
                    loyalty_config = {
                        'Loyalty Type': st.column_config.TextColumn('Loyalty Type', help='How many locations the customer visits'),
                        'Customers': st.column_config.NumberColumn('Customers', help='Number of customers in this category'),
                    }
                    st.dataframe(loyalty_distribution, use_container_width=True, hide_index=True, column_config=loyalty_config)

                # Recurring customers per location
                st.markdown("### Recurring Customers by Location")

                location_total = customer_data.groupby('location')['email'].nunique().reset_index()
                location_total.columns = ['Location', 'Total Customers']

                location_recurring = recurring_customer_data.groupby('location')['email'].nunique().reset_index()
                location_recurring.columns = ['Location', 'Recurring Customers']

                location_stats = location_total.merge(location_recurring, on='Location')
                location_stats['Recurring Rate (%)'] = (location_stats['Recurring Customers'] / location_stats['Total Customers'] * 100).round(1)

                location_display = location_stats[['Location', 'Total Customers', 'Recurring Customers', 'Recurring Rate (%)']].copy()
                location_display = location_display.sort_values('Recurring Customers', ascending=False)

                # Add total row
                total_cust = location_display['Total Customers'].sum()
                total_rec = location_display['Recurring Customers'].sum()
                avg_rate = location_stats['Recurring Rate (%)'].mean().round(1)

                total_row = pd.DataFrame({
                    'Location': ['Total'],
                    'Total Customers': [total_cust],
                    'Recurring Customers': [total_rec],
                    'Recurring Rate (%)': [avg_rate]
                })
                location_display = pd.concat([location_display, total_row], ignore_index=True)

                location_cust_config = {
                    'Location': st.column_config.TextColumn('Location'),
                    'Total Customers': st.column_config.NumberColumn('Total Customers', help='Unique customers who booked at this location'),
                    'Recurring Customers': st.column_config.NumberColumn('Recurring Customers', help='Customers with more than one booking'),
                    'Recurring Rate (%)': st.column_config.NumberColumn('Recurring Rate (%)', help='Percentage of customers who returned'),
                }
                st.dataframe(location_display, use_container_width=True, hide_index=True, column_config=location_cust_config)

                # Explanation
                global_total = len(customer_frequency)
                global_recurring = len(customer_frequency[customer_frequency['bookings'] > 1])
                global_rate = (global_recurring / global_total * 100) if global_total > 0 else 0

                with st.expander("Why is the recurring rate different from the top metrics?"):
                    st.markdown(f"""
**Top Metrics (Global View):** {global_rate:.1f}% recurring rate
- Counts each customer once across all locations
- Example: If Sarah visits Matsu and Noord, she counts as 1 total customer

**Location Table (Location View):** {avg_rate:.1f}% average recurring rate
- Counts each customer once per location they visit
- Example: If Sarah visits Matsu and Noord, she counts as 1 customer at Matsu + 1 customer at Noord

**Why the difference?**
- Total Customers in table: {total_cust:,} (sum per location)
- Total Unique Customers (top): {global_total:,} (unique people)
- Difference: {total_cust - global_total:,} extra counts from customers visiting multiple locations

This is actually good news - it means your recurring customers are loyal across multiple locations!
                    """)

            # Insights
            st.markdown("---")
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

    # Reset button
    st.sidebar.markdown("---")
    if st.sidebar.button("Clear All & Start Over", key="cust_reset"):
        st.session_state.df1 = None
        st.session_state.df2 = None
        st.rerun()
