"""
Kuuma Booking Analyzer - Revenue & Value Page
Revenue analysis, promotion effectiveness, and customer value
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Kuuma - Revenue & Value",
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

st.markdown("## Revenue & Value Analysis")
st.markdown("Revenue trends, promotion effectiveness, and customer lifetime value")

# Initialize session state
if 'df1' not in st.session_state:
    st.session_state.df1 = None

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

# Reserve container for navigation at top of sidebar
nav_container = st.sidebar.container()

# Sidebar - Upload section
st.sidebar.header("Upload & Configure")

# File uploader - only need Booking Creation Dates for this page
uploaded_file1 = st.sidebar.file_uploader(
    "Booking Creation Dates (.xls/.xlsx)",
    type=["xls", "xlsx"],
    help="Upload the file containing booking data with revenue information",
    key="rev_file1"
)

# Load File
if uploaded_file1 is not None:
    df1, error1 = load_excel_file(uploaded_file1)
    if error1:
        st.sidebar.error(f"Error reading file: {error1}")
    else:
        st.session_state.df1 = df1
        st.sidebar.success(f"File loaded: {len(df1):,} rows")

# Fill navigation container (now that file is loaded)
if st.session_state.df1 is not None:
    with nav_container:
        st.markdown("### Navigation")
        st.page_link("app.py", label="Booking Patterns", icon=":material/bar_chart:")
        st.page_link("pages/3_Customers.py", label="Recurring Customers", icon=":material/group:")
        st.page_link("pages/4_Revenue.py", label="Revenue & Value", icon=":material/payments:")
        st.page_link("pages/5_Promotions.py", label="Promotions", icon=":material/sell:")
        st.markdown("---")

# Main content
if st.session_state.df1 is None:
    st.info("**No data loaded.** Please upload your Booking Creation Dates file using the sidebar to begin analysis.")
    st.markdown("""
    ### Getting Started
    1. Upload your booking creation dates file
    2. Configure column mappings
    3. Explore revenue insights!
    """)
else:
    df1 = st.session_state.df1

    # Column mapping
    st.sidebar.markdown("---")
    st.sidebar.subheader("Column Mapping")

    # Smart defaults
    default_id_col = "Booking number" if "Booking number" in df1.columns else df1.columns[0]
    default_created_col = "Created" if "Created" in df1.columns else df1.columns[0]

    if "Tour" in df1.columns:
        default_location_col = "Tour"
    elif "Activity" in df1.columns:
        default_location_col = "Activity"
    else:
        default_location_col = None

    id_col = st.sidebar.selectbox(
        "Booking ID",
        options=df1.columns.tolist(),
        index=df1.columns.tolist().index(default_id_col) if default_id_col in df1.columns else 0,
        key="rev_id_col"
    )

    date_col = st.sidebar.selectbox(
        "Booking Date",
        options=df1.columns.tolist(),
        index=df1.columns.tolist().index(default_created_col) if default_created_col in df1.columns else 0,
        key="rev_date_col"
    )

    # Location column (optional)
    location_options = ["None"] + df1.columns.tolist()
    location_default_index = location_options.index(default_location_col) if default_location_col in location_options else 0

    location_col = st.sidebar.selectbox(
        "Location (optional)",
        options=location_options,
        index=location_default_index,
        key="rev_location_col"
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
        key="rev_revenue_col"
    )

    # Email column (for customer value)
    default_email_col = "Email address" if "Email address" in df1.columns else None
    email_options = ["None"] + df1.columns.tolist()
    email_default_index = email_options.index(default_email_col) if default_email_col in email_options else 0

    email_col = st.sidebar.selectbox(
        "Email Address",
        options=email_options,
        index=email_default_index,
        help="Required for customer value analysis",
        key="rev_email_col"
    )

    # Check required columns
    if revenue_col == "None":
        st.warning("""
        **Revenue column not configured.**

        To use revenue analysis, please select a Revenue Column in the sidebar (e.g., "Total gross").
        """)
    else:
        # Process data directly from Booking Creation Dates file
        @st.cache_data
        def process_revenue_data(df1, id_col, date_col, location_col, revenue_col, email_col):
            # Prepare dataframe
            data = pd.DataFrame()
            data['booking_id'] = df1[id_col]
            data['booking_date'] = pd.to_datetime(df1[date_col], errors='coerce')

            # Add optional columns
            if location_col != "None":
                data['location'] = df1[location_col].values

            if revenue_col != "None":
                data['revenue'] = pd.to_numeric(df1[revenue_col], errors='coerce').fillna(0)

            if email_col != "None":
                data['email'] = df1[email_col].values

            # Filter invalid records
            data_clean = data[data['booking_date'].notna()].copy()

            return data_clean

        with st.spinner("Processing data..."):
            processed_data = process_revenue_data(
                df1, id_col, date_col, location_col, revenue_col, email_col
            )

        if len(processed_data) == 0:
            st.error("No valid booking data found.")
        else:
            # ==================== REVENUE ANALYSIS ====================
            st.markdown("### Revenue Analysis")

            # Key revenue metrics
            total_revenue = processed_data['revenue'].sum()
            avg_booking_value = processed_data['revenue'].mean()
            median_booking_value = processed_data['revenue'].median()
            total_bookings = len(processed_data)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Total Revenue",
                    f"€{total_revenue:,.0f}",
                    help="Sum of all booking revenue in the dataset."
                )
            with col2:
                st.metric(
                    "Total Bookings",
                    f"{total_bookings:,}",
                    help="Total number of bookings with valid revenue data."
                )
            with col3:
                st.metric(
                    "Avg Booking Value",
                    f"€{avg_booking_value:.2f}",
                    help="Mean revenue per booking (total revenue ÷ bookings)."
                )
            with col4:
                st.metric(
                    "Median Booking Value",
                    f"€{median_booking_value:.2f}",
                    help="Middle booking value. Less affected by high-value outliers."
                )

            # Revenue by location
            if location_col != "None" and 'location' in processed_data.columns:
                st.markdown("#### Revenue by Location")

                location_revenue = processed_data.groupby('location').agg({
                    'revenue': ['sum', 'mean', 'median', 'count']
                }).round(2)
                location_revenue.columns = ['Total Revenue', 'Avg Booking', 'Median Booking', 'Bookings']
                location_revenue = location_revenue.sort_values('Total Revenue', ascending=False)
                location_revenue['% of Revenue'] = (location_revenue['Total Revenue'] / total_revenue * 100).round(1)
                location_revenue = location_revenue[['Total Revenue', '% of Revenue', 'Bookings', 'Avg Booking', 'Median Booking']]

                # Format currency
                location_display = location_revenue.copy()
                location_display['Total Revenue'] = location_display['Total Revenue'].apply(lambda x: f"€{x:,.0f}")
                location_display['Avg Booking'] = location_display['Avg Booking'].apply(lambda x: f"€{x:.2f}")
                location_display['Median Booking'] = location_display['Median Booking'].apply(lambda x: f"€{x:.2f}")

                st.dataframe(location_display, use_container_width=True)

                # Revenue chart
                fig_rev_loc = px.bar(
                    location_revenue.reset_index(),
                    x='location',
                    y='Total Revenue',
                    title="Revenue by Location",
                    labels={'location': 'Location', 'Total Revenue': 'Revenue (€)'}
                )
                fig_rev_loc.update_traces(marker_color='#2ecc71')
                fig_rev_loc.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig_rev_loc, use_container_width=True)

            # ==================== CUSTOMER VALUE ====================
            st.markdown("---")
            st.markdown("### Customer Value Analysis")

            if email_col == "None" or 'email' not in processed_data.columns:
                st.info("Select an Email Address column in the sidebar to analyze customer value.")
            else:
                # Calculate customer metrics
                customer_data = processed_data[processed_data['email'].notna() & (processed_data['email'] != '')].copy()

                if len(customer_data) == 0:
                    st.warning("No valid email addresses found in the data.")
                else:
                    customer_value = customer_data.groupby('email').agg({
                        'booking_id': 'count',
                        'revenue': 'sum'
                    }).reset_index()
                    customer_value.columns = ['email', 'bookings', 'lifetime_value']

                    # Customer tiers (same as Recurring Customers page)
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

                    customer_value['tier'] = customer_value['bookings'].apply(categorize_customer)
                    tier_order = ['One-time', 'Light (2-3)', 'Regular (4-6)', 'Frequent (7-10)', 'VIP (11+)']
                    customer_value['tier'] = pd.Categorical(customer_value['tier'], categories=tier_order, ordered=True)

                    # Key metrics
                    total_customers = len(customer_value)
                    avg_ltv = customer_value['lifetime_value'].mean()
                    median_ltv = customer_value['lifetime_value'].median()
                    top_10_pct_revenue = customer_value.nlargest(int(total_customers * 0.1), 'lifetime_value')['lifetime_value'].sum()
                    top_10_share = top_10_pct_revenue / customer_value['lifetime_value'].sum() * 100

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(
                            "Unique Customers",
                            f"{total_customers:,}",
                            help="Number of unique customers identified by email address."
                        )
                    with col2:
                        st.metric(
                            "Avg Current Value",
                            f"€{avg_ltv:.2f}",
                            help="Average total spend per customer to date."
                        )
                    with col3:
                        st.metric(
                            "Median Current Value",
                            f"€{median_ltv:.2f}",
                            help="Middle value of customer spending. Shows what a typical customer spends."
                        )
                    with col4:
                        st.metric(
                            "Top 10% Revenue Share",
                            f"{top_10_share:.1f}%",
                            help="Percentage of total revenue from your top 10% highest-spending customers."
                        )

                    # Explanation of Current Value calculations
                    with st.expander("How Current Value is Calculated"):
                        st.markdown("""
                        **Current Value** shows what each customer has actually spent to date.

                        **Calculation Steps:**
                        1. Group all bookings by customer (email address)
                        2. Sum the total revenue for each customer
                        3. This sum = that customer's **Current Value**

                        **Avg Current Value:**
                        ```
                        Sum of all customers' current values ÷ Number of customers
                        ```

                        **Median Current Value:**
                        ```
                        The middle value when all customers are sorted by their current value
                        ```
                        """)

                        # Show actual calculation
                        total_customer_revenue = customer_value['lifetime_value'].sum()
                        st.markdown(f"""
                        **Your Data:**
                        | Metric | Calculation | Result |
                        |--------|-------------|--------|
                        | Avg Current Value | €{total_customer_revenue:,.0f} ÷ {total_customers:,} customers | **€{avg_ltv:.2f}** |
                        | Median Current Value | Middle value of {total_customers:,} customers | **€{median_ltv:.2f}** |

                        **Why both metrics?**
                        - **Average** can be skewed by a few high-spending customers
                        - **Median** shows what a "typical" customer spends
                        - If Avg >> Median: You have some very high-value customers pulling up the average
                        - If Avg ≈ Median: Customer spending is relatively evenly distributed
                        """)

                    # ==================== CUSTOMER LIFETIME VALUE (CLV) ====================
                    st.markdown("---")
                    st.markdown("#### Customer Lifetime Value (CLV)")

                    st.markdown("""
                    CLV predicts how much revenue a customer will generate over their entire relationship with Kuuma.
                    """)

                    # ===== SEGMENTATION =====
                    # VIP: 5+ bookings, Regular: 2-4 bookings, New: 1 booking
                    def categorize_segment(bookings):
                        if bookings >= 5:
                            return "VIP"
                        elif bookings >= 2:
                            return "Regular"
                        else:
                            return "New"

                    customer_value['segment'] = customer_value['bookings'].apply(categorize_segment)
                    segment_order = ['New', 'Regular', 'VIP']
                    customer_value['segment'] = pd.Categorical(customer_value['segment'], categories=segment_order, ordered=True)

                    # ===== STEP 1: AVERAGE ORDER VALUE =====
                    total_revenue_for_clv = customer_data['revenue'].sum()
                    total_bookings_for_clv = len(customer_data)
                    aov = total_revenue_for_clv / total_bookings_for_clv if total_bookings_for_clv > 0 else 0

                    # ===== STEP 2: RETENTION RATE (Cohort-based) =====
                    # Find earliest complete month
                    min_date = customer_data['booking_date'].min()
                    max_date = customer_data['booking_date'].max()

                    # Get the first complete month
                    first_month_start = min_date.replace(day=1)
                    if min_date.day > 1:
                        # Move to next month if we don't have the full first month
                        first_month_start = (first_month_start + pd.DateOffset(months=1))

                    first_month_end = first_month_start + pd.DateOffset(months=1) - pd.Timedelta(days=1)
                    two_months_later = first_month_start + pd.DateOffset(months=3) - pd.Timedelta(days=1)

                    # Find customers whose FIRST booking was in the cohort month
                    customer_first_booking = customer_data.groupby('email')['booking_date'].min().reset_index()
                    customer_first_booking.columns = ['email', 'first_booking_date']

                    cohort_customers = customer_first_booking[
                        (customer_first_booking['first_booking_date'] >= first_month_start) &
                        (customer_first_booking['first_booking_date'] <= first_month_end)
                    ]['email'].tolist()

                    cohort_size = len(cohort_customers)

                    # Count how many made at least one more booking in the following 2 months
                    if cohort_size > 0:
                        # Get all bookings for cohort customers
                        cohort_bookings = customer_data[customer_data['email'].isin(cohort_customers)]

                        # Find customers who booked again after their first booking within 2 months
                        returning_customers = 0
                        for email in cohort_customers:
                            cust_bookings = cohort_bookings[cohort_bookings['email'] == email]['booking_date'].sort_values()
                            if len(cust_bookings) > 1:
                                # Check if second booking was within 2 months of cohort period end
                                second_booking = cust_bookings.iloc[1]
                                if second_booking <= two_months_later:
                                    returning_customers += 1

                        retention_rate = returning_customers / cohort_size
                    else:
                        retention_rate = 0.3  # Default fallback

                    # ===== STEP 3: CHURN RATE =====
                    churn_rate = 1 - retention_rate

                    # ===== STEP 4: CALCULATE CLV =====
                    # CLV = AOV × (1 + retention_rate / churn_rate)
                    if churn_rate > 0:
                        clv = aov * (1 + retention_rate / churn_rate)
                    else:
                        clv = aov * 10  # Cap at 10x AOV if no churn

                    # ===== DETECT LIKELY CHURNED CUSTOMERS =====
                    data_end_date = max_date

                    # Calculate per-customer metrics
                    customer_metrics = customer_data.groupby('email').agg({
                        'booking_date': ['min', 'max', 'count', lambda x: x.sort_values().diff().mean().days if len(x) > 1 else None],
                        'revenue': 'sum'
                    })
                    customer_metrics.columns = ['first_booking', 'last_booking', 'num_bookings', 'avg_interval_days', 'total_revenue']
                    customer_metrics = customer_metrics.reset_index()

                    # Calculate days since last booking
                    customer_metrics['days_since_last'] = (data_end_date - customer_metrics['last_booking']).dt.days

                    # Flag as "likely churned":
                    # - Repeat customers: days since last > 2.5× their average interval
                    # - Single-booking customers: 60 days threshold
                    def is_likely_churned(row):
                        if row['num_bookings'] == 1:
                            return row['days_since_last'] > 60
                        else:
                            return row['days_since_last'] > (row['avg_interval_days'] * 2.5)

                    customer_metrics['is_likely_churned'] = customer_metrics.apply(is_likely_churned, axis=1)

                    churned_count = customer_metrics['is_likely_churned'].sum()
                    active_count = len(customer_metrics) - churned_count

                    # ===== KEY METRICS =====
                    st.markdown("##### Key CLV Metrics")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(
                            "Customer Lifetime Value",
                            f"€{clv:.2f}",
                            help="Predicted total revenue from a customer over their entire relationship. Formula: AOV × (1 + Retention ÷ Churn)"
                        )
                    with col2:
                        st.metric(
                            "Avg Order Value",
                            f"€{aov:.2f}",
                            help="Average revenue per booking. Calculated as total revenue divided by number of bookings."
                        )
                    with col3:
                        st.metric(
                            "Retention Rate",
                            f"{retention_rate:.1%}",
                            help="Percentage of new customers who return within 2 months. Based on cohort analysis of first-time customers."
                        )
                    with col4:
                        st.metric(
                            "Churn Rate",
                            f"{churn_rate:.1%}",
                            help="Percentage of customers who don't return (1 − Retention Rate). Lower is better."
                        )

                    with st.expander("View CLV Formula & Calculation Details", expanded=False):
                        st.markdown("""
                        **Formula:**
                        ```
                        CLV = Average Order Value × (1 + Retention Rate ÷ Churn Rate)
                        ```
                        """)

                        st.markdown("---")
                        st.markdown("**Step 1: Average Order Value (AOV)**")
                        st.markdown(f"""
                        | Calculation | Result |
                        |-------------|--------|
                        | Sum of Total Gross ÷ Count of Bookings | €{total_revenue_for_clv:,.0f} ÷ {total_bookings_for_clv:,} = **€{aov:.2f}** |
                        """)

                        st.markdown("---")
                        st.markdown("**Step 2: Retention Rate (Cohort-based)**")
                        st.markdown(f"""
                        | Step | Description | Value |
                        |------|-------------|-------|
                        | Cohort Month | First complete month in data | {first_month_start.strftime('%B %Y')} |
                        | Cohort Size | Customers whose first booking was in cohort month | {cohort_size:,} |
                        | Returning Customers | Made at least 1 more booking within 2 months | {returning_customers if cohort_size > 0 else 'N/A':,} |
                        | **Retention Rate** | Returning ÷ Cohort Size | **{retention_rate:.1%}** |
                        """)

                        st.markdown("---")
                        st.markdown("**Step 3: Churn Rate**")
                        st.markdown(f"""
                        | Calculation | Result |
                        |-------------|--------|
                        | 1 − Retention Rate | 1 − {retention_rate:.2f} = **{churn_rate:.1%}** |
                        """)

                        st.markdown("---")
                        st.markdown("**Step 4: Customer Lifetime Value**")
                        st.markdown(f"""
                        | Calculation | Result |
                        |-------------|--------|
                        | AOV × (1 + Retention ÷ Churn) | €{aov:.2f} × (1 + {retention_rate:.2f} ÷ {churn_rate:.2f}) = **€{clv:.2f}** |
                        """)

                    # ===== 3-MONTH CLV PROJECTION =====
                    st.markdown("---")
                    st.markdown("##### 3-Month CLV Projection")

                    # Calculate average monthly bookings per active customer
                    # Use data span to determine average frequency
                    data_span_days = (max_date - min_date).days
                    data_span_months = max(data_span_days / 30.44, 1)  # Average days per month

                    # Monthly bookings per customer
                    avg_monthly_bookings_per_customer = total_bookings_for_clv / total_customers / data_span_months

                    # Convert 2-month retention rate to monthly retention rate
                    # If R₂ = 2-month retention, then monthly retention r = √R₂
                    monthly_retention_rate = retention_rate ** 0.5

                    # Account for churn: apply monthly retention rate over 3 months
                    # Month 1: 100% of customers, Month 2: r, Month 3: r²
                    retention_multiplier = 1 + monthly_retention_rate + (monthly_retention_rate ** 2)
                    adjusted_bookings_3m = avg_monthly_bookings_per_customer * retention_multiplier

                    # 3-month projected revenue per customer
                    projected_revenue_per_customer = adjusted_bookings_3m * aov

                    # Total 3-month projection for all active customers
                    total_projected_revenue_3m = projected_revenue_per_customer * active_count

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Active Customers",
                            f"{active_count:,}",
                            help="Customers not flagged as churned (still booking within expected intervals)."
                        )
                    with col2:
                        st.metric(
                            "Projected Revenue (3 mo)",
                            f"€{total_projected_revenue_3m:,.0f}",
                            help="Expected total revenue from active customers over the next 3 months."
                        )
                    with col3:
                        st.metric(
                            "Per Customer (3 mo)",
                            f"€{projected_revenue_per_customer:.2f}",
                            help="Expected revenue per active customer over the next 3 months."
                        )

                    with st.expander("How the 3-Month Projection is Calculated"):
                        st.markdown(f"""
**Formula:**
```
3-Month Projection = Active Customers × AOV × Adjusted Bookings
```

**Step-by-Step:**

| Step | Calculation | Result |
|------|-------------|--------|
| Data span | {data_span_days:,} days = {data_span_months:.1f} months | |
| Avg monthly bookings/customer | {total_bookings_for_clv:,} bookings ÷ {total_customers:,} customers ÷ {data_span_months:.1f} months | {avg_monthly_bookings_per_customer:.3f} |
| 2-month retention rate | (from cohort analysis) | {retention_rate:.1%} |
| Monthly retention rate | √{retention_rate:.2f} | {monthly_retention_rate:.1%} |
| Retention multiplier | 1 + {monthly_retention_rate:.2f} + {monthly_retention_rate:.2f}² | {retention_multiplier:.2f} |
| Adjusted bookings (3 mo) | {avg_monthly_bookings_per_customer:.3f} × {retention_multiplier:.2f} | {adjusted_bookings_3m:.3f} |
| Revenue per customer | {adjusted_bookings_3m:.3f} × €{aov:.2f} | **€{projected_revenue_per_customer:.2f}** |
| **Total projection** | €{projected_revenue_per_customer:.2f} × {active_count:,} customers | **€{total_projected_revenue_3m:,.0f}** |

*The 2-month retention rate is converted to monthly (√R₂) for use in the geometric series.*
                        """)

                    # ===== CLV BY SEGMENT =====
                    st.markdown("---")
                    st.markdown("##### CLV by Customer Segment")

                    st.markdown("""
                    **Segmentation:** VIP (5+ bookings) • Regular (2-4 bookings) • New (1 booking)
                    """)

                    # Calculate segment-specific metrics with segment-specific retention rates
                    segment_clv_data = []
                    for segment in segment_order:
                        seg_customers = customer_value[customer_value['segment'] == segment]
                        if len(seg_customers) > 0:
                            seg_emails = seg_customers['email'].tolist()
                            seg_bookings = customer_data[customer_data['email'].isin(seg_emails)]

                            # Segment AOV
                            seg_aov = seg_bookings['revenue'].sum() / len(seg_bookings) if len(seg_bookings) > 0 else 0

                            # Segment retention rate (cohort-based within segment)
                            seg_cohort = [e for e in cohort_customers if e in seg_emails]
                            if len(seg_cohort) > 0:
                                seg_returning = 0
                                for email in seg_cohort:
                                    cust_bookings = seg_bookings[seg_bookings['email'] == email]['booking_date'].sort_values()
                                    if len(cust_bookings) > 1:
                                        second_booking = cust_bookings.iloc[1]
                                        if second_booking <= two_months_later:
                                            seg_returning += 1
                                seg_retention = seg_returning / len(seg_cohort) if len(seg_cohort) > 0 else retention_rate
                            else:
                                # Use segment-based estimation
                                if segment == 'VIP':
                                    seg_retention = min(retention_rate * 1.5, 0.95)  # VIPs are stickier
                                elif segment == 'Regular':
                                    seg_retention = retention_rate * 1.2
                                else:
                                    seg_retention = retention_rate * 0.7  # New customers churn more

                            seg_churn = 1 - seg_retention

                            # Segment CLV
                            if seg_churn > 0:
                                seg_clv = seg_aov * (1 + seg_retention / seg_churn)
                            else:
                                seg_clv = seg_aov * 10

                            # Churned count for segment
                            seg_metrics = customer_metrics[customer_metrics['email'].isin(seg_emails)]
                            seg_churned = seg_metrics['is_likely_churned'].sum()

                            segment_clv_data.append({
                                'Segment': segment,
                                'Customers': len(seg_customers),
                                '% of Total': len(seg_customers) / total_customers * 100,
                                'AOV': seg_aov,
                                'Retention Rate': seg_retention,
                                'Churn Rate': seg_churn,
                                'CLV': seg_clv,
                                'Likely Churned': seg_churned
                            })

                    segment_clv_df = pd.DataFrame(segment_clv_data)

                    # Format for display
                    segment_display = segment_clv_df.copy()
                    segment_display['% of Total'] = segment_display['% of Total'].apply(lambda x: f"{x:.1f}%")
                    segment_display['AOV'] = segment_display['AOV'].apply(lambda x: f"€{x:.2f}")
                    segment_display['Retention Rate'] = segment_display['Retention Rate'].apply(lambda x: f"{x:.1%}")
                    segment_display['Churn Rate'] = segment_display['Churn Rate'].apply(lambda x: f"{x:.1%}")
                    segment_display['CLV'] = segment_display['CLV'].apply(lambda x: f"€{x:.2f}")

                    st.dataframe(segment_display[['Segment', 'Customers', '% of Total', 'AOV', 'Retention Rate', 'Churn Rate', 'CLV', 'Likely Churned']],
                                use_container_width=True, hide_index=True)

                    # Segment CLV Chart
                    col1, col2 = st.columns(2)

                    with col1:
                        fig_seg_clv = px.bar(
                            segment_clv_df,
                            x='Segment',
                            y='CLV',
                            title="CLV by Segment",
                            labels={'CLV': 'CLV (€)'},
                            text=segment_clv_df['CLV'].apply(lambda x: f"€{x:.0f}"),
                            color='Segment',
                            color_discrete_map={'New': '#3498db', 'Regular': '#f39c12', 'VIP': '#9b59b6'}
                        )
                        fig_seg_clv.update_traces(textposition='outside')
                        fig_seg_clv.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig_seg_clv, use_container_width=True)

                    with col2:
                        fig_seg_retention = px.bar(
                            segment_clv_df,
                            x='Segment',
                            y='Retention Rate',
                            title="Retention Rate by Segment",
                            labels={'Retention Rate': 'Retention Rate'},
                            text=segment_clv_df['Retention Rate'].apply(lambda x: f"{x:.1%}"),
                            color='Segment',
                            color_discrete_map={'New': '#3498db', 'Regular': '#f39c12', 'VIP': '#9b59b6'}
                        )
                        fig_seg_retention.update_traces(textposition='outside')
                        fig_seg_retention.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig_seg_retention, use_container_width=True)

                    # ===== CLV BY LOCATION =====
                    if location_col != "None" and 'location' in customer_data.columns:
                        st.markdown("---")
                        st.markdown("##### CLV by Location")

                        st.markdown("*Location-specific analysis shows where to focus marketing efforts.*")

                        # Calculate location-specific CLV
                        location_clv_data = []
                        locations = customer_data['location'].dropna().unique()

                        for loc in locations:
                            loc_bookings = customer_data[customer_data['location'] == loc]
                            loc_emails = loc_bookings['email'].unique()
                            loc_customer_count = len(loc_emails)

                            if loc_customer_count > 0:
                                # Location AOV
                                loc_aov = loc_bookings['revenue'].sum() / len(loc_bookings)

                                # Location retention (use overall retention as proxy, could be refined)
                                loc_cohort = [e for e in cohort_customers if e in loc_emails]
                                if len(loc_cohort) > 3:  # Need enough data
                                    loc_returning = 0
                                    for email in loc_cohort:
                                        cust_bookings = customer_data[customer_data['email'] == email]['booking_date'].sort_values()
                                        if len(cust_bookings) > 1:
                                            second_booking = cust_bookings.iloc[1]
                                            if second_booking <= two_months_later:
                                                loc_returning += 1
                                    loc_retention = loc_returning / len(loc_cohort)
                                else:
                                    loc_retention = retention_rate

                                loc_churn = 1 - loc_retention

                                # Location CLV
                                if loc_churn > 0:
                                    loc_clv = loc_aov * (1 + loc_retention / loc_churn)
                                else:
                                    loc_clv = loc_aov * 10

                                # Location churned
                                loc_metrics = customer_metrics[customer_metrics['email'].isin(loc_emails)]
                                loc_churned = loc_metrics['is_likely_churned'].sum()

                                location_clv_data.append({
                                    'Location': loc,
                                    'Customers': loc_customer_count,
                                    'Bookings': len(loc_bookings),
                                    'AOV': loc_aov,
                                    'Retention Rate': loc_retention,
                                    'CLV': loc_clv,
                                    'Likely Churned': loc_churned
                                })

                        if location_clv_data:
                            location_clv_df = pd.DataFrame(location_clv_data).sort_values('CLV', ascending=False)

                            # Format for display
                            location_display = location_clv_df.copy()
                            location_display['AOV'] = location_display['AOV'].apply(lambda x: f"€{x:.2f}")
                            location_display['Retention Rate'] = location_display['Retention Rate'].apply(lambda x: f"{x:.1%}")
                            location_display['CLV'] = location_display['CLV'].apply(lambda x: f"€{x:.2f}")

                            st.dataframe(location_display, use_container_width=True, hide_index=True)

                            # Location CLV Chart
                            fig_loc_clv = px.bar(
                                location_clv_df,
                                x='Location',
                                y='CLV',
                                title="CLV by Location",
                                labels={'CLV': 'CLV (€)'},
                                text=location_clv_df['CLV'].apply(lambda x: f"€{x:.0f}")
                            )
                            fig_loc_clv.update_traces(marker_color='#1abc9c', textposition='outside')
                            fig_loc_clv.update_layout(height=400, xaxis_tickangle=-45)
                            st.plotly_chart(fig_loc_clv, use_container_width=True)

                    # ===== INSIGHTS =====
                    st.markdown("---")

                    # Find highest CLV segment
                    highest_seg = segment_clv_df.loc[segment_clv_df['CLV'].idxmax()]
                    lowest_churn_seg = segment_clv_df.loc[segment_clv_df['Churn Rate'].idxmin()]

                    st.info(f"""
**CLV Insights:**

- **Overall CLV:** €{clv:.2f} per customer
- **Highest Value Segment:** {highest_seg['Segment']} with €{highest_seg['CLV']:.2f} CLV
- **Most Loyal Segment:** {lowest_churn_seg['Segment']} with {lowest_churn_seg['Retention Rate']:.1%} retention rate
- **Retention Impact:** Improving retention by 10% (from {retention_rate:.1%} to {retention_rate + 0.1:.1%}) would increase CLV to €{aov * (1 + (retention_rate + 0.1) / (churn_rate - 0.1)) if churn_rate > 0.1 else clv * 1.5:.2f}

**Strategic Actions:**
- **VIPs ({segment_clv_df[segment_clv_df['Segment']=='VIP']['Customers'].values[0] if 'VIP' in segment_clv_df['Segment'].values else 0} customers):** Loyalty rewards, exclusive access, personal outreach
- **Regulars ({segment_clv_df[segment_clv_df['Segment']=='Regular']['Customers'].values[0] if 'Regular' in segment_clv_df['Segment'].values else 0} customers):** Upgrade campaigns to VIP status
- **New ({segment_clv_df[segment_clv_df['Segment']=='New']['Customers'].values[0] if 'New' in segment_clv_df['Segment'].values else 0} customers):** Second-visit incentives (critical for retention)
- **Likely Churned ({churned_count:,} customers):** Win-back campaigns, special offers
                    """)

    # Reset button
    st.sidebar.markdown("---")
    if st.sidebar.button("Clear All & Start Over", key="rev_reset"):
        st.session_state.df1 = None
        st.rerun()
