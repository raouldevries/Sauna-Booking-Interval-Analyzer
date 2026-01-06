"""
Kuuma Booking Analyzer - Marketing Campaign Analysis Page
Analyze Google Ads and Meta Ads campaign performance with SEE-THINK-DO-CARE framework
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import sys
sys.path.insert(0, '..')
from data_loader import init_session_state

# Page configuration
st.set_page_config(
    page_title="Kuuma - Marketing Analysis",
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

# Location keyword mapping for filtering campaigns
# Maps campaign keywords to display names
# Note: Campaigns often target multiple locations like "Amsterdam M, IJ, N & Sloterplas"
# Use patterns that match these abbreviated formats
LOCATION_KEYWORDS = {
    'Marineterrein Matsu': ['matsu', ' m,', ' m ', '& m ', ', m&', 'marine', 'marineterrein'],  # Matches "M," in campaign names
    'Marineterrein Bjørk': ['bjørk', 'bjork', 'marine', 'marineterrein'],  # Also matches generic "marine" campaigns
    'Amsterdam Sloterplas': ['sloterplas', 'sloterpas'],
    'Amsterdam Noord': [' n ', ' n,', ' n&', '& n ', ', n,', 'noord'],  # Matches "N" in "M, IJ, N &"
    'Amsterdam IJ': ['ij ', ', ij', '& ij', 'ij,', 'ij&', 'amsterdam ij', 'aan t ij', 'centrum'],  # Matches "IJ" patterns
    'Nijmegen Lent': ['nijmegen lent', 'lent', 'nijmegen'],  # Generic 'nijmegen' maps here
    'Nijmegen NYMA': ['nijmegen nyma', 'nyma'],
    'Rotterdam Rijnhaven': ['rijnhaven', 'rotterdam'],  # Generic 'rotterdam' maps here
    'Rotterdam Delfshaven': ['delfshaven'],
    'Scheveningen': ['scheveningen'],
    'Den Bosch': ['den bosch', 'denbosch'],
    'Katwijk': ['katwijk'],
    'Wijk aan Zee': ['wijk aan zee', 'wijk'],
    'Bloemendaal': ['bloemendaal'],
}

# Capacity data from Bezettings analyse (weekly capacity per location)
# dal = off-peak (Mon-Thu 10:00-16:00), piek = peak (Mon-Thu other hours)
# weekday = dal + piek, weekend = Fri-Sun
LOCATION_CAPACITY = {
    'Kuuma Marineterrein Matsu': {'dal': 96, 'piek': 366, 'weekday': 462, 'weekend': 198, 'cluster': 'Flagship'},
    'Kuuma Marineterrein Bjørk': {'dal': 128, 'piek': 488, 'weekday': 616, 'weekend': 264, 'cluster': 'Flagship'},
    'Kuuma Noord': {'dal': 96, 'piek': 336, 'weekday': 432, 'weekend': 192, 'cluster': 'Groeier'},
    'Kuuma Sloterplas': {'dal': 112, 'piek': 392, 'weekday': 504, 'weekend': 224, 'cluster': 'Groeier'},
    'Kuuma Aan ´t IJ (Centrum)': {'dal': 96, 'piek': 324, 'weekday': 420, 'weekend': 180, 'cluster': 'Groeier'},
    'Kuuma Nijmegen Lent': {'dal': 80, 'piek': 300, 'weekday': 380, 'weekend': 160, 'cluster': 'Groeier'},
    'Kuuma Nijmegen NYMA': {'dal': 80, 'piek': 300, 'weekday': 380, 'weekend': 160, 'cluster': 'Groeier'},
    'Kuuma Rotterdam Rijnhaven': {'dal': 80, 'piek': 300, 'weekday': 380, 'weekend': 160, 'cluster': 'Groeier'},
    'Kuuma Rotterdam Delfshaven': {'dal': 80, 'piek': 300, 'weekday': 380, 'weekend': 160, 'cluster': 'Groeier'},
    'Kuuma Scheveningen': {'dal': 120, 'piek': 450, 'weekday': 570, 'weekend': 240, 'cluster': 'Flagship'},
    'Kuuma Den Bosch': {'dal': 80, 'piek': 300, 'weekday': 380, 'weekend': 160, 'cluster': 'Groeier'},
    'Kuuma Katwijk': {'dal': 80, 'piek': 300, 'weekday': 380, 'weekend': 160, 'cluster': 'Groeier'},
    'Kuuma Wijk aan Zee': {'dal': 80, 'piek': 300, 'weekday': 380, 'weekend': 160, 'cluster': 'Groeier'},
    'Kuuma Bloemendaal': {'dal': 80, 'piek': 300, 'weekday': 380, 'weekend': 160, 'cluster': 'Groeier'},
}

# Map marketing location names to booking data location names
LOCATION_NAME_MAP = {
    # Marketing name -> Booking data location names (list for multiple matches)
    'Marineterrein Matsu': ['Kuuma Marineterrein Matsu'],
    'Marineterrein Bjørk': ['Kuuma Marineterrein Bjørk'],
    'Amsterdam Sloterplas': ['Kuuma Sloterplas'],
    'Amsterdam Noord': ['Kuuma Noord'],
    'Amsterdam IJ': ['Kuuma Aan ´t IJ (Centrum)'],
    'Nijmegen Lent': ['Kuuma Nijmegen Lent'],
    'Nijmegen NYMA': ['Kuuma Nijmegen NYMA'],
    'Rotterdam Rijnhaven': ['Kuuma Rotterdam Rijnhaven'],
    'Rotterdam Delfshaven': ['Kuuma Rotterdam Delfshaven'],
    'Scheveningen': ['Kuuma Scheveningen'],
    'Den Bosch': ['Kuuma Den Bosch'],
    'Katwijk': ['Kuuma Katwijk'],
    'Wijk aan Zee': ['Kuuma Wijk aan Zee'],
    'Bloemendaal': ['Kuuma Bloemendaal'],
}

# STDC phase colors (matching the plan)
STDC_COLORS = {
    'SEE': '#3498db',      # Blue
    'THINK': '#f39c12',    # Orange
    'DO': '#27ae60',       # Green
    'CARE': '#9b59b6',     # Purple
    'Untagged': '#9ca3af'  # Gray
}

# Default STDC suggestions based on campaign keywords
# Google Ads naming: NL | S | ... (Search), NL | PM | ... (Performance Max)
# Meta naming: Clicks | ..., Reach - ..., Conversions | ...
STDC_SUGGESTIONS = {
    'SEE': ['display', 'demand gen', 'reach', 'awareness', 'see'],
    'THINK': ['non-branded', 'non branded', 'think', 'consideration', 'clicks |'],
    'DO': ['| s |', '| pm |', 'branded', 'brand', 'conversions', 'conversion', 'do', 'purchase'],
    'CARE': ['retargeting', 'rm', 'remarketing', 'care', 'loyalty']
}


def suggest_stdc_phase(campaign_name):
    """Suggest STDC phase based on campaign name keywords."""
    if pd.isna(campaign_name) or not isinstance(campaign_name, str):
        return 'Untagged'
    name_lower = campaign_name.lower()

    # Priority 1: Check for Google Ads campaign types (S=Search, PM=Performance Max) → DO
    if '| s |' in name_lower or '| pm |' in name_lower:
        return 'DO'

    # Priority 2: Check for remarketing/retargeting → CARE
    for keyword in STDC_SUGGESTIONS['CARE']:
        if keyword in name_lower:
            return 'CARE'

    # Priority 3: Check for conversion campaigns → DO
    for keyword in ['conversions', 'conversion', 'purchase']:
        if keyword in name_lower:
            return 'DO'

    # Priority 4: Check for awareness campaigns → SEE
    for keyword in STDC_SUGGESTIONS['SEE']:
        if keyword in name_lower:
            return 'SEE'

    # Priority 5: Check for consideration campaigns → THINK
    for keyword in STDC_SUGGESTIONS['THINK']:
        if keyword in name_lower:
            return 'THINK'

    return 'Untagged'


def campaign_matches_location(campaign_name, locations=None):
    """Check if campaign name contains any location keywords. Returns first match."""
    if pd.isna(campaign_name) or not isinstance(campaign_name, str):
        return None

    name_lower = campaign_name.lower()

    # Check for "alle locaties" (all locations)
    if 'alle locaties' in name_lower or 'all locations' in name_lower:
        return 'all'

    # Search all location keywords
    for location, keywords in LOCATION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in name_lower:
                return location
    return None


def campaign_matches_all_locations(campaign_name):
    """Return ALL locations that match a campaign name (for multi-location campaigns).
    Note: Returns empty list for 'alle locaties' campaigns - these are handled separately."""
    if pd.isna(campaign_name) or not isinstance(campaign_name, str):
        return []

    name_lower = campaign_name.lower()

    # "Alle locaties" campaigns are handled separately in the "All locations" row
    if 'alle locaties' in name_lower or 'all locations' in name_lower:
        return []

    # Find all matching locations
    matched_locations = []
    for location, keywords in LOCATION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in name_lower:
                matched_locations.append(location)
                break  # Only add each location once

    return matched_locations


def get_revenue_per_location(df1, location_col='Tour', revenue_col='Total gross', split_marine=False):
    """Calculate revenue and bookings per location from booking data."""
    if df1 is None or location_col not in df1.columns:
        return {}

    # Create lowercase version of location column for case-insensitive matching
    df1_lower = df1.copy()
    df1_lower['_location_lower'] = df1_lower[location_col].astype(str).str.lower()

    result = {}
    for marketing_loc, booking_locs in LOCATION_NAME_MAP.items():
        # Create lowercase versions of booking location names
        booking_locs_lower = [loc.lower() for loc in booking_locs]

        # Special handling for Marineterrein - split into Matsu and Bjørk
        if split_marine and marketing_loc in ['Marineterrein Matsu', 'Marineterrein Bjørk']:
            for booking_loc in booking_locs:
                mask = df1_lower['_location_lower'] == booking_loc.lower()
                location_bookings = df1_lower[mask]

                bookings = len(location_bookings)
                revenue = 0
                if revenue_col in df1_lower.columns:
                    revenue = pd.to_numeric(location_bookings[revenue_col], errors='coerce').fillna(0).sum()

                result[marketing_loc] = {
                    'bookings': bookings,
                    'revenue': revenue,
                }
        else:
            # Filter bookings matching any of the booking location names (case-insensitive)
            mask = df1_lower['_location_lower'].isin(booking_locs_lower)
            location_bookings = df1_lower[mask]

            bookings = len(location_bookings)
            revenue = 0
            if revenue_col in df1_lower.columns:
                revenue = pd.to_numeric(location_bookings[revenue_col], errors='coerce').fillna(0).sum()

            result[marketing_loc] = {
                'bookings': bookings,
                'revenue': revenue
            }

    return result


def get_capacity_per_location(marketing_loc, num_weeks=4, custom_capacity=None):
    """Get weekly capacity for a marketing location."""
    if marketing_loc not in LOCATION_NAME_MAP:
        return None

    # Check for custom capacity first
    if custom_capacity and marketing_loc in custom_capacity:
        total_weekly_capacity = custom_capacity[marketing_loc]
    else:
        # Calculate from LOCATION_CAPACITY
        booking_locs = LOCATION_NAME_MAP[marketing_loc]
        total_weekly_capacity = 0

        for booking_loc in booking_locs:
            if booking_loc in LOCATION_CAPACITY:
                cap = LOCATION_CAPACITY[booking_loc]
                total_weekly_capacity += cap['weekday'] + cap['weekend']

    # Return capacity for the period (num_weeks)
    return {
        'weekly_total': total_weekly_capacity,
        'period_total': total_weekly_capacity * num_weeks,
    }


def calculate_lead_time_by_location(df1, df2, location_col='Tour', date_col='Created'):
    """Calculate average lead time (booking to visit) per location.

    Returns dict of {location: {'avg_lead_time': X, 'median_lead_time': Y, 'bookings': Z}}
    """
    if df1 is None or df2 is None:
        return {}

    # Find common ID column
    id_col = None
    for col in ['Booking ID', 'booking_id', 'ID', 'id']:
        if col in df1.columns and col in df2.columns:
            id_col = col
            break

    if id_col is None:
        return {}

    # Merge df1 (booking dates) with df2 (visit dates)
    df1_prep = df1[[id_col, date_col, location_col]].copy()
    df1_prep.columns = ['booking_id', 'booking_date', 'location']

    # Find visit date column in df2
    visit_col = None
    for col in ['Activity start', 'Start', 'Visit date', 'Date']:
        if col in df2.columns:
            visit_col = col
            break
    if visit_col is None:
        return {}

    df2_prep = df2[[id_col, visit_col]].copy()
    df2_prep.columns = ['booking_id', 'visit_date']

    merged = df1_prep.merge(df2_prep, on='booking_id', how='inner')
    merged['booking_date'] = pd.to_datetime(merged['booking_date'], errors='coerce')
    merged['visit_date'] = pd.to_datetime(merged['visit_date'], errors='coerce')
    merged['interval_days'] = (merged['visit_date'] - merged['booking_date']).dt.days

    # Filter valid records
    merged = merged[(merged['interval_days'] >= 0) & (merged['interval_days'].notna())]

    if len(merged) == 0:
        return {}

    # Group by location
    result = {}
    for loc in merged['location'].unique():
        loc_data = merged[merged['location'] == loc]
        result[loc] = {
            'avg_lead_time': loc_data['interval_days'].mean(),
            'median_lead_time': loc_data['interval_days'].median(),
            'bookings': len(loc_data)
        }

    return result


def time_correlated_attribution(campaigns_df, bookings_df, location_col='Tour', date_col='Created',
                                 revenue_col='Total gross', conversion_window_days=14):
    """
    Correlate marketing campaigns with booking data using time-based attribution.

    For each campaign with date range:
    1. Find locations it targets (from campaign name)
    2. Count bookings made during campaign period + conversion window
    3. Calculate attributed revenue

    Returns DataFrame with campaign attribution metrics.
    """
    if campaigns_df is None or bookings_df is None:
        return None

    # Check required columns
    if 'report_start' not in campaigns_df.columns or 'report_end' not in campaigns_df.columns:
        return None

    # Prepare bookings data
    bookings = bookings_df.copy()
    bookings['booking_date'] = pd.to_datetime(bookings[date_col], errors='coerce')
    bookings['location'] = bookings[location_col].astype(str).str.lower()
    bookings['revenue'] = pd.to_numeric(bookings[revenue_col], errors='coerce').fillna(0)
    bookings = bookings[bookings['booking_date'].notna()]

    results = []

    for _, campaign in campaigns_df.iterrows():
        if pd.isna(campaign.get('report_start')) or pd.isna(campaign.get('report_end')):
            continue

        campaign_name = campaign.get('campaign_name', '')
        campaign_start = pd.to_datetime(campaign['report_start'])
        campaign_end = pd.to_datetime(campaign['report_end'])

        # Extended window for conversions
        attribution_end = campaign_end + pd.Timedelta(days=conversion_window_days)

        # Find target locations
        target_locations = campaign_matches_all_locations(campaign_name)
        is_all_locations = 'alle locaties' in str(campaign_name).lower() or 'all locations' in str(campaign_name).lower()

        # Filter bookings by date range
        date_mask = (bookings['booking_date'] >= campaign_start) & (bookings['booking_date'] <= attribution_end)
        period_bookings = bookings[date_mask]

        if is_all_locations:
            # Attribute to all locations proportionally
            attributed_bookings = len(period_bookings)
            attributed_revenue = period_bookings['revenue'].sum()
            target_locations_str = 'All Locations'
        elif target_locations:
            # Filter by target locations
            location_mask = period_bookings['location'].apply(
                lambda x: any(loc.lower() in x for loc in target_locations)
            )
            location_bookings = period_bookings[location_mask]
            attributed_bookings = len(location_bookings)
            attributed_revenue = location_bookings['revenue'].sum()
            target_locations_str = ', '.join(target_locations)
        else:
            # No location match - general/brand campaign
            attributed_bookings = 0
            attributed_revenue = 0
            target_locations_str = 'Unknown'

        results.append({
            'campaign_name': campaign_name,
            'Platform': campaign.get('Platform', ''),
            'spend': campaign.get('spend', 0),
            'campaign_start': campaign_start,
            'campaign_end': campaign_end,
            'target_locations': target_locations_str,
            'attributed_bookings': attributed_bookings,
            'attributed_revenue': attributed_revenue,
            'stdc_phase': campaign.get('stdc_phase', 'Untagged')
        })

    if not results:
        return None

    result_df = pd.DataFrame(results)

    # Calculate efficiency metrics
    result_df['roi'] = ((result_df['attributed_revenue'] - result_df['spend']) /
                        result_df['spend'].replace(0, float('nan')) * 100).fillna(0)
    result_df['cost_per_booking'] = (result_df['spend'] /
                                      result_df['attributed_bookings'].replace(0, float('nan'))).fillna(0)

    return result_df


def location_marketing_summary(attribution_df, bookings_df, location_col='Tour',
                                date_col='Created', revenue_col='Total gross'):
    """
    Aggregate marketing attribution by location.

    Returns DataFrame with per-location marketing metrics.
    """
    if attribution_df is None or len(attribution_df) == 0:
        return None

    # Get overall period from campaigns
    min_date = attribution_df['campaign_start'].min()
    max_date = attribution_df['campaign_end'].max()

    # Calculate total revenue per location for the period
    bookings = bookings_df.copy()
    bookings['booking_date'] = pd.to_datetime(bookings[date_col], errors='coerce')
    bookings['revenue'] = pd.to_numeric(bookings[revenue_col], errors='coerce').fillna(0)

    # Filter by campaign date range - but extend to include conversion window
    conversion_window = pd.Timedelta(days=14)
    bookings_in_period = bookings[
        (bookings['booking_date'] >= min_date) &
        (bookings['booking_date'] <= max_date + conversion_window)
    ]

    # If very few bookings in period, use full booking data with a warning
    if len(bookings_in_period) < 100:
        # Fall back to full data
        bookings_in_period = bookings[bookings['booking_date'].notna()]

    bookings_in_period = bookings_in_period.copy()
    bookings_in_period['_loc_lower'] = bookings_in_period[location_col].astype(str).str.lower()

    # Aggregate attribution by location
    location_results = []

    for marketing_loc, booking_locs in LOCATION_NAME_MAP.items():
        # Get campaigns targeting this location
        loc_campaigns = attribution_df[
            attribution_df['target_locations'].str.contains(marketing_loc, case=False, na=False) |
            (attribution_df['target_locations'] == 'All Locations')
        ]

        if len(loc_campaigns) == 0:
            continue

        # Only include spend from campaigns specifically targeting this location
        # Exclude "All Locations" campaigns as they can't be reliably attributed
        specific_campaigns = loc_campaigns[loc_campaigns['target_locations'] != 'All Locations']

        total_spend = specific_campaigns['spend'].sum()

        # Get booking location totals - use flexible matching
        # Match if booking location contains any of the key words from the marketing location
        loc_keywords = marketing_loc.lower().replace('amsterdam ', '').split()

        # Also include the full booking location names for exact matching
        booking_loc_lower = [loc.lower() for loc in booking_locs]

        # Create mask: exact match OR contains key location word
        mask = bookings_in_period['_loc_lower'].isin(booking_loc_lower)

        # Add partial matching for key terms
        for keyword in loc_keywords:
            if len(keyword) > 3:  # Only match meaningful keywords
                mask = mask | bookings_in_period['_loc_lower'].str.contains(keyword, case=False, na=False)

        loc_bookings = bookings_in_period[mask]

        location_results.append({
            'location': marketing_loc,
            'marketing_spend': total_spend,
            'total_bookings': len(loc_bookings),
            'total_revenue': loc_bookings['revenue'].sum(),
            'campaigns_count': len(loc_campaigns)
        })

    if not location_results:
        return None

    result_df = pd.DataFrame(location_results)

    # Calculate metrics
    result_df['revenue_per_spend'] = (result_df['total_revenue'] /
                                       result_df['marketing_spend'].replace(0, float('nan'))).fillna(0)
    result_df['cost_per_booking'] = (result_df['marketing_spend'] /
                                      result_df['total_bookings'].replace(0, float('nan'))).fillna(0)

    return result_df.sort_values('marketing_spend', ascending=False)


def _match_campaign_to_location(campaign_name):
    """Helper to match a campaign name to a single location. Returns (location, is_excluded)."""
    if pd.isna(campaign_name) or not isinstance(campaign_name, str):
        return None, True

    name_lower = campaign_name.lower()

    # Check if "alle locaties" campaign
    if 'alle locaties' in name_lower or 'all locations' in name_lower:
        return None, True

    # Find matching locations
    matched_locations = campaign_matches_all_locations(campaign_name)

    if len(matched_locations) == 1:
        return matched_locations[0], False
    else:
        # Multi-location or no match - exclude
        return None, True


@st.cache_data(show_spinner=False)
def get_marketing_by_location(_df_hash, df_json):
    """
    Aggregate marketing metrics by location from campaign data.
    Only includes campaigns that target a SINGLE specific location.
    Excludes "alle locaties" and multi-location campaigns.

    Uses caching for performance. Pass hash of df as _df_hash for cache key.

    Returns: DataFrame with location, spend, conversions, conversion_value, excluded_spend
    """
    campaigns_df = pd.read_json(StringIO(df_json))

    if campaigns_df is None or len(campaigns_df) == 0:
        return None, 0

    # Single apply call - get both location and is_excluded in one pass
    match_results = campaigns_df['campaign_name'].apply(_match_campaign_to_location)
    campaigns_df['_matched_loc'] = match_results.apply(lambda x: x[0])
    campaigns_df['_is_excluded'] = match_results.apply(lambda x: x[1])

    # Calculate excluded spend
    excluded_spend = campaigns_df.loc[campaigns_df['_is_excluded'], 'spend'].sum()

    # Filter to only matched campaigns
    matched_df = campaigns_df[campaigns_df['_matched_loc'].notna()].copy()

    if len(matched_df) == 0:
        return None, excluded_spend

    # Vectorized aggregation by location
    result_df = matched_df.groupby('_matched_loc').agg({
        'spend': 'sum',
        'clicks': 'sum',
        'conversions': 'sum',
        'conversion_value': 'sum',
        'campaign_name': 'count'
    }).reset_index()

    result_df.columns = ['location', 'ad_spend', 'clicks', 'conversions', 'conversion_value', 'campaigns']

    # Calculate CPA
    result_df['cpa'] = (result_df['ad_spend'] /
                        result_df['conversions'].replace(0, float('nan'))).fillna(0)

    return result_df, excluded_spend


def _build_location_matcher():
    """Pre-build a mapping from booking location patterns to marketing locations."""
    # Build reverse mapping: booking_loc_lower -> marketing_loc
    exact_matches = {}
    for marketing_loc, booking_locs in LOCATION_NAME_MAP.items():
        for booking_loc in booking_locs:
            exact_matches[booking_loc.lower()] = marketing_loc
    return exact_matches


# Pre-build location matcher once
_LOCATION_EXACT_MATCHES = _build_location_matcher()


def _map_booking_to_marketing_loc(booking_loc_lower):
    """Map a booking location to marketing location name."""
    if pd.isna(booking_loc_lower):
        return None

    # Try exact match first
    if booking_loc_lower in _LOCATION_EXACT_MATCHES:
        return _LOCATION_EXACT_MATCHES[booking_loc_lower]

    # Try keyword matching
    for marketing_loc, booking_locs in LOCATION_NAME_MAP.items():
        loc_keywords = marketing_loc.lower().replace('amsterdam ', '').split()
        for keyword in loc_keywords:
            if len(keyword) > 3 and keyword in booking_loc_lower:
                return marketing_loc

    return None


@st.cache_data(show_spinner=False)
def _get_revenue_by_location_cached(_cache_key, locations_json, revenues_json, start_date_str, end_date_str):
    """
    Cached revenue by location calculation.
    """
    locations = pd.read_json(StringIO(locations_json), typ='series')
    revenues = pd.read_json(StringIO(revenues_json), typ='series')

    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    # Build DataFrame
    df = pd.DataFrame({'location': locations, 'revenue': revenues})

    # Vectorized location mapping using .map() with pre-built dict
    df['_location_lower'] = df['location'].astype(str).str.lower()
    df['_marketing_loc'] = df['_location_lower'].map(_LOCATION_EXACT_MATCHES)

    # For unmatched, try keyword matching (only for those not in exact matches)
    unmatched_mask = df['_marketing_loc'].isna()
    if unmatched_mask.any():
        df.loc[unmatched_mask, '_marketing_loc'] = df.loc[unmatched_mask, '_location_lower'].apply(_map_booking_to_marketing_loc)

    # Filter to matched locations only
    matched_df = df[df['_marketing_loc'].notna()]

    if len(matched_df) == 0:
        return {loc: {'revenue': 0, 'bookings': 0} for loc in LOCATION_NAME_MAP.keys()}

    # Vectorized aggregation
    agg_df = matched_df.groupby('_marketing_loc').agg({
        'revenue': 'sum',
        '_marketing_loc': 'count'
    })
    agg_df.columns = ['revenue', 'bookings']

    # Convert to dict and fill missing locations with zeros
    result = agg_df.to_dict('index')
    for loc in LOCATION_NAME_MAP.keys():
        if loc not in result:
            result[loc] = {'revenue': 0, 'bookings': 0}

    return result


def get_revenue_by_location_filtered(df1, start_date, end_date, location_col='Location', date_col='Created', revenue_col='Total gross'):
    """
    Get revenue and bookings per location filtered by date range. Uses caching.

    Returns: dict of {marketing_location: {'revenue': X, 'bookings': Y}}
    """
    if df1 is None or len(df1) == 0:
        return {}

    # Detect location column
    if location_col not in df1.columns:
        for col in ['Location', 'Tour', 'Activity']:
            if col in df1.columns:
                location_col = col
                break
        else:
            return {}

    # Filter by date range first (before caching prep)
    df1_copy = df1.copy()
    df1_copy['_date'] = pd.to_datetime(df1_copy[date_col], errors='coerce')
    date_mask = (df1_copy['_date'] >= start_date) & (df1_copy['_date'] <= end_date)
    df = df1_copy.loc[date_mask]

    if len(df) == 0:
        return {}

    # Prepare data for cached function
    locations = df[location_col]
    revenues = pd.to_numeric(df[revenue_col], errors='coerce').fillna(0) if revenue_col in df.columns else pd.Series([0] * len(df))

    # Create cache key
    cache_key = hash((len(df), round(revenues.sum(), 2), str(start_date), str(end_date)))

    return _get_revenue_by_location_cached(
        cache_key,
        locations.to_json(),
        revenues.to_json(),
        str(start_date),
        str(end_date)
    )


def create_marketing_roi_table(marketing_df, revenue_data):
    """
    Combine marketing metrics with revenue data into a single table.

    Returns: DataFrame with location, revenue, bookings, clicks, conversions, conv_rate, conversion_value, ad_spend, cpa, roas
    """
    if marketing_df is None or len(marketing_df) == 0:
        return None

    # Convert revenue_data dict to DataFrame for vectorized merge
    rev_df = pd.DataFrame([
        {'location': loc, 'revenue': data['revenue'], 'bookings': data['bookings']}
        for loc, data in revenue_data.items()
    ])

    # Merge marketing data with revenue data (vectorized)
    result = marketing_df.merge(rev_df, on='location', how='left')
    result['revenue'] = result['revenue'].fillna(0)
    result['bookings'] = result['bookings'].fillna(0)

    # Vectorized calculations
    result['ROAS'] = (result['conversion_value'] / result['ad_spend'].replace(0, float('nan'))).fillna(0)
    result['Conv. Rate %'] = (result['conversions'] / result['clicks'].replace(0, float('nan')) * 100).fillna(0)

    # Rename and select columns
    result = result.rename(columns={
        'location': 'Location',
        'revenue': 'Revenue',
        'bookings': 'Bookings',
        'clicks': 'Clicks',
        'conversions': 'Conversions',
        'conversion_value': 'Conv. Value',
        'ad_spend': 'Ad Spend',
        'cpa': 'CPA'
    })

    # Select and order columns
    columns = ['Location', 'Revenue', 'Bookings', 'Clicks', 'Conversions', 'Conv. Rate %', 'Conv. Value', 'Ad Spend', 'CPA', 'ROAS']
    result = result[columns].sort_values('Ad Spend', ascending=False)

    return result


@st.cache_data(show_spinner=False)
def _calculate_cpa_metrics_cached(_df_hash, total_revenue, total_bookings, date_min_str, date_max_str,
                                   email_data_json, date_col, revenue_col):
    """
    Cached CPA metrics calculation. Called by calculate_cpa_metrics().
    """
    aov = total_revenue / total_bookings if total_bookings > 0 else 0

    # Calculate data span
    if date_min_str and date_max_str:
        min_date = pd.to_datetime(date_min_str)
        max_date = pd.to_datetime(date_max_str)
        data_span_days = (max_date - min_date).days
        data_span_months = data_span_days / 30.44
    else:
        data_span_months = 0

    # Calculate retention rate
    retention_rate = None
    cohort_size = 0
    returning_customers = 0

    if email_data_json and data_span_months >= 2:
        try:
            customer_data = pd.read_json(StringIO(email_data_json))
            customer_data['booking_date'] = pd.to_datetime(customer_data['booking_date'], errors='coerce')
            customer_data = customer_data.dropna(subset=['booking_date', 'email'])

            if len(customer_data) > 0:
                min_date = customer_data['booking_date'].min()
                first_month_start = (min_date + pd.offsets.MonthBegin(1)).normalize()
                first_month_end = (first_month_start + pd.offsets.MonthEnd(0)).normalize()

                first_booking_dates = customer_data.groupby('email')['booking_date'].min().reset_index()
                first_booking_dates.columns = ['email', 'first_booking']

                cohort_customers = first_booking_dates[
                    (first_booking_dates['first_booking'] >= first_month_start) &
                    (first_booking_dates['first_booking'] <= first_month_end)
                ]['email'].tolist()

                cohort_size = len(cohort_customers)

                if cohort_size > 0:
                    two_months_later = first_month_end + pd.DateOffset(months=2)
                    cohort_set = set(cohort_customers)

                    customer_data_ranked = customer_data.copy()
                    customer_data_ranked['booking_rank'] = customer_data_ranked.groupby('email')['booking_date'].rank(method='first')

                    second_bookings = customer_data_ranked[
                        (customer_data_ranked['email'].isin(cohort_set)) &
                        (customer_data_ranked['booking_rank'] == 2)
                    ]

                    returning_customers = (second_bookings['booking_date'] <= two_months_later).sum()
                    retention_rate = returning_customers / cohort_size
        except Exception:
            retention_rate = None

    return {
        'aov': aov,
        'retention_rate': retention_rate,
        'data_span_months': data_span_months,
        'has_sufficient_data': data_span_months >= 2,
        'total_bookings': total_bookings,
        'total_revenue': total_revenue,
        'cohort_size': cohort_size,
        'returning_customers': returning_customers
    }


def calculate_cpa_metrics(df1, df2=None, date_col='Created', revenue_col='Total gross', email_col='Email address'):
    """
    Calculate CPA-related metrics from booking data. Uses caching for performance.
    """
    if df1 is None or len(df1) == 0:
        return None

    if revenue_col not in df1.columns:
        return None

    # Pre-calculate hashable values
    total_revenue = pd.to_numeric(df1[revenue_col], errors='coerce').fillna(0).sum()
    total_bookings = len(df1)

    # Get date range
    date_min_str = None
    date_max_str = None
    if date_col in df1.columns:
        dates = pd.to_datetime(df1[date_col], errors='coerce')
        date_min_str = str(dates.min())
        date_max_str = str(dates.max())

    # Prepare email data for retention calculation
    email_data_json = None
    if email_col in df1.columns and date_col in df1.columns:
        email_data = df1[[email_col, date_col]].copy()
        email_data.columns = ['email', 'booking_date']
        email_data_json = email_data.to_json(date_format='iso')

    # Create hash for cache key
    df_hash = hash((total_bookings, round(total_revenue, 2), date_min_str, date_max_str))

    return _calculate_cpa_metrics_cached(
        df_hash, total_revenue, total_bookings, date_min_str, date_max_str,
        email_data_json, date_col, revenue_col
    )


def calculate_cpa_targets(aov, bedrijfskosten_pct, winstmarge_pct, retention_rate=None):
    """
    Calculate break-even and target CPA values.

    Args:
        aov: Average Order Value
        bedrijfskosten_pct: Operating costs percentage (0-100)
        winstmarge_pct: Target profit margin percentage (0-100)
        retention_rate: Optional retention rate for LTV calculations

    Returns dict with per-booking and LTV-based CPA targets
    """
    bedrijfskosten = bedrijfskosten_pct / 100
    winstmarge = winstmarge_pct / 100

    # Per-booking calculations
    breakeven_cpa = aov * (1 - bedrijfskosten)
    target_cpa = breakeven_cpa * (1 - winstmarge)

    result = {
        'breakeven_cpa': breakeven_cpa,
        'target_cpa': target_cpa,
    }

    # LTV-based calculations (only if retention rate available)
    if retention_rate is not None and retention_rate > 0 and retention_rate < 1:
        expected_bookings = 1 / (1 - retention_rate)
        ltv = aov * expected_bookings
        breakeven_cpa_ltv = ltv * (1 - bedrijfskosten)
        target_cpa_ltv = breakeven_cpa_ltv * (1 - winstmarge)

        result.update({
            'expected_bookings': expected_bookings,
            'ltv': ltv,
            'breakeven_cpa_ltv': breakeven_cpa_ltv,
            'target_cpa_ltv': target_cpa_ltv,
        })

    return result


@st.cache_data(show_spinner=False)
def calculate_marketing_metrics(_df_hash, df_json):
    """Calculate all marketing metrics with caching. Uses df_hash for cache key."""
    df = pd.read_json(StringIO(df_json))

    total_spend = df['spend'].sum()
    total_conversions = df['conversions'].sum()
    total_conv_value = df['conversion_value'].sum() if 'conversion_value' in df.columns else 0
    roas = (total_conv_value / total_spend * 100) if total_spend > 0 else 0
    cpa = (total_spend / total_conversions) if total_conversions > 0 else 0

    # Platform breakdown
    google_df = df[df['Platform'] == 'Google Ads']
    meta_df = df[df['Platform'] == 'Meta Ads']

    return {
        'total_spend': total_spend,
        'total_conversions': total_conversions,
        'total_conv_value': total_conv_value,
        'roas': roas,
        'cpa': cpa,
        'google_spend': google_df['spend'].sum(),
        'meta_spend': meta_df['spend'].sum(),
        'google_conv': google_df['conversions'].sum(),
        'meta_conv': meta_df['conversions'].sum(),
        'google_conv_value': google_df['conversion_value'].sum() if 'conversion_value' in google_df.columns else 0,
        'meta_conv_value': meta_df['conversion_value'].sum() if 'conversion_value' in meta_df.columns else 0,
    }


@st.cache_data(show_spinner=False)
def calculate_stdc_phase_metrics(_df_hash, df_json):
    """Calculate STDC phase metrics with caching."""
    df = pd.read_json(StringIO(df_json))

    # Ensure columns exist
    if 'impressions' not in df.columns:
        df['impressions'] = 0
    if 'reach' not in df.columns:
        df['reach'] = 0
    if 'clicks' not in df.columns:
        df['clicks'] = 0
    if 'conversions' not in df.columns:
        df['conversions'] = 0

    # Aggregate by phase
    phase_agg = df.groupby('stdc_phase').agg({
        'spend': 'sum',
        'impressions': 'sum',
        'reach': 'sum',
        'clicks': 'sum',
        'conversions': 'sum'
    }).to_dict('index')

    # Aggregate by phase + platform
    platform_agg = df.groupby(['stdc_phase', 'Platform']).agg({
        'spend': 'sum',
        'impressions': 'sum',
        'reach': 'sum',
        'clicks': 'sum',
        'conversions': 'sum'
    })

    results = {}
    default_metrics = {'spend': 0, 'reach': 0, 'clicks': 0, 'conversions': 0, 'cpm': 0, 'ctr': 0, 'cpa': 0, 'conv_rate': 0,
                       'google': {'spend': 0, 'reach': 0, 'clicks': 0, 'conversions': 0, 'cpm': 0, 'ctr': 0, 'cpa': 0, 'conv_rate': 0},
                       'meta': {'spend': 0, 'reach': 0, 'clicks': 0, 'conversions': 0, 'cpm': 0, 'ctr': 0, 'cpa': 0, 'conv_rate': 0}}

    for phase in ['SEE', 'THINK', 'DO', 'CARE', 'Untagged']:
        if phase in phase_agg:
            p = phase_agg[phase]
            total_reach = p['impressions'] + p['reach']
            spend, clicks, conversions = p['spend'], p['clicks'], p['conversions']
        else:
            total_reach = spend = clicks = conversions = 0

        cpm = (spend / total_reach * 1000) if total_reach > 0 else 0
        ctr = (clicks / total_reach * 100) if total_reach > 0 else 0
        cpa = (spend / conversions) if conversions > 0 else 0
        conv_rate = (conversions / clicks * 100) if clicks > 0 else 0

        g_data = platform_agg.loc[(phase, 'Google Ads')] if (phase, 'Google Ads') in platform_agg.index else {'spend': 0, 'impressions': 0, 'reach': 0, 'clicks': 0, 'conversions': 0}
        m_data = platform_agg.loc[(phase, 'Meta Ads')] if (phase, 'Meta Ads') in platform_agg.index else {'spend': 0, 'impressions': 0, 'reach': 0, 'clicks': 0, 'conversions': 0}

        g_spend, g_impr, g_clicks, g_conv = g_data['spend'], g_data['impressions'], g_data['clicks'], g_data['conversions']
        m_spend, m_reach, m_clicks, m_conv = m_data['spend'], m_data['reach'], m_data['clicks'], m_data['conversions']

        g_cpm = (g_spend / g_impr * 1000) if g_impr > 0 else 0
        m_cpm = (m_spend / m_reach * 1000) if m_reach > 0 else 0
        g_ctr = (g_clicks / g_impr * 100) if g_impr > 0 else 0
        m_ctr = (m_clicks / m_reach * 100) if m_reach > 0 else 0
        g_cpa = (g_spend / g_conv) if g_conv > 0 else 0
        m_cpa = (m_spend / m_conv) if m_conv > 0 else 0
        g_conv_rate = (g_conv / g_clicks * 100) if g_clicks > 0 else 0
        m_conv_rate = (m_conv / m_clicks * 100) if m_clicks > 0 else 0

        results[phase] = {
            'spend': spend, 'reach': total_reach, 'clicks': clicks, 'conversions': conversions,
            'cpm': cpm, 'ctr': ctr, 'cpa': cpa, 'conv_rate': conv_rate,
            'google': {'spend': g_spend, 'reach': g_impr, 'clicks': g_clicks, 'conversions': g_conv, 'cpm': g_cpm, 'ctr': g_ctr, 'cpa': g_cpa, 'conv_rate': g_conv_rate},
            'meta': {'spend': m_spend, 'reach': m_reach, 'clicks': m_clicks, 'conversions': m_conv, 'cpm': m_cpm, 'ctr': m_ctr, 'cpa': m_cpa, 'conv_rate': m_conv_rate}
        }
    return results


@st.cache_data(show_spinner=False)
def parse_google_ads_csv(uploaded_file):
    """Parse Google Ads CSV export (has 2-line header)."""
    try:
        content = uploaded_file.getvalue().decode('utf-8')
        lines = content.split('\n')

        # Skip first 2 lines (title and date range)
        csv_content = '\n'.join(lines[2:])

        df = pd.read_csv(StringIO(csv_content))
        df['Platform'] = 'Google Ads'

        # Standardize column names (with fallback variations)
        column_mapping = {
            'Campaign': 'campaign_name',
            'Cost': 'spend',
            'Conversions': 'conversions',
            'Conv. value': 'conversion_value',
            'Conv. Value': 'conversion_value',
            'Impr.': 'impressions',
            'Impressions': 'impressions',
            'Clicks': 'clicks',
            'CTR': 'ctr',
            'Avg. CPC': 'cpc',
            'CPC': 'cpc'
        }

        df = df.rename(columns=column_mapping)

        # Filter out summary/total rows (Total is in Campaign status column)
        if 'Campaign status' in df.columns:
            df = df[~df['Campaign status'].astype(str).str.startswith('Total:')]

        # Clean numeric columns
        for col in ['spend', 'conversions', 'conversion_value', 'impressions', 'clicks', 'cpc']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.replace(' --', '0'), errors='coerce').fillna(0)

        # Clean CTR
        if 'ctr' in df.columns:
            df['ctr'] = df['ctr'].astype(str).str.replace('%', '').str.replace(' --', '0')
            df['ctr'] = pd.to_numeric(df['ctr'], errors='coerce').fillna(0)

        return df, None
    except Exception as e:
        return None, str(e)


@st.cache_data(show_spinner=False)
def parse_meta_ads_csv(uploaded_file):
    """Parse Meta Ads CSV export."""
    try:
        df = pd.read_csv(uploaded_file)
        df['Platform'] = 'Meta Ads'

        # Standardize column names
        # Note: 'Purchases' is actual conversions, 'Results' is campaign objective result
        column_mapping = {
            'Campaign name': 'campaign_name',
            'Amount spent (EUR)': 'spend',
            'Purchases': 'conversions',  # Use Purchases for actual conversions
            'Purchases conversion value': 'conversion_value',
            'Reach': 'reach',
            'Link clicks': 'clicks',
            'CTR (link click-through rate)': 'ctr',
            'CPC (cost per link click) (EUR)': 'cpc',
            'CPM (cost per 1,000 impressions) (EUR)': 'cpm',
            'Results': 'results'  # Keep results separate (campaign objective)
        }

        df = df.rename(columns=column_mapping)

        # Clean numeric columns
        for col in ['spend', 'conversions', 'conversion_value', 'reach', 'clicks', 'cpc', 'cpm', 'purchases']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Clean CTR
        if 'ctr' in df.columns:
            df['ctr'] = pd.to_numeric(df['ctr'], errors='coerce').fillna(0)

        return df, None
    except Exception as e:
        return None, str(e)


# Header with logo
col1, col2 = st.columns([1, 5])
with col1:
    st.image("assets/logo_black.svg", width=120)
with col2:
    st.title("Kuuma Booking Analyzer")
    st.markdown("**Customer insights & booking intelligence**")

st.markdown("## Marketing Campaign Analysis")
st.markdown("Analyze Google Ads and Meta Ads performance using the SEE-THINK-DO-CARE framework")

# Reserve container for date range selector (filled after data loads)
date_range_container = st.container()

# Default file paths for auto-loading (only used for local development)
import os
DEFAULT_DATA_PATH = "/Users/raouldevries/Work/Kuuma/Booking analyzer"
DEFAULT_GOOGLE_ADS = f"{DEFAULT_DATA_PATH}/marketing data/google_ads.csv"
DEFAULT_META_ADS = f"{DEFAULT_DATA_PATH}/marketing data/meta_ads.csv"
DEFAULT_BOOKING_CREATION = f"{DEFAULT_DATA_PATH}/booking data/the day the booking was made.xls"
DEFAULT_VISIT_DATES = f"{DEFAULT_DATA_PATH}/booking data/the date booked.xls"

@st.cache_data(show_spinner=False)
def _load_local_excel(file_path, engine='xlrd'):
    """Load Excel file with caching."""
    if os.path.exists(file_path):
        return pd.read_excel(file_path, engine=engine)
    return None

@st.cache_data(show_spinner=False)
def _load_local_csv(file_path):
    """Load and process local CSV file with caching."""
    if not os.path.exists(file_path):
        return None
    return pd.read_csv(file_path)

def load_default_files():
    """Auto-load default data files if they exist (local development only)."""
    # Skip if already attempted loading
    if st.session_state.get('_local_files_checked', False):
        return
    st.session_state._local_files_checked = True

    # Load booking data (df1 - booking creation dates)
    if st.session_state.df1 is None:
        df = _load_local_excel(DEFAULT_BOOKING_CREATION)
        if df is not None:
            st.session_state.df1 = df

    # Load booking data (df2 - visit dates)
    if st.session_state.df2 is None:
        df = _load_local_excel(DEFAULT_VISIT_DATES)
        if df is not None:
            st.session_state.df2 = df

    # Load Google Ads data
    if st.session_state.google_ads_df is None:
        df = _load_local_csv(DEFAULT_GOOGLE_ADS)
        if df is not None:
            # Process Google Ads CSV (skip header rows already handled by pandas)
            df['Platform'] = 'Google Ads'
            column_mapping = {
                'Campaign': 'campaign_name',
                'Cost': 'spend',
                'Conversions': 'conversions',
                'Conv. value': 'conversion_value',
                'Conv. Value': 'conversion_value',
                'Impr.': 'impressions',
                'Impressions': 'impressions',
                'Clicks': 'clicks',
                'CTR': 'ctr',
                'Avg. CPC': 'cpc',
                'CPC': 'cpc'
            }
            df = df.rename(columns=column_mapping)
            if 'Campaign status' in df.columns:
                df = df[~df['Campaign status'].astype(str).str.startswith('Total:')]
            for col in ['spend', 'conversions', 'conversion_value', 'impressions', 'clicks', 'cpc']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.replace(' --', '0'), errors='coerce').fillna(0)
            if 'ctr' in df.columns:
                df['ctr'] = pd.to_numeric(df['ctr'].astype(str).str.replace('%', '').str.replace(' --', '0'), errors='coerce').fillna(0)
            st.session_state.google_ads_df = df

    # Load Meta Ads data
    if st.session_state.meta_ads_df is None:
        df = _load_local_csv(DEFAULT_META_ADS)
        if df is not None:
            df['Platform'] = 'Meta Ads'
            column_mapping = {
                'Campaign name': 'campaign_name',
                'Amount spent (EUR)': 'spend',
                'Purchases': 'conversions',
                'Purchases conversion value': 'conversion_value',
                'Reach': 'reach',
                'Link clicks': 'clicks',
                'CTR (link click-through rate)': 'ctr',
                'CPC (cost per link click) (EUR)': 'cpc',
                'CPM (cost per 1,000 impressions) (EUR)': 'cpm',
                'Results': 'results'
            }
            df = df.rename(columns=column_mapping)
            if 'campaign_name' in df.columns:
                df = df[df['campaign_name'].notna()]
                df = df[df['campaign_name'].astype(str).str.strip() != '']
            for col in ['spend', 'conversions', 'conversion_value', 'reach', 'clicks', 'cpc', 'cpm']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            if 'ctr' in df.columns:
                df['ctr'] = pd.to_numeric(df['ctr'], errors='coerce').fillna(0)
            st.session_state.meta_ads_df = df

# Initialize session state using centralized function
init_session_state()

# Check authentication
if not st.session_state.get('authenticated', False):
    st.warning("Please log in to access this page.")
    st.page_link("app.py", label="Go to Login", icon=":material/login:")
    st.stop()

# Auto-load default files
load_default_files()

# Always reset STDC tags to apply latest auto-tagging defaults
st.session_state.stdc_tags = {}

# Reserve container for navigation at top of sidebar
nav_container = st.sidebar.container()

# Sidebar - Upload section
st.sidebar.header("Data")

# Show loaded status
if st.session_state.google_ads_df is not None:
    st.sidebar.success(f"Google Ads: {len(st.session_state.google_ads_df)} campaigns")
if st.session_state.meta_ads_df is not None:
    st.sidebar.success(f"Meta Ads: {len(st.session_state.meta_ads_df)} campaigns")
if st.session_state.df1 is not None:
    st.sidebar.success(f"Booking data: {len(st.session_state.df1):,} rows")

# Manual upload option (in expander)
with st.sidebar.expander("Upload different files"):
    # Google Ads uploader (multiple files)
    uploaded_google_files = st.file_uploader(
        "Google Ads CSV",
        type=["csv"],
        help="Export from Google Ads campaign report. You can select multiple files.",
        key="google_ads_file",
        accept_multiple_files=True
    )

    if uploaded_google_files:
        dfs = []
        errors = []
        for uploaded_file in uploaded_google_files:
            df, error = parse_google_ads_csv(uploaded_file)
            if error:
                errors.append(f"{uploaded_file.name}: {error}")
            elif df is not None:
                dfs.append(df)

        if errors:
            for error in errors:
                st.error(f"Google Ads error: {error}")
        elif dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            st.session_state.google_ads_df = merged_df
            if len(dfs) == 1:
                st.success(f"Google Ads loaded: {len(merged_df)} campaigns")
            else:
                st.success(f"Google Ads merged {len(dfs)} files: {len(merged_df)} campaigns")

    # Meta Ads uploader (multiple files)
    uploaded_meta_files = st.file_uploader(
        "Meta Ads CSV",
        type=["csv"],
        help="Export from Meta Ads Manager. You can select multiple files.",
        key="meta_ads_file",
        accept_multiple_files=True
    )

    if uploaded_meta_files:
        dfs = []
        errors = []
        for uploaded_file in uploaded_meta_files:
            df, error = parse_meta_ads_csv(uploaded_file)
            if error:
                errors.append(f"{uploaded_file.name}: {error}")
            elif df is not None:
                dfs.append(df)

        if errors:
            for error in errors:
                st.error(f"Meta Ads error: {error}")
        elif dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            st.session_state.meta_ads_df = merged_df
            if len(dfs) == 1:
                st.success(f"Meta Ads loaded: {len(merged_df)} campaigns")
            else:
                st.success(f"Meta Ads merged {len(dfs)} files: {len(merged_df)} campaigns")

# Fill navigation container
if st.session_state.df2 is not None or st.session_state.google_ads_df is not None or st.session_state.meta_ads_df is not None:
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
if st.session_state.google_ads_df is None and st.session_state.meta_ads_df is None:
    st.info("**No marketing data loaded.** Please upload Google Ads and/or Meta Ads CSV files using the sidebar.")
    st.markdown("""
    ### Getting Started
    1. Upload Google Ads CSV export
    2. Upload Meta Ads CSV export
    3. Tag campaigns with SEE-THINK-DO-CARE phases
    4. Analyze performance across the customer journey

    ### SEE-THINK-DO-CARE Framework
    - **SEE**: Awareness stage - reaching broad audiences (Display, Reach campaigns)
    - **THINK**: Consideration stage - engaging interested users (Non-branded, Clicks campaigns)
    - **DO**: Conversion stage - driving actions (Branded, Conversion campaigns)
    - **CARE**: Loyalty stage - retaining customers (Retargeting, Remarketing campaigns)
    """)
else:
    # Show loading message while processing
    loading_placeholder = st.empty()
    loading_placeholder.info("Loading marketing analysis...")

    # Combine data from both platforms
    dfs_to_combine = []

    if st.session_state.google_ads_df is not None:
        dfs_to_combine.append(st.session_state.google_ads_df)

    if st.session_state.meta_ads_df is not None:
        dfs_to_combine.append(st.session_state.meta_ads_df)

    combined_df = pd.concat(dfs_to_combine, ignore_index=True)

    # Extract date range from data
    data_min_date = None
    data_max_date = None

    # Try to get dates from Meta Ads data
    if st.session_state.meta_ads_df is not None:
        meta_df_dates = st.session_state.meta_ads_df
        if 'Reporting starts' in meta_df_dates.columns and 'Reporting ends' in meta_df_dates.columns:
            try:
                start_dates = pd.to_datetime(meta_df_dates['Reporting starts'], errors='coerce')
                end_dates = pd.to_datetime(meta_df_dates['Reporting ends'], errors='coerce')
                data_min_date = start_dates.min()
                data_max_date = end_dates.max()
            except:
                pass

    # Add date columns to combined_df for filtering
    if 'Reporting starts' in combined_df.columns:
        combined_df['report_start'] = pd.to_datetime(combined_df['Reporting starts'], errors='coerce')
    if 'Reporting ends' in combined_df.columns:
        combined_df['report_end'] = pd.to_datetime(combined_df['Reporting ends'], errors='coerce')

    # Filter out campaigns with 0 spend or invalid names
    if 'spend' in combined_df.columns:
        combined_df = combined_df[combined_df['spend'] > 0]
    if 'campaign_name' in combined_df.columns:
        combined_df = combined_df[combined_df['campaign_name'].notna()]
        combined_df = combined_df[combined_df['campaign_name'].astype(str).str.strip() != '']
        # Filter out summary rows like "--" which are not actual campaigns
        combined_df = combined_df[~combined_df['campaign_name'].astype(str).str.strip().str.match(r'^-+$')]
        combined_df = combined_df[~combined_df['campaign_name'].astype(str).str.contains(r'^\s*-+\s*$', regex=True)]

    # Get locations from booking data
    available_locations = []
    if st.session_state.df2 is not None:
        if 'Location' in st.session_state.df2.columns:
            location_col = 'Location'
        elif 'Activity' in st.session_state.df2.columns:
            location_col = 'Activity'
        elif 'Tour' in st.session_state.df2.columns:
            location_col = 'Tour'
        else:
            location_col = None
        if location_col:
            available_locations = st.session_state.df2[location_col].unique().tolist()
            available_locations = [loc for loc in available_locations if loc in LOCATION_KEYWORDS]

    # Filter settings in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")

    show_all_campaigns = st.sidebar.checkbox(
        "Show all campaigns",
        value=True,
        help="Show all campaigns regardless of location"
    )

    # Filter campaigns by location (only if not showing all)
    if not show_all_campaigns and available_locations:
        # Vectorized location matching
        def matches_any_location(campaign_name):
            if pd.isna(campaign_name) or not isinstance(campaign_name, str):
                return False
            return campaign_matches_location(campaign_name, available_locations) is not None

        # Apply vectorized filter
        mask = combined_df['campaign_name'].apply(matches_any_location)
        filtered_df = combined_df[mask]

        if len(filtered_df) > 0:
            combined_df = filtered_df
            st.sidebar.info(f"Showing {len(combined_df)} campaigns matching your locations")
        else:
            st.warning("No campaigns match the locations in your booking data.")
            combined_df = pd.DataFrame()
    else:
        st.sidebar.info(f"Showing all {len(combined_df)} campaigns")

    if len(combined_df) > 0:
        # Check if campaign_name column exists
        if 'campaign_name' not in combined_df.columns:
            st.error("Marketing data is missing 'campaign_name' column. Please reboot the app in Streamlit Cloud (Manage app → Reboot) to reload data with correct column mapping.")
            st.write("**Available columns:**", list(combined_df.columns))
            st.stop()

        # Initialize STDC tags for new campaigns
        for campaign in combined_df['campaign_name'].unique():
            if campaign not in st.session_state.stdc_tags:
                st.session_state.stdc_tags[campaign] = suggest_stdc_phase(campaign)

        # Add STDC phase to dataframe
        combined_df['stdc_phase'] = combined_df['campaign_name'].map(st.session_state.stdc_tags)

        # Date range selector
        if pd.notna(data_min_date) and pd.notna(data_max_date):
            with date_range_container:
                date_col1, date_col2 = st.columns([2, 4])
                with date_col1:
                    date_range = st.date_input(
                        "Date Range",
                        value=(data_min_date.date(), data_max_date.date()),
                        min_value=data_min_date.date(),
                        max_value=data_max_date.date(),
                        help="Filter marketing data by reporting period",
                        key="mkt_date_range"
                    )

                # Apply date filter if dates are selected
                if len(date_range) == 2 and 'report_start' in combined_df.columns:
                    start_date, end_date = date_range
                    # Only filter rows that have date info (Meta Ads)
                    # Keep rows without dates (Google Ads) - they don't have reporting period columns
                    has_dates = combined_df['report_start'].notna() & combined_df['report_end'].notna()
                    date_filter = (
                        (combined_df['report_start'].dt.date <= end_date) &
                        (combined_df['report_end'].dt.date >= start_date)
                    )
                    # Keep rows without dates OR rows that pass the date filter
                    combined_df = combined_df[~has_dates | date_filter]

        # Clear loading message now that data is ready
        loading_placeholder.empty()

        # Key Metrics
        st.markdown("### Key Metrics")

        total_spend = combined_df['spend'].sum()
        total_conversions = combined_df['conversions'].sum()
        total_conv_value = combined_df['conversion_value'].sum() if 'conversion_value' in combined_df.columns else 0
        roas = (total_conv_value / total_spend * 100) if total_spend > 0 else 0
        cpa = (total_spend / total_conversions) if total_conversions > 0 else 0

        # Calculate platform distribution for all metrics
        google_df = combined_df[combined_df['Platform'] == 'Google Ads']
        meta_df = combined_df[combined_df['Platform'] == 'Meta Ads']

        google_spend = google_df['spend'].sum()
        meta_spend = meta_df['spend'].sum()
        google_conv = google_df['conversions'].sum()
        meta_conv = meta_df['conversions'].sum()
        google_conv_value = google_df['conversion_value'].sum() if 'conversion_value' in google_df.columns else 0
        meta_conv_value = meta_df['conversion_value'].sum() if 'conversion_value' in meta_df.columns else 0
        google_roas = (google_conv_value / google_spend * 100) if google_spend > 0 else 0
        meta_roas = (meta_conv_value / meta_spend * 100) if meta_spend > 0 else 0
        google_cpa = (google_spend / google_conv) if google_conv > 0 else 0
        meta_cpa = (meta_spend / meta_conv) if meta_conv > 0 else 0

        # Helper for platform split tooltip
        def platform_tooltip(g_val, m_val, fmt='currency'):
            if fmt == 'currency':
                g_str = f"€{g_val:,.0f}" if g_val > 0 else "-"
                m_str = f"€{m_val:,.0f}" if m_val > 0 else "-"
            elif fmt == 'percent':
                g_str = f"{g_val:.1f}%" if g_val > 0 else "-"
                m_str = f"{m_val:.1f}%" if m_val > 0 else "-"
            else:
                g_str = f"{g_val:,.0f}" if g_val > 0 else "-"
                m_str = f"{m_val:,.0f}" if m_val > 0 else "-"
            return f"G Ads: {g_str} · M Ads: {m_str}"

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Spend", f"€{total_spend:,.0f}", help=platform_tooltip(google_spend, meta_spend))
        with col2:
            st.metric("Conversions", f"{total_conversions:,.0f}", help=platform_tooltip(google_conv, meta_conv, 'number'))
        with col3:
            st.metric("Conv. Value", f"€{total_conv_value:,.0f}", help=platform_tooltip(google_conv_value, meta_conv_value))
        with col4:
            st.metric("ROAS", f"{roas:,.0f}%", help=platform_tooltip(google_roas, meta_roas, 'percent'))
        with col5:
            st.metric("Avg CPA", f"€{cpa:,.0f}", help=platform_tooltip(google_cpa, meta_cpa))

        # ===== VISUALIZATION B: Stage KPI Cards =====
        st.markdown("---")
        st.markdown("### STDC Performance")

        # Show loading spinner for STDC section
        stdc_loading = st.empty()
        with stdc_loading:
            st.markdown("""
            <div style="display: flex; align-items: center; padding: 1rem; background-color: #e3f2fd; border-radius: 8px; margin: 1rem 0;">
                <div style="width: 24px; height: 24px; border: 3px solid #1976d2; border-top-color: transparent; border-radius: 50%; animation: spin 1s linear infinite; margin-right: 12px;"></div>
                <span style="color: #1976d2; font-weight: 500;">Calculating STDC metrics...</span>
            </div>
            <style>
                @keyframes spin {
                    to { transform: rotate(360deg); }
                }
            </style>
            """, unsafe_allow_html=True)

        def format_platform_tooltip(google_val, meta_val, fmt='number'):
            """Format platform split as tooltip text."""
            if fmt == 'currency':
                g_str = f"€{google_val:,.0f}" if google_val > 0 else "-"
                m_str = f"€{meta_val:,.0f}" if meta_val > 0 else "-"
            elif fmt == 'percent':
                g_str = f"{google_val:.1f}%" if google_val > 0 else "-"
                m_str = f"{meta_val:.1f}%" if meta_val > 0 else "-"
            else:
                g_str = f"{google_val:,.0f}" if google_val > 0 else "-"
                m_str = f"{meta_val:,.0f}" if meta_val > 0 else "-"
            return f"G Ads: {g_str} · M Ads: {m_str}"

        # Calculate all phase metrics using cached function
        # Create hash for cache key based on relevant columns
        stdc_cols = ['stdc_phase', 'Platform', 'spend', 'impressions', 'reach', 'clicks', 'conversions']
        stdc_cols = [c for c in stdc_cols if c in combined_df.columns]
        df_for_stdc = combined_df[stdc_cols].copy()
        df_hash = hash(df_for_stdc.to_json())
        all_phase_metrics = calculate_stdc_phase_metrics(df_hash, df_for_stdc.to_json())
        see_metrics = all_phase_metrics.get('SEE', {'spend': 0, 'reach': 0, 'clicks': 0, 'conversions': 0, 'cpm': 0, 'ctr': 0, 'cpa': 0, 'conv_rate': 0, 'google': {'spend': 0, 'reach': 0, 'clicks': 0, 'conversions': 0, 'cpm': 0, 'ctr': 0, 'cpa': 0, 'conv_rate': 0}, 'meta': {'spend': 0, 'reach': 0, 'clicks': 0, 'conversions': 0, 'cpm': 0, 'ctr': 0, 'cpa': 0, 'conv_rate': 0}})
        think_metrics = all_phase_metrics.get('THINK', {'spend': 0, 'reach': 0, 'clicks': 0, 'conversions': 0, 'cpm': 0, 'ctr': 0, 'cpa': 0, 'conv_rate': 0, 'google': {'spend': 0, 'reach': 0, 'clicks': 0, 'conversions': 0, 'cpm': 0, 'ctr': 0, 'cpa': 0, 'conv_rate': 0}, 'meta': {'spend': 0, 'reach': 0, 'clicks': 0, 'conversions': 0, 'cpm': 0, 'ctr': 0, 'cpa': 0, 'conv_rate': 0}})
        do_metrics = all_phase_metrics.get('DO', {'spend': 0, 'reach': 0, 'clicks': 0, 'conversions': 0, 'cpm': 0, 'ctr': 0, 'cpa': 0, 'conv_rate': 0, 'google': {'spend': 0, 'reach': 0, 'clicks': 0, 'conversions': 0, 'cpm': 0, 'ctr': 0, 'cpa': 0, 'conv_rate': 0}, 'meta': {'spend': 0, 'reach': 0, 'clicks': 0, 'conversions': 0, 'cpm': 0, 'ctr': 0, 'cpa': 0, 'conv_rate': 0}})
        care_metrics = all_phase_metrics.get('CARE', {'spend': 0, 'reach': 0, 'clicks': 0, 'conversions': 0, 'cpm': 0, 'ctr': 0, 'cpa': 0, 'conv_rate': 0, 'google': {'spend': 0, 'reach': 0, 'clicks': 0, 'conversions': 0, 'cpm': 0, 'ctr': 0, 'cpa': 0, 'conv_rate': 0}, 'meta': {'spend': 0, 'reach': 0, 'clicks': 0, 'conversions': 0, 'cpm': 0, 'ctr': 0, 'cpa': 0, 'conv_rate': 0}})

        # Save original THINK spend (from THINK-tagged campaigns only)
        original_think_spend = think_metrics['spend']
        original_think_google_spend = think_metrics['google']['spend']
        original_think_meta_spend = think_metrics['meta']['spend']

        # Override THINK clicks/CTR to use ALL campaigns (engagement metrics)
        # but keep spend from THINK-tagged campaigns only (to avoid double counting)
        all_impressions = combined_df['impressions'].sum() if 'impressions' in combined_df.columns else 0
        all_reach = combined_df['reach'].sum() if 'reach' in combined_df.columns else 0
        all_total_reach = all_impressions + all_reach
        all_clicks = combined_df['clicks'].sum() if 'clicks' in combined_df.columns else 0

        # Platform breakdown for all campaigns (clicks/CTR only)
        google_all = combined_df[combined_df['Platform'] == 'Google Ads']
        meta_all = combined_df[combined_df['Platform'] == 'Meta Ads']
        g_all_impr = google_all['impressions'].sum() if 'impressions' in google_all.columns else 0
        m_all_reach = meta_all['reach'].sum() if 'reach' in meta_all.columns else 0
        g_all_clicks = google_all['clicks'].sum() if 'clicks' in google_all.columns else 0
        m_all_clicks = meta_all['clicks'].sum() if 'clicks' in meta_all.columns else 0

        # Calculate CTR for all campaigns
        all_ctr = (all_clicks / all_total_reach * 100) if all_total_reach > 0 else 0
        g_all_ctr = (g_all_clicks / g_all_impr * 100) if g_all_impr > 0 else 0
        m_all_ctr = (m_all_clicks / m_all_reach * 100) if m_all_reach > 0 else 0

        # Update think_metrics: clicks/CTR from all campaigns, spend from THINK-tagged only
        think_metrics['clicks'] = all_clicks
        think_metrics['ctr'] = all_ctr
        think_metrics['spend'] = original_think_spend  # Keep original THINK spend
        think_metrics['google']['clicks'] = g_all_clicks
        think_metrics['google']['ctr'] = g_all_ctr
        think_metrics['google']['spend'] = original_think_google_spend  # Keep original
        think_metrics['meta']['clicks'] = m_all_clicks
        think_metrics['meta']['ctr'] = m_all_ctr
        think_metrics['meta']['spend'] = original_think_meta_spend  # Keep original

        # Calculate rebooking metrics and CLV from booking data if available
        rebookings = 0
        rebook_rate = 0
        total_customers = 0
        clv = 0
        if st.session_state.df1 is not None:
            df1 = st.session_state.df1
            email_col = 'Email address' if 'Email address' in df1.columns else None
            revenue_col = 'Total gross' if 'Total gross' in df1.columns else None
            date_col = 'Created' if 'Created' in df1.columns else None

            if email_col:
                customer_bookings = df1.groupby(email_col).size()
                total_customers = len(customer_bookings)
                repeat_customers = (customer_bookings > 1).sum()
                rebookings = customer_bookings[customer_bookings > 1].sum()
                rebook_rate = (repeat_customers / total_customers * 100) if total_customers > 0 else 0

                # Calculate CLV if revenue and date columns exist
                if revenue_col and date_col:
                    # Filter valid data
                    clv_data = df1[[email_col, revenue_col, date_col]].copy()
                    clv_data.columns = ['email', 'revenue', 'booking_date']
                    clv_data['revenue'] = pd.to_numeric(clv_data['revenue'], errors='coerce').fillna(0)
                    clv_data['booking_date'] = pd.to_datetime(clv_data['booking_date'], errors='coerce')
                    clv_data = clv_data[clv_data['booking_date'].notna() & clv_data['email'].notna()]

                    if len(clv_data) > 0:
                        # Step 1: Average Order Value
                        aov = clv_data['revenue'].sum() / len(clv_data) if len(clv_data) > 0 else 0

                        # Step 2: Retention Rate (cohort-based)
                        min_date = clv_data['booking_date'].min()
                        max_date = clv_data['booking_date'].max()

                        first_month_start = min_date.replace(day=1)
                        if min_date.day > 1:
                            first_month_start = first_month_start + pd.DateOffset(months=1)
                        first_month_end = first_month_start + pd.DateOffset(months=1) - pd.Timedelta(days=1)
                        two_months_later = first_month_start + pd.DateOffset(months=3) - pd.Timedelta(days=1)

                        # Find customers whose first booking was in cohort month
                        customer_first = clv_data.groupby('email')['booking_date'].min().reset_index()
                        customer_first.columns = ['email', 'first_booking']
                        cohort_customers = customer_first[
                            (customer_first['first_booking'] >= first_month_start) &
                            (customer_first['first_booking'] <= first_month_end)
                        ]['email'].tolist()

                        cohort_size = len(cohort_customers)
                        if cohort_size > 0:
                            # Vectorized approach: get all bookings for cohort customers
                            cohort_set = set(cohort_customers)
                            cohort_bookings = clv_data[clv_data['email'].isin(cohort_set)].copy()

                            # Rank bookings per customer by date
                            cohort_bookings['booking_rank'] = cohort_bookings.groupby('email')['booking_date'].rank(method='first')

                            # Get second bookings only
                            second_bookings = cohort_bookings[cohort_bookings['booking_rank'] == 2]

                            # Count customers whose second booking was within the time window
                            returning_customers = (second_bookings['booking_date'] <= two_months_later).sum()

                            retention_rate = returning_customers / cohort_size
                        else:
                            retention_rate = 0.3  # Default fallback

                        # Step 3: Churn Rate
                        churn_rate = 1 - retention_rate

                        # Step 4: CLV = AOV × (1 + retention_rate / churn_rate)
                        if churn_rate > 0:
                            clv = aov * (1 + retention_rate / churn_rate)
                        else:
                            clv = aov * 10  # Cap at 10x AOV if no churn

        # Clear loading placeholder - metrics are ready
        stdc_loading.empty()

        # Stage KPI Cards with colored backgrounds
        st.markdown("""
        <style>
        .stdc-card {
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 0.5rem;
        }
        .stdc-see { background: linear-gradient(135deg, #3498db20 0%, #3498db10 100%); border-left: 4px solid #3498db; }
        .stdc-think { background: linear-gradient(135deg, #f39c1220 0%, #f39c1210 100%); border-left: 4px solid #f39c12; }
        .stdc-do { background: linear-gradient(135deg, #27ae6020 0%, #27ae6010 100%); border-left: 4px solid #27ae60; }
        .stdc-care { background: linear-gradient(135deg, #9b59b620 0%, #9b59b610 100%); border-left: 4px solid #9b59b6; }
        </style>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="stdc-card stdc-see">', unsafe_allow_html=True)
            st.markdown("**SEE** - Awareness")
            st.metric("CPM", f"€{see_metrics['cpm']:.2f}", help=format_platform_tooltip(see_metrics['google']['cpm'], see_metrics['meta']['cpm'], 'currency'))
            st.metric("Reach", f"{see_metrics['reach']:,.0f}", help=format_platform_tooltip(see_metrics['google']['reach'], see_metrics['meta']['reach']))
            st.metric("Spend", f"€{see_metrics['spend']:,.0f}", help=format_platform_tooltip(see_metrics['google']['spend'], see_metrics['meta']['spend'], 'currency'))
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="stdc-card stdc-think">', unsafe_allow_html=True)
            st.markdown("**THINK** - Consideration")
            st.metric("CTR", f"{think_metrics['ctr']:.2f}%", help=format_platform_tooltip(think_metrics['google']['ctr'], think_metrics['meta']['ctr'], 'percent'))
            st.metric("Clicks", f"{think_metrics['clicks']:,.0f}", help=format_platform_tooltip(think_metrics['google']['clicks'], think_metrics['meta']['clicks']))
            st.metric("Spend", f"€{think_metrics['spend']:,.0f}", help=format_platform_tooltip(think_metrics['google']['spend'], think_metrics['meta']['spend'], 'currency'))
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="stdc-card stdc-do">', unsafe_allow_html=True)
            st.markdown("**DO** - Conversion")
            st.metric("CPA", f"€{do_metrics['cpa']:.2f}" if do_metrics['cpa'] > 0 else "N/A", help=format_platform_tooltip(do_metrics['google']['cpa'], do_metrics['meta']['cpa'], 'currency'))
            st.metric("Conversions", f"{do_metrics['conversions']:,.0f} ({do_metrics['conv_rate']:.1f}%)", help=format_platform_tooltip(do_metrics['google']['conversions'], do_metrics['meta']['conversions']))
            st.metric("Spend", f"€{do_metrics['spend']:,.0f}", help=format_platform_tooltip(do_metrics['google']['spend'], do_metrics['meta']['spend'], 'currency'))
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="stdc-card stdc-care">', unsafe_allow_html=True)
            st.markdown("**CARE** - Loyalty")
            st.metric("Rebook Rate", f"{rebook_rate:.1f}%" if total_customers > 0 else "N/A", help="From booking data")
            st.metric("CLV", f"€{clv:,.0f}" if clv > 0 else "N/A", help="Customer Lifetime Value from Revenue & Value analysis")
            st.metric("Spend", f"€{care_metrics['spend']:,.0f}", help=format_platform_tooltip(care_metrics['google']['spend'], care_metrics['meta']['spend'], 'currency'))
            st.markdown('</div>', unsafe_allow_html=True)

        # Show untagged campaigns count if any
        untagged_count = len(combined_df[combined_df['stdc_phase'] == 'Untagged'])
        if untagged_count > 0:
            st.caption(f"Note: {untagged_count} campaigns are untagged. Tag them in the Campaign Performance table below.")

        # Platform comparison
        st.markdown("---")
        st.markdown("### Platform Comparison")

        # Better color contrast: Google = Blue, Meta = Teal
        platform_colors = {'Google Ads': '#4285f4', 'Meta Ads': '#00C4B4'}

        # Calculate platform metrics with more detail
        platform_metrics = combined_df.groupby('Platform').agg({
            'spend': 'sum',
            'conversions': 'sum',
            'clicks': 'sum',
            'conversion_value': 'sum'
        }).reset_index()

        # Add reach/impressions
        google_reach = combined_df[combined_df['Platform'] == 'Google Ads']['impressions'].sum() if 'impressions' in combined_df.columns else 0
        meta_reach = combined_df[combined_df['Platform'] == 'Meta Ads']['reach'].sum() if 'reach' in combined_df.columns else 0
        platform_metrics['reach'] = [google_reach, meta_reach] if len(platform_metrics) == 2 else [google_reach + meta_reach]

        # Calculate CPA per platform
        platform_metrics['cpa'] = platform_metrics.apply(
            lambda x: x['spend'] / x['conversions'] if x['conversions'] > 0 else 0, axis=1
        )

        # Row 1: Spend, Conversions, Conv Value
        col1, col2, col3 = st.columns(3)

        with col1:
            fig_plat_spend = px.pie(
                platform_metrics,
                values='spend',
                names='Platform',
                title='Spend',
                color='Platform',
                color_discrete_map=platform_colors
            )
            fig_plat_spend.update_layout(height=350, showlegend=True)
            fig_plat_spend.update_traces(textinfo='percent+value', texttemplate='%{percent:.1%}<br>€%{value:,.0f}')
            st.plotly_chart(fig_plat_spend, use_container_width=True)

        with col2:
            fig_plat_conv = px.pie(
                platform_metrics,
                values='conversions',
                names='Platform',
                title='Conversions',
                color='Platform',
                color_discrete_map=platform_colors
            )
            fig_plat_conv.update_layout(height=350, showlegend=True)
            fig_plat_conv.update_traces(textinfo='percent+value', texttemplate='%{percent:.1%}<br>%{value:,.0f}')
            st.plotly_chart(fig_plat_conv, use_container_width=True)

        with col3:
            fig_plat_value = px.pie(
                platform_metrics,
                values='conversion_value',
                names='Platform',
                title='Conversion Value',
                color='Platform',
                color_discrete_map=platform_colors
            )
            fig_plat_value.update_layout(height=350, showlegend=True)
            fig_plat_value.update_traces(textinfo='percent+value', texttemplate='%{percent:.1%}<br>€%{value:,.0f}')
            st.plotly_chart(fig_plat_value, use_container_width=True)

        # Row 2: Clicks, Reach, CPA comparison
        col1, col2, col3 = st.columns(3)

        with col1:
            fig_plat_clicks = px.pie(
                platform_metrics,
                values='clicks',
                names='Platform',
                title='Clicks',
                color='Platform',
                color_discrete_map=platform_colors
            )
            fig_plat_clicks.update_layout(height=350, showlegend=True)
            fig_plat_clicks.update_traces(textinfo='percent+value', texttemplate='%{percent:.1%}<br>%{value:,.0f}')
            st.plotly_chart(fig_plat_clicks, use_container_width=True)

        with col2:
            fig_plat_reach = px.pie(
                platform_metrics,
                values='reach',
                names='Platform',
                title='Reach / Impressions',
                color='Platform',
                color_discrete_map=platform_colors
            )
            fig_plat_reach.update_layout(height=350, showlegend=True)
            fig_plat_reach.update_traces(textinfo='percent+value', texttemplate='%{percent:.1%}<br>%{value:,.0f}')
            st.plotly_chart(fig_plat_reach, use_container_width=True)

        with col3:
            # CTR comparison as pie chart
            # Calculate CTR per platform
            google_ctr = (google_df['clicks'].sum() / google_df['impressions'].sum() * 100) if 'impressions' in google_df.columns and google_df['impressions'].sum() > 0 else 0
            meta_ctr = (meta_df['clicks'].sum() / meta_df['reach'].sum() * 100) if 'reach' in meta_df.columns and meta_df['reach'].sum() > 0 else 0

            ctr_data = pd.DataFrame({
                'Platform': ['Google Ads', 'Meta Ads'],
                'ctr': [google_ctr, meta_ctr]
            })

            fig_plat_ctr = px.pie(
                ctr_data,
                values='ctr',
                names='Platform',
                title='CTR Distribution',
                color='Platform',
                color_discrete_map=platform_colors
            )
            fig_plat_ctr.update_layout(height=350, showlegend=True)
            fig_plat_ctr.update_traces(textinfo='percent+value', texttemplate='%{percent:.1%}<br>%{value:.2f}%')
            st.plotly_chart(fig_plat_ctr, use_container_width=True)

        # Campaign Performance Table
        st.markdown("---")
        st.markdown("### Campaign Performance")

        # Prepare display table with more metrics
        display_df = combined_df.copy()

        # Filter out campaigns with invalid names (containing "--" or empty)
        display_df = display_df[~display_df['campaign_name'].astype(str).str.contains('^-+$|^ *$', regex=True, na=True)]
        display_df = display_df[display_df['campaign_name'].notna()]
        display_df = display_df[display_df['campaign_name'].astype(str).str.strip() != '']

        # Calculate additional metrics (vectorized)
        display_df['cpc'] = (display_df['spend'] / display_df['clicks'].replace(0, float('nan'))).fillna(0)
        display_df['cpa'] = (display_df['spend'] / display_df['conversions'].replace(0, float('nan'))).fillna(0)
        display_df['conv_rate'] = (display_df['conversions'] / display_df['clicks'].replace(0, float('nan')) * 100).fillna(0)

        # Create combined Reach/Impressions column (impressions for Google, reach for Meta)
        if 'impressions' not in display_df.columns:
            display_df['impressions'] = 0
        if 'reach' not in display_df.columns:
            display_df['reach'] = 0
        # Vectorized: use impressions for Google, reach for Meta
        display_df['reach_impr'] = display_df['impressions'].where(
            display_df['Platform'] == 'Google Ads', display_df['reach']
        )

        # Select and rename columns
        display_cols = ['campaign_name', 'Platform', 'stdc_phase', 'spend', 'reach_impr', 'clicks', 'conversions', 'conversion_value', 'cpc', 'cpa', 'conv_rate']
        display_cols = [c for c in display_cols if c in display_df.columns]

        display_df = display_df[display_cols].copy()
        display_df = display_df.rename(columns={
            'campaign_name': 'Campaign',
            'stdc_phase': 'STDC',
            'spend': 'Spend',
            'reach_impr': 'Reach/Impr',
            'clicks': 'Clicks',
            'conversions': 'Conv',
            'conversion_value': 'Value',
            'cpc': 'CPC',
            'cpa': 'CPA',
            'conv_rate': 'Conv %'
        })

        # Round numeric columns appropriately
        int_cols = ['Spend', 'Reach/Impr', 'Clicks', 'Conv', 'Value']
        for col in int_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(0).astype(int)

        # Keep CPC and CPA with 2 decimal places
        if 'CPC' in display_df.columns:
            display_df['CPC'] = display_df['CPC'].round(2)
        if 'CPA' in display_df.columns:
            display_df['CPA'] = display_df['CPA'].round(2)

        if 'Conv %' in display_df.columns:
            display_df['Conv %'] = display_df['Conv %'].round(0).astype(int)

        # Sort by Platform (Google Ads first) then by Spend descending
        display_df = display_df.sort_values(['Platform', 'Spend'], ascending=[True, False])

        # Color coding for STDC phase
        def style_stdc(val):
            if val == 'SEE':
                return 'background-color: #dbeafe; color: #1e40af'
            elif val == 'THINK':
                return 'background-color: #fef3c7; color: #92400e'
            elif val == 'DO':
                return 'background-color: #dcfce7; color: #166534'
            elif val == 'CARE':
                return 'background-color: #fce7f3; color: #9d174d'
            return 'background-color: #f3f4f6; color: #4b5563'

        styled_df = display_df.style.map(style_stdc, subset=['STDC'])

        campaign_config = {
            'Campaign': st.column_config.TextColumn('Campaign', help='Campaign name from ad platform'),
            'Platform': st.column_config.TextColumn('Platform', help='Google Ads or Meta Ads'),
            'STDC': st.column_config.TextColumn('STDC', help='SEE-THINK-DO-CARE funnel phase'),
            'Spend': st.column_config.NumberColumn('Spend', help='Total ad spend for this campaign', format='€%d'),
            'Reach/Impr': st.column_config.NumberColumn('Reach/Impr', help='Impressions (Google) or Reach (Meta)'),
            'Clicks': st.column_config.NumberColumn('Clicks', help='Link clicks on ads'),
            'Conv': st.column_config.NumberColumn('Conv', help='Purchases/conversions tracked'),
            'Value': st.column_config.NumberColumn('Value', help='Conversion value reported', format='€%d'),
            'CPC': st.column_config.NumberColumn('CPC', help='Cost per Click = Spend / Clicks', format='€%.2f'),
            'CPA': st.column_config.NumberColumn('CPA', help='Cost per Acquisition = Spend / Conversions', format='€%.2f'),
            'Conv %': st.column_config.NumberColumn('Conv %', help='Conversion Rate = Conversions / Clicks'),
        }
        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=400, column_config=campaign_config)

        # STDC Tag Configuration
        with st.expander("Configure STDC Tags", expanded=False):
            st.markdown("""
            Assign each campaign to a SEE-THINK-DO-CARE phase.
            Default suggestions are based on campaign name keywords.
            """)

            campaigns = sorted(combined_df['campaign_name'].unique().tolist())

            for campaign in campaigns:
                current_tag = st.session_state.stdc_tags.get(campaign, 'Untagged')

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(campaign[:60] + '...' if len(campaign) > 60 else campaign)
                with col2:
                    new_tag = st.selectbox(
                        "Phase",
                        options=['SEE', 'THINK', 'DO', 'CARE', 'Untagged'],
                        index=['SEE', 'THINK', 'DO', 'CARE', 'Untagged'].index(current_tag),
                        key=f"stdc_{campaign}",
                        label_visibility="collapsed"
                    )
                    st.session_state.stdc_tags[campaign] = new_tag

            if st.button("Reset All to Suggestions", key="reset_stdc"):
                for campaign in campaigns:
                    st.session_state.stdc_tags[campaign] = suggest_stdc_phase(campaign)
                st.rerun()

        # ===== Marketing ROI by Location Section =====
        st.markdown("---")
        st.markdown("### Marketing ROI by Location")

        # Platform filter with toggles
        platform_options = []
        if 'Platform' in combined_df.columns:
            platform_options = combined_df['Platform'].unique().tolist()

        selected_platforms = []
        if len(platform_options) > 1:
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                google_on = st.toggle("Google Ads", value=True, key="roi_google_toggle")
                if google_on and 'Google Ads' in platform_options:
                    selected_platforms.append('Google Ads')
            with col2:
                meta_on = st.toggle("Meta Ads", value=False, key="roi_meta_toggle")
                if meta_on and 'Meta Ads' in platform_options:
                    selected_platforms.append('Meta Ads')
            # Filter combined_df for this section
            roi_filtered_df = combined_df[combined_df['Platform'].isin(selected_platforms)]
        else:
            roi_filtered_df = combined_df
            selected_platforms = platform_options

        # Check if we have booking data
        if st.session_state.df1 is None:
            st.info("Upload booking data to see revenue by location alongside marketing metrics.")
        elif len(selected_platforms) == 0:
            st.warning("Please select at least one platform.")
        else:
            # Get marketing metrics by location (with caching)
            roi_df_hash = hash(roi_filtered_df.to_json())
            roi_df_json = roi_filtered_df.to_json()
            marketing_by_loc, excluded_spend = get_marketing_by_location(roi_df_hash, roi_df_json)

            if marketing_by_loc is None or len(marketing_by_loc) == 0:
                st.warning("No campaigns could be attributed to specific locations. Campaigns must mention a single location in their name.")
            else:
                # Get date range from marketing data
                if 'report_start' in combined_df.columns and combined_df['report_start'].notna().any():
                    min_date = combined_df['report_start'].min()
                    max_date = combined_df['report_end'].max() if 'report_end' in combined_df.columns else min_date
                else:
                    # Fallback: use booking data date range
                    date_col = 'Created' if 'Created' in st.session_state.df1.columns else 'Date'
                    if date_col in st.session_state.df1.columns:
                        dates = pd.to_datetime(st.session_state.df1[date_col], errors='coerce')
                        min_date = dates.min()
                        max_date = dates.max()
                    else:
                        min_date = None
                        max_date = None

                # Get revenue by location for the same date range
                if min_date is not None and max_date is not None:
                    # Detect columns
                    location_col = 'Location' if 'Location' in st.session_state.df1.columns else 'Tour' if 'Tour' in st.session_state.df1.columns else 'Activity'
                    date_col = 'Created' if 'Created' in st.session_state.df1.columns else 'Date'
                    revenue_col = 'Total gross' if 'Total gross' in st.session_state.df1.columns else None

                    if revenue_col:
                        revenue_by_loc = get_revenue_by_location_filtered(
                            st.session_state.df1, min_date, max_date,
                            location_col=location_col, date_col=date_col, revenue_col=revenue_col
                        )

                        # Create combined table
                        roi_table = create_marketing_roi_table(marketing_by_loc, revenue_by_loc)

                        if roi_table is not None and len(roi_table) > 0:
                            # Show excluded spend warning
                            total_spend = roi_filtered_df['spend'].sum()
                            attributed_spend = roi_table['Ad Spend'].sum()
                            excluded_pct = (excluded_spend / total_spend * 100) if total_spend > 0 else 0

                            st.info(f"⚠️ **€{excluded_spend:,.0f}** ({excluded_pct:.1f}%) of ad spend is excluded from this table. "
                                    f"This includes 'alle locaties' campaigns, multi-location campaigns, and campaigns without a clear location target.")

                            # Show date range
                            st.caption(f"Data period: {pd.to_datetime(min_date).strftime('%d %b %Y')} - {pd.to_datetime(max_date).strftime('%d %b %Y')}")

                            # Key metrics
                            total_bookings = roi_table['Bookings'].sum()
                            total_clicks = roi_table['Clicks'].sum()
                            total_conversions = roi_table['Conversions'].sum()
                            total_conv_value = roi_table['Conv. Value'].sum()
                            overall_roas = (total_conv_value / attributed_spend) if attributed_spend > 0 else 0
                            overall_cpa = attributed_spend / total_conversions if total_conversions > 0 else 0
                            overall_conv_rate = (total_conversions / total_clicks * 100) if total_clicks > 0 else 0

                            col1, col2, col3, col4, col5, col6 = st.columns(6)
                            with col1:
                                st.metric("Ad Spend", f"€{attributed_spend:,.0f}", help="Total ad spend from campaigns targeting specific locations")
                            with col2:
                                st.metric("Conv. Value", f"€{total_conv_value:,.0f}", help="Total conversion value reported by ad platforms")
                            with col3:
                                st.metric("Bookings", f"{total_bookings:,}", help="Total bookings at these locations during the period")
                            with col4:
                                st.metric("Conv. Rate", f"{overall_conv_rate:.1f}%", help="Conversions ÷ Clicks × 100 (% of clicks that converted)")
                            with col5:
                                st.metric("Avg CPA", f"€{overall_cpa:,.2f}", help="Cost per Acquisition = Ad Spend ÷ Conversions")
                            with col6:
                                st.metric("ROAS", f"{overall_roas:.1f}x", help="Return on Ad Spend = Conv. Value ÷ Ad Spend")

                            # Per-location profit margins
                            locations = roi_table['Location'].tolist()

                            with st.expander("Profit Margins per Location", expanded=False):
                                st.caption("Set profit margin for each location to calculate ROAS thresholds. Default: 70%")
                                margin_cols = st.columns(min(len(locations), 4))
                                location_margins = {}
                                for i, loc in enumerate(locations):
                                    col_idx = i % 4
                                    with margin_cols[col_idx]:
                                        # Shorten location name for display
                                        short_name = loc.replace('Amsterdam ', '').replace('Rotterdam ', '')
                                        location_margins[loc] = st.number_input(
                                            short_name,
                                            min_value=10,
                                            max_value=95,
                                            value=70,
                                            step=5,
                                            key=f"margin_{loc}",
                                            help=f"Profit margin % for {loc}"
                                        )

                            # Format table for display
                            display_df = roi_table.copy()
                            display_df['Revenue'] = display_df['Revenue'].round(0).astype(int)
                            display_df['Bookings'] = display_df['Bookings'].round(0).astype(int)
                            display_df['Clicks'] = display_df['Clicks'].round(0).astype(int)
                            display_df['Conversions'] = display_df['Conversions'].round(0).astype(int)
                            display_df['Conv. Rate %'] = (display_df['Conv. Rate %'] * 10).round() / 10  # Round to 1 decimal
                            display_df['Conv. Value'] = display_df['Conv. Value'].round(0).astype(int)
                            display_df['Ad Spend'] = display_df['Ad Spend'].round(0).astype(int)
                            display_df['CPA'] = (display_df['CPA'] * 100).round() / 100  # Round to 2 decimals
                            display_df['ROAS'] = (display_df['ROAS'] * 10).round() / 10  # Round to 1 decimal

                            # Calculate ROAS thresholds per location
                            # Break-even ROAS = 1 / (margin/100) = 100 / margin
                            def get_roas_thresholds(margin_pct):
                                margin_decimal = margin_pct / 100
                                breakeven = 1 / margin_decimal
                                return breakeven * 1.5, breakeven * 2  # good, excellent

                            # Color code ROAS based on per-location margin thresholds
                            def style_roas_row(row):
                                loc = row['Location']
                                margin = location_margins.get(loc, 70)
                                good_threshold, excellent_threshold = get_roas_thresholds(margin)
                                roas_val = row['ROAS']

                                styles = [''] * len(row)
                                roas_idx = row.index.get_loc('ROAS')

                                if roas_val >= excellent_threshold:
                                    styles[roas_idx] = 'background-color: #dcfce7; color: #166534'  # Green
                                elif roas_val >= good_threshold:
                                    styles[roas_idx] = 'background-color: #fef3c7; color: #92400e'  # Yellow
                                else:
                                    styles[roas_idx] = 'background-color: #fecaca; color: #991b1b'  # Red
                                return styles

                            styled_roi = display_df.style.apply(style_roas_row, axis=1)

                            # Show threshold legend
                            st.caption("ROAS colors: 🔴 < 1.5× break-even | 🟡 1.5× - 2× break-even | 🟢 > 2× break-even (thresholds vary by location margin)")

                            roi_config = {
                                'Location': st.column_config.TextColumn('Location'),
                                'Revenue': st.column_config.NumberColumn('Revenue', format='€%d', help='Total booking revenue for this location'),
                                'Bookings': st.column_config.NumberColumn('Bookings', help='Number of bookings'),
                                'Clicks': st.column_config.NumberColumn('Clicks', help='Total ad clicks'),
                                'Conversions': st.column_config.NumberColumn('Conversions', help='Ad conversions reported'),
                                'Conv. Rate %': st.column_config.NumberColumn('Conv. Rate %', format='%.1f', help='Conversions / Clicks × 100 (% of clicks that converted)'),
                                'Conv. Value': st.column_config.NumberColumn('Conv. Value', format='€%d', help='Conversion value from ads'),
                                'Ad Spend': st.column_config.NumberColumn('Ad Spend', format='€%d', help='Total ad spend for this location'),
                                'CPA': st.column_config.NumberColumn('CPA', format='€%.2f', help='Cost per Acquisition = Ad Spend / Conversions'),
                                'ROAS': st.column_config.NumberColumn('ROAS', format='%.1f', help='Conv. Value / Ad Spend (e.g., 5x = €5 conv value per €1 spent)'),
                            }

                            st.dataframe(styled_roi, use_container_width=True, hide_index=True, column_config=roi_config)

                            # Explanation
                            with st.expander("How this table works", expanded=False):
                                st.markdown(f"""
                                **Data Sources:**
                                - **Revenue & Bookings**: From your booking data, filtered to the marketing data date range
                                - **Conversions, Conv. Value, Ad Spend**: From campaigns that target a single specific location

                                **What's Excluded:**
                                - "Alle locaties" / "All locations" campaigns
                                - Campaigns targeting multiple locations (e.g., "Amsterdam M, IJ, N & Sloterplas")
                                - Campaigns without a recognizable location in the name

                                **Metrics:**
                                - **Conv. Rate %** = Conversions ÷ Clicks × 100 (what % of ad clicks converted)
                                - **CPA** (Cost per Acquisition) = Ad Spend ÷ Conversions
                                - **ROAS** (Return on Ad Spend) = Conv. Value ÷ Ad Spend (e.g., 5x = €5 conversion value per €1 spent)

                                **ROAS Color Thresholds (per-location, based on profit margin):**
                                - Break-even ROAS = 100 ÷ margin% (e.g., 70% margin → 1.43x break-even)
                                - 🔴 Red: < 1.5× break-even (losing money or marginal)
                                - 🟡 Yellow: 1.5× - 2× break-even (good return)
                                - 🟢 Green: > 2× break-even (excellent return)

                                Set margins per location in the "Profit Margins per Location" expander above.

                                **Note:** Revenue includes ALL bookings at each location during the period, not just those from ads.
                                """)
                        else:
                            st.warning("Could not create ROI table. Check that booking data contains revenue information.")
                    else:
                        st.warning("Revenue column ('Total gross') not found in booking data.")
                else:
                    st.warning("Could not determine date range from marketing data.")

        # ===== CPA Targets Section =====
        st.markdown("---")
        st.markdown("### CPA Targets")
        st.caption("Calculate break-even and target CPA based on your booking data and cost structure.")

        if st.session_state.df1 is None:
            st.info("Upload booking data to calculate CPA targets.")
        else:
            # Detect columns
            date_col = 'Created' if 'Created' in st.session_state.df1.columns else 'Date'
            revenue_col = 'Total gross' if 'Total gross' in st.session_state.df1.columns else None
            email_col = 'Email address' if 'Email address' in st.session_state.df1.columns else None

            if revenue_col is None:
                st.warning("Revenue column ('Total gross') not found in booking data.")
            else:
                # Calculate metrics from booking data
                cpa_metrics = calculate_cpa_metrics(
                    st.session_state.df1,
                    date_col=date_col,
                    revenue_col=revenue_col,
                    email_col=email_col if email_col else 'Email address'
                )

                if cpa_metrics is None:
                    st.warning("Could not calculate CPA metrics from booking data.")
                else:
                    # Platform toggles and user inputs
                    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                    with col1:
                        cpa_google_on = st.toggle("Google Ads", value=True, key="cpa_google_toggle")
                    with col2:
                        cpa_meta_on = st.toggle("Meta Ads", value=True, key="cpa_meta_toggle")
                    with col3:
                        bedrijfskosten = st.number_input(
                            "Bedrijfskosten %",
                            min_value=0,
                            max_value=100,
                            value=70,
                            step=5,
                            key="cpa_bedrijfskosten",
                            help="Operating costs as percentage of revenue (e.g., 70% means €0.70 of each €1 goes to costs)"
                        )
                    with col4:
                        winstmarge = st.number_input(
                            "Winstmarge %",
                            min_value=0,
                            max_value=100,
                            value=30,
                            step=5,
                            key="cpa_winstmarge",
                            help="Target profit margin after all costs (e.g., 30% means you want €0.30 profit per €1 revenue after costs)"
                        )

                    # Build platform filter
                    cpa_selected_platforms = []
                    if cpa_google_on:
                        cpa_selected_platforms.append('Google Ads')
                    if cpa_meta_on:
                        cpa_selected_platforms.append('Meta Ads')

                    if len(cpa_selected_platforms) == 0:
                        st.warning("Please select at least one platform to calculate Actual CPA.")

                    # Calculate CPA targets
                    cpa_targets = calculate_cpa_targets(
                        cpa_metrics['aov'],
                        bedrijfskosten,
                        winstmarge,
                        cpa_metrics['retention_rate']
                    )

                    # Get actual CPA from marketing data if available
                    # Exclude SEE and THINK campaigns (awareness/consideration - not measured on CPA)
                    actual_cpa = None
                    excluded_phases = ['SEE', 'THINK']
                    if 'combined_df' in dir() and combined_df is not None and len(combined_df) > 0:
                        # Filter by selected platforms
                        cpa_df = combined_df.copy()
                        if 'Platform' in cpa_df.columns and len(cpa_selected_platforms) > 0:
                            cpa_df = cpa_df[cpa_df['Platform'].isin(cpa_selected_platforms)]
                        # Filter to DO and CARE campaigns only
                        if 'stdc_phase' in cpa_df.columns:
                            cpa_df = cpa_df[~cpa_df['stdc_phase'].isin(excluded_phases)]
                        total_spend = cpa_df['spend'].sum()
                        total_conversions = cpa_df['conversions'].sum()
                        if total_conversions > 0:
                            actual_cpa = total_spend / total_conversions

                    # Build metrics table
                    st.markdown("#### Per-Booking CPA")

                    # Per-booking metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(
                            "AOV",
                            f"€{cpa_metrics['aov']:.2f}",
                            help="Average Order Value = Total Revenue ÷ Total Bookings"
                        )
                    with col2:
                        st.metric(
                            "Break-even CPA",
                            f"€{cpa_targets['breakeven_cpa']:.2f}",
                            help=f"Maximum CPA to avoid loss = AOV × (1 - Bedrijfskosten%) = €{cpa_metrics['aov']:.2f} × {(100-bedrijfskosten)/100:.2f}"
                        )
                    with col3:
                        st.metric(
                            "Target CPA",
                            f"€{cpa_targets['target_cpa']:.2f}",
                            help=f"CPA to achieve target profit = Break-even × (1 - Winstmarge%) = €{cpa_targets['breakeven_cpa']:.2f} × {(100-winstmarge)/100:.2f}"
                        )
                    with col4:
                        if actual_cpa is not None:
                            delta = actual_cpa - cpa_targets['target_cpa']
                            delta_color = "inverse" if delta > 0 else "normal"
                            st.metric(
                                "Actual CPA",
                                f"€{actual_cpa:.2f}",
                                delta=f"€{delta:+.2f} vs target",
                                delta_color=delta_color,
                                help="Current CPA from DO & CARE campaigns only (excludes SEE & THINK awareness campaigns)"
                            )
                        else:
                            st.metric(
                                "Actual CPA",
                                "N/A",
                                help="Upload marketing data to see actual CPA"
                            )

                    # LTV-based CPA (only if sufficient data)
                    if cpa_metrics['has_sufficient_data'] and cpa_metrics['retention_rate'] is not None and 'ltv' in cpa_targets:
                        st.markdown("#### LTV-Based CPA")
                        st.caption("Based on customer lifetime value - allows higher CPA for long-term profitability.")

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(
                                "Retention Rate",
                                f"{cpa_metrics['retention_rate']:.1%}",
                                help=f"Percentage of customers who return within 2 months. Based on cohort of {cpa_metrics['cohort_size']} customers."
                            )
                        with col2:
                            st.metric(
                                "Expected Bookings",
                                f"{cpa_targets['expected_bookings']:.1f}",
                                help=f"Expected bookings per customer = 1 ÷ (1 - Retention Rate) = 1 ÷ {1 - cpa_metrics['retention_rate']:.2f}"
                            )
                        with col3:
                            st.metric(
                                "LTV",
                                f"€{cpa_targets['ltv']:.2f}",
                                help=f"Customer Lifetime Value = AOV × Expected Bookings = €{cpa_metrics['aov']:.2f} × {cpa_targets['expected_bookings']:.1f}"
                            )
                        with col4:
                            st.metric(
                                "Break-even CPA (LTV)",
                                f"€{cpa_targets['breakeven_cpa_ltv']:.2f}",
                                help=f"Maximum CPA based on LTV = LTV × (1 - Bedrijfskosten%) = €{cpa_targets['ltv']:.2f} × {(100-bedrijfskosten)/100:.2f}"
                            )

                        # Target CPA (LTV) with comparison
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(
                                "Target CPA (LTV)",
                                f"€{cpa_targets['target_cpa_ltv']:.2f}",
                                help=f"Target CPA based on LTV = Break-even (LTV) × (1 - Winstmarge%) = €{cpa_targets['breakeven_cpa_ltv']:.2f} × {(100-winstmarge)/100:.2f}"
                            )
                        with col2:
                            if actual_cpa is not None:
                                delta_ltv = actual_cpa - cpa_targets['target_cpa_ltv']
                                delta_color_ltv = "inverse" if delta_ltv > 0 else "normal"
                                st.metric(
                                    "Actual vs Target (LTV)",
                                    f"€{actual_cpa:.2f}",
                                    delta=f"€{delta_ltv:+.2f}",
                                    delta_color=delta_color_ltv,
                                    help="Comparison of actual CPA to LTV-based target"
                                )
                    else:
                        # Show why LTV metrics are hidden
                        if not cpa_metrics['has_sufficient_data']:
                            st.info(f"📊 LTV-based CPA requires at least 2 months of data. Current data span: {cpa_metrics['data_span_months']:.1f} months.")
                        elif cpa_metrics['retention_rate'] is None:
                            st.info("📊 LTV-based CPA requires customer email data to calculate retention rate.")

                    # Detailed table
                    with st.expander("View Calculation Details", expanded=False):
                        table_data = [
                            {"Metric": "AOV (Average Order Value)", "Value": f"€{cpa_metrics['aov']:.2f}", "Source": "Booking data", "Formula": "Total Revenue ÷ Total Bookings"},
                            {"Metric": "Bedrijfskosten %", "Value": f"{bedrijfskosten}%", "Source": "User input", "Formula": "-"},
                            {"Metric": "Break-even CPA", "Value": f"€{cpa_targets['breakeven_cpa']:.2f}", "Source": "Calculated", "Formula": f"AOV × (1 - {bedrijfskosten}%) = €{cpa_metrics['aov']:.2f} × {(100-bedrijfskosten)/100:.2f}"},
                            {"Metric": "Winstmarge %", "Value": f"{winstmarge}%", "Source": "User input", "Formula": "-"},
                            {"Metric": "Target CPA", "Value": f"€{cpa_targets['target_cpa']:.2f}", "Source": "Calculated", "Formula": f"Break-even × (1 - {winstmarge}%) = €{cpa_targets['breakeven_cpa']:.2f} × {(100-winstmarge)/100:.2f}"},
                        ]

                        if cpa_metrics['retention_rate'] is not None and 'ltv' in cpa_targets:
                            table_data.extend([
                                {"Metric": "Retention Rate", "Value": f"{cpa_metrics['retention_rate']:.1%}", "Source": "Booking data", "Formula": "Returning customers ÷ Cohort size"},
                                {"Metric": "Expected Bookings", "Value": f"{cpa_targets['expected_bookings']:.1f}", "Source": "Calculated", "Formula": f"1 ÷ (1 - {cpa_metrics['retention_rate']:.1%})"},
                                {"Metric": "LTV", "Value": f"€{cpa_targets['ltv']:.2f}", "Source": "Calculated", "Formula": f"AOV × Expected Bookings"},
                                {"Metric": "Break-even CPA (LTV)", "Value": f"€{cpa_targets['breakeven_cpa_ltv']:.2f}", "Source": "Calculated", "Formula": f"LTV × (1 - {bedrijfskosten}%)"},
                                {"Metric": "Target CPA (LTV)", "Value": f"€{cpa_targets['target_cpa_ltv']:.2f}", "Source": "Calculated", "Formula": f"Break-even (LTV) × (1 - {winstmarge}%)"},
                            ])

                        if actual_cpa is not None:
                            table_data.append({"Metric": "Actual CPA (DO & CARE only)", "Value": f"€{actual_cpa:.2f}", "Source": "Marketing data", "Formula": "Ad Spend ÷ Conversions (excl. SEE & THINK)"})

                        st.dataframe(
                            pd.DataFrame(table_data),
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                'Metric': st.column_config.TextColumn('Metric', width='medium'),
                                'Value': st.column_config.TextColumn('Value', width='small'),
                                'Source': st.column_config.TextColumn('Source', width='small'),
                                'Formula': st.column_config.TextColumn('Formula', width='large'),
                            }
                        )

                        st.markdown(f"""
                        **Data Summary:**
                        - Total Bookings: {cpa_metrics['total_bookings']:,}
                        - Total Revenue: €{cpa_metrics['total_revenue']:,.0f}
                        - Data Span: {cpa_metrics['data_span_months']:.1f} months
                        """)

                        if cpa_metrics['retention_rate'] is not None:
                            st.markdown(f"""
                            **Retention Analysis:**
                            - Cohort Size: {cpa_metrics['cohort_size']} first-time customers
                            - Returning Customers: {cpa_metrics['returning_customers']} (within 2 months)
                            - Retention Rate: {cpa_metrics['retention_rate']:.1%}
                            """)

    # Reset button
    st.sidebar.markdown("---")
    if st.sidebar.button("Clear All & Start Over", key="mkt_reset"):
        st.session_state.google_ads_df = None
        st.session_state.meta_ads_df = None
        st.session_state.stdc_tags = {}
        st.rerun()
