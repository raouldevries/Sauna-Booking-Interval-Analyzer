# Kuuma Booking Analyzer - Project Guidelines

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`

## Project Structure

```
sauna-booking-analyzer/
├── app.py              # Main Streamlit application (single-page, all tabs)
├── data_loader.py      # Centralized data loading and caching
├── utils.py            # Shared utilities (navigation, constants, charts)
├── requirements.txt    # Python dependencies
├── venv/               # Virtual environment (not in git)
└── assets/             # Logo and static files
```

## Architecture

**Single-page app** with tab-based navigation:
- All functionality lives in `app.py`
- `data_loader.py` handles data loading, merging, and caching via `@st.cache_data`
- `utils.py` provides shared constants (LOCATIONS, PAGES) and helper functions

## Data Flow

1. **Data Sources**:
   - Google Drive (production) - reads from shared folder
   - Local file upload (development) - sidebar uploader

2. **Data Files**:
   - `df1`: Booking creation dates (when customer booked)
   - `df2`: Visit dates (when customer visited)
   - `google_ads_df`: Google Ads CSV data
   - `meta_ads_df`: Meta Ads CSV data

3. **Session State**: All data stored in `st.session_state` for cross-tab persistence

## UI/UX Guidelines

### Icons vs Emoticons
- **Always use Material icons** (`:material/icon_name:`) instead of emoticons/emojis
- Example: Use `:material/lightbulb:` instead of the lightbulb emoji
- This ensures a consistent, professional look across the app

### Common Material Icons Used
- `:material/home:` - Home/Overview
- `:material/calendar_month:` - Booking Patterns
- `:material/group:` - Customers
- `:material/payments:` - Revenue
- `:material/local_offer:` - Promotions
- `:material/analytics:` - Capacity Analysis
- `:material/campaign:` - Marketing
- `:material/science:` - Chart Test
- `:material/lightbulb:` - Insights/Tips
- `:material/login:` - Login

## Column Name Patterns

Location column may appear as: `'Location'`, `'Tour'`, or `'Activity'`

Use `get_location_column(df)` from `data_loader.py` to get the correct column name.

## Deployment

- Deployed on **Streamlit Cloud**
- Local file storage does not persist (ephemeral)
- For persistent data, use external services (Google Sheets, etc.)
- Google Drive API configured via `st.secrets["gcp_service_account"]`

## Key Functions

| Function | File | Purpose |
|----------|------|---------|
| `init_session_state()` | data_loader.py | Initialize all session state vars |
| `load_files_from_drive()` | app.py | Load data from Google Drive |
| `process_booking_data()` | data_loader.py | Merge and calculate intervals |
| `get_location_column()` | data_loader.py | Get location column name |
| `calculate_distribution_data()` | data_loader.py | Lead time distribution |

## Business Context

This app analyzes booking data for **Kuuma** - a Nordic community sauna business:
- 75-minute sessions
- Pricing: Non-member, member discount (~5 euro korting)
- Seasonal business: October - May
- Multiple locations in Amsterdam, Rotterdam, Nijmegen, and beach locations
