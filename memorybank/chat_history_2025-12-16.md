# Chat History - December 16, 2025

## Project Overview
**Kuuma Sauna Booking Analyzer** - A Streamlit-based analytics dashboard for Kuuma Sauna locations in the Netherlands. The app analyzes booking data, marketing performance, and capacity utilization across multiple sauna locations.

## Key Files
- `app.py` - Main application with login, data loading from Google Drive
- `pages/1_Overview.py` - Overview dashboard
- `pages/3_Customers.py` - Customer analysis
- `pages/4_Revenue.py` - Revenue analysis
- `pages/5_Promotions.py` - Promotions analysis
- `pages/6_Capacity.py` - Capacity/occupancy analysis
- `pages/7_Marketing.py` - Marketing performance analysis
- `data_loader.py` - Centralized session state initialization

## Locations (14 total)
1. Marineterrein Matsu (Flagship)
2. Marineterrein Bjørk (Flagship)
3. Scheveningen (Flagship)
4. Amsterdam Noord (Groeier)
5. Amsterdam Sloterplas (Groeier)
6. Amsterdam IJ / Aan 't IJ Centrum (Groeier)
7. Nijmegen Lent (Groeier)
8. Rotterdam Rijnhaven (Groeier)
9. Rotterdam Delfshaven (Onderbenut)
10. Katwijk (Onderbenut)
11. Wijk aan Zee (Onderbenut)
12. Bloemendaal
13. Den Bosch
14. Nijmegen NYMA

## Changes Made This Session

### 1. Marketing Page - Removed "Performance by Location" Section
**Commit:** `05cc90b`
- Removed the entire "Performance by Location" section with 3 tabs (Marketing, Revenue & Efficiency, Capacity)
- This was removed because location attribution from campaign names was unreliable
- The Marketing page now shows: Overall metrics, STDC analysis, Platform comparison, Campaign Performance table

### 2. Capacity Page - Data-Driven Location Display
**Commit:** `05cc90b`
- Changed from filtering against predefined `LOCATION_CAPACITY` list to showing all locations from uploaded data
- Added flexible location matching functions:
  - `find_capacity_match(location_name)` - Matches location names with case-insensitive and partial matching
  - `get_capacity_for_location(location_name)` - Gets capacity data with flexible matching
- Example: "Wijk aan Zee" now matches "Kuuma Wijk aan Zee" in capacity config

### 3. Loading Progress Bar Improvements
**Commit:** `70c8302`
- Progress bar now shows which file is being loaded (e.g., "Loading: Bookings_Q4_2024.xlsx...")
- Progress updates continuously during loading (never stands still)
- File names truncated to 40 chars if too long
- Progress flow: 5% (connecting) → 10% (scanning) → 15-85% (loading files) → 90% (finalizing) → 100% (complete)
- Modified `load_files_from_drive()` to accept `progress_callback` parameter

### 4. Filter Test Data from Capacity Page
**Commit:** `eb91b7a`
- Added filter to only show locations starting with "Kuuma"
- Excludes test data like "UTM test" from the location dropdown and calculations

### 5. Revenue Page Performance Optimization
**Commit:** `1e10dd1`
- Replaced slow O(n²) loops and lambda functions with vectorized pandas operations
- Optimizations made:
  - Retention rate calculation: Use `rank()` instead of per-customer loop
  - Customer metrics: Vectorized interval calculation instead of lambda in groupby
  - Segment CLV: Pre-calculated booking ranks and set operations
  - Location CLV: Pre-aggregated stats and vectorized retention

### 6. Data Quality Explanation
**Commit:** `7212f01`
- Added "(likely cancelled bookings)" explanation to unmatched records in Data Quality summary
- Message now shows: "X unmatched (likely cancelled bookings)"

### 7. Hide "Press Enter to apply" Tooltip
**Commit:** `5f26010`
- Added CSS to hide the InputInstructions tooltip that overlapped with password field eye icon
- Added `[data-testid="InputInstructions"] { display: none; }` to global styles

### 8. Local Data Loading for Development
- Added `load_local_files()` and `has_local_data()` functions to app.py
- App now automatically loads data from local folders when running locally (before trying Google Drive)
- **Folder structure:**
  ```
  ../booking data/
  ├── The day the booking was made*.xls  → Loaded as df1 (booking creation dates)
  ├── The date booked*.xls               → Loaded as df2 (visit dates)
  └── Bezettings analyse.xlsx            → (capacity reference)

  ../marketing data/
  ├── *google*.csv                       → Loaded as google_ads_df
  └── *meta*.csv                         → Loaded as meta_ads_df
  ```
- File detection based on filename patterns (case-insensitive)
- Priority: Local data → Google Drive → Manual upload

## Technical Details

### Location Capacity Configuration (LOCATION_CAPACITY)
```python
LOCATION_CAPACITY = {
    'Kuuma Marineterrein Matsu': {'dal': 96, 'piek': 366, 'weekday': 462, 'weekend': 198, 'cluster': 'Flagship'},
    'Kuuma Marineterrein Bjørk': {'dal': 128, 'piek': 488, 'weekday': 616, 'weekend': 264, 'cluster': 'Flagship'},
    'Kuuma Scheveningen': {'dal': 96, 'piek': 520, 'weekday': 616, 'weekend': 264, 'cluster': 'Flagship'},
    'Kuuma Noord': {'dal': 96, 'piek': 336, 'weekday': 432, 'weekend': 192, 'cluster': 'Groeier'},
    # ... etc
}
```

### Time Period Categories (Capacity)
- **Dal uren**: Monday-Thursday, 10:00-16:00 (off-peak)
- **Piek uren**: Monday-Thursday outside 10:00-16:00 (peak)
- **Weekend**: Friday, Saturday, Sunday (all hours)
- **Ma-Do**: Combined weekday (dal + piek)

### Cluster Targets
- **Flagship**: 75-80% total, 90-95% weekend, 80-85% peak, 55-60% off-peak
- **Groeier**: 60-65% total, 80-85% weekend, 70-75% peak, 30-40% off-peak
- **Onderbenut**: 50-55% total, 70-75% weekend, 60-65% peak, 25-30% off-peak

## Data Sources
- **Booking data (df1)**: Excel files containing booking creation data
- **Visit data (df2)**: Excel files with visit dates (used for capacity analysis)
- **Google Ads**: CSV files with campaign performance
- **Meta Ads**: CSV files with campaign performance

**Data Loading Priority:**
1. Local folders (`../booking data/` and `../marketing data/`) - for development
2. Google Drive (configured via Streamlit secrets) - for production
3. Manual file upload via sidebar

## Known Issues / Notes
- Some locations may not have capacity data configured (Bloemendaal, Den Bosch, Nijmegen NYMA)
- Marketing location attribution was problematic - campaigns targeting multiple locations couldn't be reliably split
- Location names in booking data must start with "Kuuma" to appear in Capacity page

## GitHub Repository
https://github.com/raouldevries/Sauna-Booking-Interval-Analyzer.git
