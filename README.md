# Sauna Booking Interval Analyzer

A Streamlit web application that analyzes the time gap between when sauna bookings are made and when customers actually visit. Helps understand booking behavior patterns for better capacity planning and staffing.

## Features

- **Upload Excel Files**: Support for .xls and .xlsx formats
- **Smart Column Mapping**: Auto-detects common column names with fallback to manual selection
- **Key Metrics**: Average lead time, median, total bookings, same-day booking percentage
- **Visualizations**:
  - Distribution chart showing booking intervals
  - Trend chart showing average lead time over time
  - Location comparison table
- **Filters**: Date range and location filtering
- **Export**: Download results as CSV
- **Data Quality**: Tracking of matched, unmatched, and invalid records

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Setup

1. **Navigate to the project folder**:
   ```bash
   cd "/Users/raouldevries/Work/Kuuma/Data app/sauna-booking-analyzer"
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the App

Start the Streamlit app:

```bash
streamlit run app.py
```

The app will automatically open in your default browser at `http://localhost:8501`

### Using the App

1. **Upload Files**:
   - Upload your booking creation dates file (File 1)
   - Upload your visit dates file (File 2)

2. **Map Columns**:
   - The app will auto-detect common column names
   - Verify or adjust the column mappings:
     - Booking ID (unique identifier in both files)
     - Booking Creation Date (when customer made the booking)
     - Visit Date (when customer visited)
     - Location (optional, for multi-location comparison)

3. **View Results**:
   - Key metrics displayed at the top
   - Distribution chart shows booking patterns
   - Trend chart shows changes over time
   - Location breakdown table (if location column selected)

4. **Apply Filters** (optional):
   - Adjust date range
   - Select specific locations

5. **Export Results**:
   - Click "Download Results (CSV)" in the sidebar

## Data Format

### Expected File Structure

Both Excel files should contain:

- **Booking ID**: A unique identifier that links records across both files
- **Dates**: In any common format (DD/MM/YYYY, YYYY-MM-DD, Excel date format)
- **Location** (optional): Branch or location name

### Example Data Structure

**File 1 (Booking Creation)**:
| Booking number | Created | Activity |
|----------------|---------|----------|
| 123456 | 2025-09-15 | Kuuma Noord |
| 123457 | 2025-09-16 | Kuuma Sloterplas |

**File 2 (Visit Dates)**:
| Booking number | Start | Activity |
|----------------|-------|----------|
| 123456 | 2025-10-01 | Kuuma Noord |
| 123457 | 2025-10-05 | Kuuma Sloterplas |

## Understanding the Results

### Metrics

- **Average Lead Time**: Mean number of days between booking and visit
- **Median Lead Time**: Middle value (less affected by outliers)
- **Total Bookings**: Number of analyzed bookings after filtering
- **Same-Day Bookings**: Percentage of bookings made on the day of visit

### Interval Categories

- **Same day**: 0 days lead time
- **1-3 days**: Short-term bookings
- **4-7 days**: One week advance
- **1-2 weeks**: Two weeks advance
- **2+ weeks**: Long-term planning

## Troubleshooting

### No matching booking IDs found

- Verify that both files contain the same Booking ID column
- Check that IDs are formatted the same way in both files
- Ensure you've selected the correct ID columns

### Invalid dates warning

- Some rows have dates that couldn't be parsed
- These rows are excluded from analysis
- Check that date columns contain valid dates

### Negative intervals warning

- Some bookings have visit dates before booking dates
- This indicates data quality issues
- These rows are automatically excluded

## Deployment

### Streamlit Community Cloud (Free)

1. Push code to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. Go to [share.streamlit.io](https://share.streamlit.io)

3. Sign in with GitHub

4. Click "New app" and select:
   - Repository: your GitHub repo
   - Branch: main
   - Main file: app.py

5. Click "Deploy"

Your app will be live at a shareable URL within minutes!

## Tech Stack

- **Streamlit**: Web framework and UI
- **Pandas**: Data processing and analysis
- **Plotly**: Interactive charts
- **openpyxl**: Excel file parsing

## Project Structure

```
sauna-booking-analyzer/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Verify your data format matches the expected structure
3. Ensure all dependencies are installed correctly

## License

This project is created for Kuuma sauna booking analysis.

---

**Built with Streamlit** | Last updated: December 2025
