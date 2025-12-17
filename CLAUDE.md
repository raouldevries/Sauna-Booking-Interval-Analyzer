# Kuuma Booking Analyzer - Project Guidelines

## UI/UX Guidelines

### Icons vs Emoticons
- **Always use Material icons** (`:material/icon_name:`) instead of emoticons/emojis
- Example: Use `:material/lightbulb:` instead of `💡`
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

## App Deployment
- App is deployed on **Streamlit Cloud**
- Local file storage does not persist (ephemeral)
- For persistent data, use external services (Google Sheets, Supabase, etc.)

## Data Files
- Booking data: df1 (booking creation), df2 (visit dates)
- Marketing data: Google Ads, Meta Ads CSV files
- Location column variations: 'Location', 'Tour', 'Activity'
