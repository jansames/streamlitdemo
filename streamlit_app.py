import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io

# Set page configuration
st.set_page_config(
    page_title="Energy Consumption Dashboard",
    layout="wide"
)

# Add title
st.title("Energy Consumption Analysis Dashboard")

def validate_csv(df):
    """Validate the uploaded CSV file"""
    errors = []
    
    # Check number of records
    if len(df) != 35136:
        errors.append(f"CSV must contain exactly 35136 records. Found: {len(df)}")
    
    # Check columns
    required_columns = ['Timestamp', 'Consumption']
    if not all(col in df.columns for col in required_columns):
        errors.append("CSV must contain 'Timestamp' and 'Consumption' columns")
        return errors
    
    # Check timestamp format
    try:
        pd.to_datetime(df['Timestamp'])
    except:
        errors.append("Timestamp column must be in format 'YYYY-MM-DD HH:mm'")
    
    # Check consumption values
    try:
        pd.to_numeric(df['Consumption'].str.replace(',', '.'))
    except:
        errors.append("Consumption values must be numeric (using comma as decimal separator)")
    
    return errors

def process_dataframe(df):
    """Process the dataframe after validation"""
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Consumption'] = pd.to_numeric(df['Consumption'].str.replace(',', '.'))
    
    # Extract year and create normalized timestamp (without year)
    year = df['Timestamp'].dt.year.iloc[0]
    df['NormalizedDate'] = df['Timestamp'].dt.strftime('%m-%d %H:%M')
    df = df.rename(columns={'Consumption': str(year)})
    
    return df, year

def generate_forecast(df, source_year, target_year):
    """Generate forecast data"""
    forecast_series = df[str(source_year)].copy()
    return forecast_series

def aggregate_data(df, period):
    """Aggregate data based on selected period"""
    if period == 'Timestamp':
        return df
    
    agg_map = {
        'Day': 'D',
        'Week': 'W',
        'Month': 'M'
    }
    
    # Convert NormalizedDate back to datetime for aggregation
    temp_df = df.copy()
    base_year = 2000  # Use any non-leap year as base
    temp_df['TempDate'] = pd.to_datetime(f'{base_year}-' + temp_df['NormalizedDate'])
    
    # Perform aggregation
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    agg_dict = {col: 'sum' for col in numeric_columns}
    
    result = temp_df.resample(agg_map[period], on='TempDate').agg(agg_dict).reset_index()
    result['NormalizedDate'] = result['TempDate'].dt.strftime('%m-%d %H:%M')
    result = result.drop('TempDate', axis=1)
    
    return result

# File upload section
st.subheader("1. Upload Historical Data")
uploaded_file = st.file_uploader("Upload your energy consumption data (CSV)", type=['csv'])

if uploaded_file is not None:
    # Read and validate the file
    content = uploaded_file.read().decode('utf-8')
    df = pd.read_csv(io.StringIO(content), sep='\t')
    
    # Validate the data
    validation_errors = validate_csv(df)
    
    if validation_errors:
        for error in validation_errors:
            st.error(error)
        st.stop()
    
    # Process the dataframe
    df, source_year = process_dataframe(df)
    
    # Store the original data in session state
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = df
        st.session_state.source_year = source_year
    
    
    # Forecast section
    st.subheader("2. Generate Forecast")
    forecast_cols = st.columns([2, 1, 1], vertical_alignment="bottom")
    
    with forecast_cols[0]:
        forecast_year = st.number_input('Select forecast year', 
                                      min_value=source_year + 1,
                                      max_value=source_year + 10,
                                      value=source_year + 1)
        
    with forecast_cols[1]:
        if st.button('Generate Forecast', use_container_width=True):
            forecast_series = generate_forecast(df, source_year, forecast_year)
            # Store both the forecast year and the complete forecast data
            st.session_state.forecast_year = forecast_year
            st.session_state.has_forecast = True
            st.session_state.forecast_data = forecast_series
            # Update the working DataFrame
            df[str(forecast_year)] = forecast_series
            st.success(f'Forecast generated for year {forecast_year}')
    
    # Data visualization section
    st.subheader("3. Data Analysis")
    
    # Period selector
    period_options = ['Timestamp', 'Day', 'Week', 'Month']
    selected_period = st.selectbox('Select Aggregation Period', options=period_options)
    
    # Create date range slider (using normalized dates)
    df['TempDate'] = pd.to_datetime('2000-' + df['NormalizedDate'])  # Use non-leap year
    min_date = df['TempDate'].min().date()
    max_date = df['TempDate'].max().date()
    
    selected_range = st.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="DD/MM"
    )
    
    start_date, end_date = selected_range
    
    # Ensure forecast data is included in the DataFrame
    if 'has_forecast' in st.session_state and st.session_state.has_forecast:
        df[str(st.session_state.forecast_year)] = st.session_state.forecast_data
    
    # Filter and aggregate data
    mask = (df['TempDate'].dt.strftime('%m-%d') >= start_date.strftime('%m-%d')) & \
           (df['TempDate'].dt.strftime('%m-%d') <= end_date.strftime('%m-%d'))
    filtered_df = df.loc[mask].copy()
    aggregated_df = aggregate_data(filtered_df, selected_period)
    
    # Forecast adjustment section
    if 'has_forecast' in st.session_state and st.session_state.has_forecast:
        st.subheader("4. Adjust Forecast")
        adj_cols = st.columns([2, 1, 1], vertical_alignment="bottom")
        
        with adj_cols[0]:
            multiplicator = st.number_input('Enter multiplicator for forecast adjustment', 
                                          min_value=0.1, 
                                          max_value=10.0, 
                                          value=1.0,
                                          step=0.1)
        
        with adj_cols[1]:
            if st.button('Adjust Selection with Multiplicator', use_container_width=True):
                forecast_year = str(st.session_state.forecast_year)
                source_year = str(st.session_state.source_year)
                
                # Get the date range mask for the raw data
                date_mask = (df['TempDate'].dt.strftime('%m-%d') >= start_date.strftime('%m-%d')) & \
                           (df['TempDate'].dt.strftime('%m-%d') <= end_date.strftime('%m-%d'))
                
                # Create temporary series for the adjustment
                temp_forecast = df[source_year].copy()
                # Apply multiplicator only to the selected date range
                temp_forecast[date_mask] = df.loc[date_mask, source_year] * multiplicator
                # Keep original values for dates outside the selected range
                temp_forecast[~date_mask] = st.session_state.forecast_data[~date_mask]
                
                # Update both the session state and the working DataFrame
                st.session_state.forecast_data = temp_forecast
                df[forecast_year] = temp_forecast
                
                # Recompute filtered and aggregated data
                filtered_df = df.loc[mask].copy()
                aggregated_df = aggregate_data(filtered_df, selected_period)
                
                st.success('Forecast adjusted successfully')
    
    # Display summary statistics
    st.subheader("Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Historical (MWh)", 
                 f"{aggregated_df[str(source_year)].mean():.2f}")
    with col2:
        st.metric("Peak Historical (MWh)", 
                 f"{aggregated_df[str(source_year)].max():.2f}")
    
    if 'has_forecast' in st.session_state and st.session_state.has_forecast:
        forecast_year = str(st.session_state.forecast_year)
        if forecast_year in aggregated_df.columns:
            with col3:
                st.metric("Average Forecast (MWh)", 
                         f"{aggregated_df[forecast_year].mean():.2f}")
            with col4:
                st.metric("Peak Forecast (MWh)", 
                         f"{aggregated_df[forecast_year].max():.2f}")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Chart", "Data"])
    
    with tab1:
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=aggregated_df['NormalizedDate'],
            y=aggregated_df[str(source_year)],
            name=f'Historical ({source_year})',
            line=dict(color='blue')
        ))
        
        # Add forecast if available
        if 'has_forecast' in st.session_state and st.session_state.has_forecast:
            forecast_year = str(st.session_state.forecast_year)
            if forecast_year in aggregated_df.columns:
                fig.add_trace(go.Scatter(
                    x=aggregated_df['NormalizedDate'],
                    y=aggregated_df[forecast_year],
                    name=f'Forecast ({forecast_year})',
                    line=dict(color='red')
                ))
        
        fig.update_layout(
            title=f'Energy Consumption Over Time ({selected_period}ly View)',
            xaxis_title="Time",
            yaxis_title="Consumption (MWh)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Add download button
        csv = aggregated_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name=f"energy_consumption_{start_date.strftime('%m-%d')}_to_{end_date.strftime('%m-%d')}.csv",
            mime='text/csv'
        )
        
        # Configure columns based on available data
        column_config = {
            "NormalizedDate": st.column_config.Column("Date/Time"),
            str(source_year): st.column_config.NumberColumn(
                f"Historical ({source_year})", 
                format="%.2f MWh"
            )
        }
        
        # Add forecast column configuration only if forecast exists
        if 'has_forecast' in st.session_state and st.session_state.has_forecast:
            forecast_year = str(st.session_state.forecast_year)
            if forecast_year in aggregated_df.columns:
                column_config[forecast_year] = st.column_config.NumberColumn(
                    f"Forecast ({forecast_year})", 
                    format="%.2f MWh"
                )
        
        st.dataframe(
            aggregated_df,
            column_config=column_config
        )