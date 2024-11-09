import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Energy Consumption Dashboard",
    layout="wide"
)

# Add title
st.title("Energy Consumption Analysis Dashboard")

# Load data
@st.cache_data
def load_data():
    # Read tab-delimited CSV with comma as decimal separator
    df = pd.read_csv('data/consumption.csv', 
                     sep='\t',            # Tab delimiter
                     decimal=',',         # Comma decimal separator
                     parse_dates=['Timestamp'],  # Parse timestamp column
                     dayfirst=False)      # Assuming YYYY-MM-DD format
    
    # Handle any potential data cleaning
    df = df.dropna()  # Remove any rows with missing values
    
    # Ensure consumption values are numeric
    df['Consumption'] = pd.to_numeric(df['Consumption'], errors='coerce')
    
    return df

# Load the data
try:
    df = load_data()
    
    # Verify data is loaded correctly
    if df.empty:
        st.error("The dataset is empty after loading. Please check the file format.")
        st.stop()
    
    if df['Consumption'].isna().any():
        st.warning("Some consumption values could not be parsed. They have been removed from the visualization.")
        df = df.dropna()
    
    # Add time period selector
    period_options = {
        'Day': 'D',
        'Week': 'W',
        'Month': 'M'
    }
    selected_period = st.selectbox('Select Aggregation Period', options=list(period_options.keys()))
    
    # Create date range slider
    min_date = df['Timestamp'].min().date()
    max_date = df['Timestamp'].max().date()
    
    # Convert dates to datetime for slider
    slider_col1, slider_col2 = st.columns([3, 1])
    
    with slider_col1:
        selected_range = st.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="DD/MM/YYYY"
        )
    
    start_date, end_date = selected_range
    
    with slider_col2:
        st.markdown("**Selected Range**")
        st.write(f"From: {start_date.strftime('%d/%m/%Y')}")
        st.write(f"To: {end_date.strftime('%d/%m/%Y')}")
    
    # Filter data based on selected date range
    mask = (df['Timestamp'].dt.date >= start_date) & (df['Timestamp'].dt.date <= end_date)
    filtered_df = df.loc[mask].copy()
    
    # Aggregate data based on selected period
    if selected_period != 'Day':
        filtered_df = filtered_df.resample(period_options[selected_period], on='Timestamp').agg({
            'Consumption': 'sum'
        }).reset_index()
    
    # Display summary statistics
    st.subheader("Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Average Consumption (MWh)",
            f"{filtered_df['Consumption'].mean():.2f}"
        )
    
    with col2:
        st.metric(
            "Peak Consumption (MWh)",
            f"{filtered_df['Consumption'].max():.2f}"
        )
    
    with col3:
        st.metric(
            "Total Consumption (MWh)",
            f"{filtered_df['Consumption'].sum():.2f}"
        )
    
    with col4:
        st.metric(
            "Number of Records",
            len(filtered_df)
        )
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Chart", "Data"])
    
    with tab1:
        # Create interactive plot
        fig = px.line(
            filtered_df,
            x='Timestamp',
            y='Consumption',
            title=f'Energy Consumption Over Time ({selected_period}ly View)'
        )
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Consumption (MWh)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Add download button
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name=f"energy_consumption_{start_date}_to_{end_date}.csv",
            mime='text/csv'
        )
        
        # Display dataframe with filtered data
        st.dataframe(
            filtered_df,
            column_config={
                "Timestamp": st.column_config.DatetimeColumn(format="DD-MM-YYYY HH:mm"),
                "Consumption": st.column_config.NumberColumn(format="%.2f MWh")
            }
        )

except FileNotFoundError:
    st.error("Error: Cannot find consumption.csv in the data folder. Please ensure the file exists in the correct location.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.error("Please check your CSV file format. It should be tab-delimited with two columns: 'Timestamp' and 'Consumption'")
    st.code("""
Expected file format example:
Timestamp    Consumption
2024-01-01 00:00    2,19
2024-01-01 00:15    4,78
    """)
    st.stop()