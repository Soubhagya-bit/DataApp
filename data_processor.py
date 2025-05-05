import pandas as pd
import numpy as np
from datetime import datetime

def load_and_process_data(file_path):
    """
    Load and process the air quality data from a CSV file.
    
    Args:
        file_path: Path to the CSV file containing air quality data
        
    Returns:
        Processed DataFrame
    """
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Handle any missing values
    df = df.dropna(subset=['value', 'lat', 'lon'])
    
    return df

def filter_data_by_date_range(df, start_date, end_date):
    """
    Filter data by date range.
    
    Args:
        df: DataFrame containing air quality data
        start_date: Start date for filtering
        end_date: End date for filtering
        
    Returns:
        Filtered DataFrame
    """
    # Convert date objects to datetime if needed
    if not isinstance(start_date, datetime):
        start_date = pd.to_datetime(start_date)
    
    if not isinstance(end_date, datetime):
        end_date = pd.to_datetime(end_date)
    
    # Add one day to end_date to include the entire day
    end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    
    # Filter by date range
    return df[(df['date'] >= start_date) & (df['date'] < end_date)]

def filter_data_by_location(df, locations):
    """
    Filter data by selected locations.
    
    Args:
        df: DataFrame containing air quality data
        locations: List of location names to filter by
        
    Returns:
        Filtered DataFrame
    """
    if not locations:
        return df  # If no locations are selected, return the original DataFrame
    
    return df[df['location'].isin(locations)]

def filter_data_by_parameter(df, parameters):
    """
    Filter data by selected parameters.
    
    Args:
        df: DataFrame containing air quality data
        parameters: Single parameter or list of parameters to filter by
        
    Returns:
        Filtered DataFrame
    """
    if isinstance(parameters, str):
        parameters = [parameters]
    
    return df[df['parameter'].isin(parameters)]

def aggregate_data_for_time_series(df, parameter, freq='D'):
    """
    Aggregate data for time series visualization.
    
    Args:
        df: DataFrame containing filtered air quality data
        parameter: The parameter to aggregate
        freq: Frequency for aggregation (e.g., 'D' for daily)
        
    Returns:
        Aggregated DataFrame suitable for time series plotting
    """
    # Filter for the specific parameter
    param_df = df[df['parameter'] == parameter].copy()
    
    # Group by date and location, then calculate the mean value
    grouped = param_df.groupby(['date', 'location'])['value'].mean().reset_index()
    
    return grouped

def aggregate_data_for_comparison(df, parameter):
    """
    Aggregate data for location comparison visualization.
    
    Args:
        df: DataFrame containing filtered air quality data
        parameter: The parameter to compare across locations
        
    Returns:
        Aggregated DataFrame suitable for bar chart plotting
    """
    # Filter for the specific parameter
    param_df = df[df['parameter'] == parameter].copy()
    
    # Group by location and calculate mean, min, max values
    grouped = param_df.groupby('location').agg({
        'value': ['mean', 'min', 'max']
    }).reset_index()
    
    # Flatten the multi-index columns
    grouped.columns = ['location', 'mean', 'min', 'max']
    
    return grouped

def prepare_correlation_data(df, param1, param2):
    """
    Prepare data for correlation analysis between two parameters.
    
    Args:
        df: DataFrame containing air quality data
        param1: First parameter for correlation
        param2: Second parameter for correlation
        
    Returns:
        DataFrame with values for both parameters
    """
    # Get data for the first parameter
    df1 = df[df['parameter'] == param1].copy()
    df1 = df1.rename(columns={'value': param1})
    df1 = df1[['date', 'location', param1]]
    
    # Get data for the second parameter
    df2 = df[df['parameter'] == param2].copy()
    df2 = df2.rename(columns={'value': param2})
    df2 = df2[['date', 'location', param2]]
    
    # Merge the two dataframes on date and location
    merged = pd.merge(df1, df2, on=['date', 'location'], how='inner')
    
    return merged
