import pandas as pd
import os

# --- Configuration ---
# 1. Set the path to your CSV file
CSV_FILE_PATH = "merged_dataset.csv" # <--- IMPORTANT: Update this path

# 2. Set the coordinates (required for your primary key)
KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011

# 3. Set your Hopsworks Project Name
# (Script will read this from your .env file)
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded .env file.")
except ImportError:
    print("python-dotenv not found, relying on environment variables.")
    

def transform_dataframe(df):
    """
    Transforms the raw CSV data to match the Hopsworks feature group schema.
    """
    print("Transforming data...")
    
    # 1. Handle Time columns
    # Convert the string 'datetime' column to a proper datetime object
    # This becomes our 'event_time'
    df['timestamp_utc'] = pd.to_datetime(df['datetime'], utc=True)
    # Convert to naive UTC datetime (removes timezone info, as expected by Hopsworks)
    df['timestamp_utc'] = df['timestamp_utc'].dt.tz_convert(None)
    
    # Create the integer timestamp (Unix epoch) for the primary key
    # NEW CORRECTED CODE
    df['timestamp_int'] = (df['timestamp_utc'].astype('int64') // 10**9)

    # 2. Rename columns
    df = df.rename(columns={'aqi_index': 'aqi'})

    # 3. Add missing required columns
    df['latitude'] = KARACHI_LAT
    df['longitude'] = KARACHI_LON
    
    # Add 'feels_like' as a null (NA) value since it's not in the CSV
    if 'feels_like' not in df.columns:
        df['feels_like'] = pd.NA

    # 4. Define and select the final set of columns
    # This drops extra columns like 'datetime', 'visibility', 'weather_main'
    final_columns = [
        'timestamp_int', 'timestamp_utc', 'latitude', 'longitude',
        'aqi', 'co', 'no2', 'o3', 'so2', 'pm2_5', 'pm10',
        'temp', 'feels_like', 'pressure', 'humidity', 'wind_speed', 'clouds'
    ]
    
    # Ensure all expected columns are present
    for col in final_columns:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: '{col}' after transformation.")
            
    transformed_df = df[final_columns]
    print("Transformation complete.")
    transformed_df.to_csv("HopworkData.csv")


if __name__ == "__main__":
    print(f"Loading data from '{CSV_FILE_PATH}'...")
    csv = pd.read_csv(CSV_FILE_PATH)
    data = transform_dataframe(csv)

