import os
import requests
import pandas as pd
import time
from datetime import datetime, timedelta, timezone

# --- Setup: Load Environment Variables ---
# For local execution, create a .env file in the same directory with your API key:
# OPENWEATHER_API_KEY=your_paid_plan_api_key
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded API key from .env file.")
except ImportError:
    print("python-dotenv not found, relying on environment variables.")

# Get API Key from environment
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
if not OPENWEATHER_API_KEY:
    raise ValueError("OPENWEATHER_API_KEY is not set. Please set it in your environment or a .env file.")

# --- Configuration ---
KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011
DAYS_TO_FETCH = 24
OUTPUT_FILE = "historical_aqi_weather_24_days.csv"


def fetch_historical_data():
    """
    Fetches and combines historical pollutant and weather data for the specified number of days
    and saves it to a CSV file.
    """
    # --- 1. Calculate Time Range ---
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=DAYS_TO_FETCH)
    start_ts = int(start_time.timestamp())
    end_ts = int(end_time.timestamp())

    print(f"Fetching data from {start_time.isoformat()} to {end_time.isoformat()}")

    # --- 2. Fetch All Pollutant Data in a Single Call ---
    print(f"Fetching {DAYS_TO_FETCH} days of historical pollutant data...")
    pollution_url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution/history?"
        f"lat={KARACHI_LAT}&lon={KARACHI_LON}&start={start_ts}&end={end_ts}&appid={OPENWEATHER_API_KEY}"
    )
    pollution_response = requests.get(pollution_url)
    if pollution_response.status_code != 200:
        raise Exception(f"Error fetching pollutant data: {pollution_response.text}")

    pollutant_list = pollution_response.json().get('list', [])
    # Create a dictionary for fast O(1) lookups by timestamp
    pollutant_map = {item['dt']: item for item in pollutant_list}
    print(f"Found {len(pollutant_map)} hourly pollutant records.")

    # --- 3. Fetch Matching Weather Data Hour by Hour ---
    all_data_rows = []
    timestamps = sorted(pollutant_map.keys())
    total_timestamps = len(timestamps)

    print(f"Fetching matching weather data for {total_timestamps} hours. This will take approximately {round(total_timestamps * 0.7 / 60, 1)} minutes...")

    for i, ts in enumerate(timestamps):
        pollution_data = pollutant_map[ts]
        weather_url = (
            f"https://api.openweathermap.org/data/3.0/onecall/timemachine?"
            f"lat={KARACHI_LAT}&lon={KARACHI_LON}&dt={ts}&units=metric&appid={OPENWEATHER_API_KEY}"
        )
        weather_response = requests.get(weather_url)

        if weather_response.status_code != 200:
            print(f"Warning: Could not get weather for timestamp {ts}. Status: {weather_response.status_code}. Skipping.")
            continue

        weather_data = weather_response.json().get('data', [{}])[0]
        if not weather_data:
            print(f"Warning: No weather data returned for {ts}. Skipping.")
            continue

        # --- 4. Combine Data into a Single Row ---
        try:
            combined_row = {
                'timestamp_int': ts,
                'timestamp_utc': datetime.utcfromtimestamp(ts),
                'latitude': KARACHI_LAT,
                'longitude': KARACHI_LON,
                'aqi': pollution_data['main']['aqi'],
                'co': pollution_data['components']['co'],
                'no2': pollution_data['components']['no2'],
                'o3': pollution_data['components']['o3'],
                'so2': pollution_data['components']['so2'],
                'pm2_5': pollution_data['components']['pm2_5'],
                'pm10': pollution_data['components']['pm10'],
                'temp': weather_data['temp'],
                'feels_like': weather_data['feels_like'],
                'pressure': weather_data['pressure'],
                'humidity': weather_data['humidity'],
                'wind_speed': weather_data.get('wind_speed', 0),
                'clouds': weather_data.get('clouds', 0),
            }
            all_data_rows.append(combined_row)
        except KeyError as e:
            print(f"Warning: Missing data key for timestamp {ts}. Skipping. Error: {e}")

        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"  ...processed {i + 1} / {total_timestamps} records")
        
        # --- IMPORTANT: Rate Limiting ---
        # A 0.7s delay keeps you under the 100 calls/minute limit for most paid plans.
        time.sleep(0.7)

    # --- 5. Create DataFrame and Save to CSV ---
    if not all_data_rows:
        print("No data was fetched successfully. Exiting.")
        return

    print("\nCombining all data into a DataFrame...")
    df = pd.DataFrame(all_data_rows)
    df = df.sort_values(by='timestamp_int', ascending=True).reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Success! All {len(df)} data rows saved to '{OUTPUT_FILE}'")
    print("\nData Preview:")
    print(df.head())


if __name__ == "__main__":
    fetch_historical_data()