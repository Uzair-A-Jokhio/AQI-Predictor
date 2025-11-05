import os
import requests
import pandas as pd
import hopsworks
from datetime import datetime

def fetch_openweather_data(lat, lon, api_key):
    """Fetches weather and air pollution data from OpenWeatherMap."""
    
    pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    pollution_response = requests.get(pollution_url)
    pollution_data = pollution_response.json()['list'][0]

    weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    weather_response = requests.get(weather_url)
    weather_data = weather_response.json()['list'][0]
    dt_int = pollution_data['dt'] 
    dt_obj = datetime.utcfromtimestamp(dt_int)
    combined_data = {
        'timestamp_int': [dt_int],   
        'timestamp_utc': [dt_obj],    
        'latitude': [lat],
        'longitude': [lon],
        'aqi': [pollution_data['main']['aqi']],
        'co': [pollution_data['components']['co']],
        'no2': [pollution_data['components']['no2']],
        'o3': [pollution_data['components']['o3']],
        'so2': [pollution_data['components']['so2']],
        'pm2_5': [pollution_data['components']['pm2_5']],
        'pm10': [pollution_data['components']['pm10']],
        'temp': [weather_data['main']['temp']],
        'feels_like': [weather_data['main']['feels_like']],
        'pressure': [weather_data['main']['pressure']],
        'humidity': [weather_data['main']['humidity']],
        'wind_speed': [weather_data['wind']['speed']],
        'clouds': [weather_data['clouds']['all']],
    }

    df = pd.DataFrame(combined_data)
    print("Successfully fetched and combined data.")
    print(df.head())
    return df

def load_to_hopsworks(df, project_name):
    """Connects to Hopsworks and inserts data into a Feature Group."""
    
    project = hopsworks.login(project=project_name)
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="aqi_weather_data_hourly",
        version=1,
        description="Hourly weather and air quality index data for Karachi.",
        primary_key=['timestamp_int'], 
        event_time="timestamp_utc", 
        online_enabled=True,
    )

    fg.insert(df, write_options={"wait_for_job": True})
    print("Successfully inserted data into Hopsworks Feature Group.")

if __name__ == "__main__":

    OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
    HOPSWORKS_PROJECT_NAME = os.environ.get("HOPSWORKS_PROJECT_NAME")
    HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")

    KARACHI_LAT = 24.8607
    KARACHI_LON = 67.0011

    if not all([OPENWEATHER_API_KEY, HOPSWORKS_PROJECT_NAME, HOPSWORKS_API_KEY]):
        raise ValueError("One or more required environment variables are not set.")
    data_df = fetch_openweather_data(KARACHI_LAT, KARACHI_LON, OPENWEATHER_API_KEY)
    
    load_to_hopsworks(data_df, HOPSWORKS_PROJECT_NAME)