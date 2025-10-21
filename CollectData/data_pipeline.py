import requests
import pandas as pd
import os
from dotenv import load_dotenv
import hopsworks

load_dotenv()

API_TOKEN = os.environ.get("API_TOKEN")
CITY = os.environ.get("CITY", "karachi")

def fetch_aqi_data(city: str, api_token: str) -> pd.DataFrame:
    """
    Fetches AQI and pollutant data for a given city from the AQICN API,
    and returns a DataFrame with added time-based features.
    
    Parameters:
        city (str): City name (e.g. "karachi")
        api_token (str): AQICN API token
    
    Returns:
        pd.DataFrame: DataFrame containing AQI, pollutant, and time features
    """
    
    url = f"https://api.waqi.info/feed/{city}/?token={api_token}"
    response = requests.get(url)
    data = response.json()
    
    if data.get("status") == "ok":
        aqi = data["data"].get("aqi")
        dominent_pol = data["data"].get("dominentpol")
        iaqi = data["data"].get("iaqi", {})
        
        # Parse time
        time_str = data["data"]["time"]["iso"]
        time_obj = pd.to_datetime(time_str)
        
        # Base record
        record = {
            "city": "Karachi",
            "aqi": aqi,
            "dominent_pol": dominent_pol,
            "time": time_obj,
        }
        
        # Add pollutant and weather values
        for key, val in iaqi.items():
            record[key] = val["v"]
        
        # Add time-based features
        record["hour"] = time_obj.hour
        record["day"] = time_obj.day
        record["month"] = time_obj.month
        
        # Return as DataFrame
        df = pd.DataFrame([record])
        return df
    
    else:
        print("Error fetching AQI data:", data)
        return pd.DataFrame()


def store_in_hopsworks(df: pd.DataFrame):
    """
    Store the DataFrame in Hopsworks Feature Store.
    Creates feature group if it doesn't exist.
    """
    # Login to Hopsworks (using environment variables for API key)
    project = hopsworks.login(api_key_value=os.environ.get("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()
    
    # Feature group name
    feature_group_name = "aqi_features"
    
    # Create or get existing feature group
    try:
        fg = fs.get_feature_group(feature_group_name)
        print("📦 Found existing feature group.")
    except Exception:
        print("🆕 Creating new feature group...")
        fg = fs.create_feature_group(
            name=feature_group_name,
            version=1,
            primary_key=["time"],
            description="Hourly AQI and pollutant data for Karachi",
            online_enabled=True
        )
    
    # Insert new data
    fg.insert(df)
    print("✅ Data successfully inserted into Hopsworks Feature Store.")


if __name__ == "__main__":

    df = fetch_aqi_data(CITY, API_TOKEN)
    
    if not df.empty:
        print("\n✅ Successfully fetched AQI data for",)
        print(df)
        store_in_hopsworks(df)
    else:
        print("\n⚠️ No data fetched or API returned an error.")

