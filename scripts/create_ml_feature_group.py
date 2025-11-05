import pandas as pd
import numpy as np
import hopsworks
import os


MW = {'co': 28.01, 'o3': 48.00, 'no2': 46.01, 'so2': 64.07}
BREAKPOINTS = {
    'co': [((0.0, 4.4), (0, 50)), ((4.5, 9.4), (51, 100)), ((9.5, 12.4), (101, 150)), ((12.5, 15.4), (151, 200)), ((15.5, 30.4), (201, 300)), ((30.5, 40.4), (301, 400)), ((40.5, 50.4), (401, 500))],
    'no2': [((0, 53), (0, 50)), ((54, 100), (51, 100)), ((101, 360), (101, 150)), ((361, 649), (151, 200)), ((650, 1249), (201, 300)), ((1250, 1649), (301, 400)), ((1650, 2049), (401, 500))],
    'o3': [((0, 54), (0, 50)), ((55, 70), (51, 100)), ((71, 85), (101, 150)), ((86, 105), (151, 200)), ((106, 200), (201, 300))],
    'so2': [((0, 35), (0, 50)), ((36, 75), (51, 100)), ((76, 185), (101, 150)), ((186, 304), (151, 200)), ((305, 604), (201, 300)), ((605, 804), (301, 400)), ((805, 1004), (401, 500))],
    'pm2_5': [((0.0, 9.0), (0, 50)), ((9.1, 35.4), (51, 100)), ((35.5, 55.4), (101, 150)), ((55.5, 150.4), (151, 200)), ((150.5, 250.4), (201, 300)), ((250.5, 350.4), (301, 400)), ((350.5, 500.4), (401, 500))],
    'pm10': [((0, 54), (0, 50)), ((55, 154), (51, 100)), ((155, 254), (101, 150)), ((255, 354), (151, 200)), ((355, 424), (201, 300)), ((425, 504), (301, 400)), ((505, 604), (401, 500))],
}

def ugm3_to_ppb(ugm3, pollutant_name):
    if pollutant_name not in MW: return ugm3
    return (ugm3 * 24.45) / MW[pollutant_name]

def ugm3_to_ppm(ugm3, pollutant_name):
    return ugm3_to_ppb(ugm3, pollutant_name) / 1000

def calculate_sub_index(conc, pollutant):
    if pd.isna(conc): return np.nan
    if pollutant == 'pm2_5': conc = np.floor(conc * 10) / 10
    elif pollutant == 'pm10': conc = np.floor(conc)
    elif pollutant == 'co': conc = np.floor(ugm3_to_ppm(conc, pollutant) * 10) / 10
    elif pollutant in ['no2', 'o3', 'so2']: conc = np.floor(ugm3_to_ppb(conc, pollutant))
    else: return np.nan
    for (cl, ch), (al, ah) in BREAKPOINTS[pollutant]:
        if cl <= conc <= ch:
            return round(((ah - al) / (ch - cl)) * (conc - cl) + al)
    if conc > BREAKPOINTS[pollutant][-1][0][1]: return 500
    return np.nan

def calculate_overall_aqi(row):
    pollutants = ['pm2_5', 'pm10', 'o3', 'co', 'no2', 'so2']
    sub_indices = [calculate_sub_index(row[p], p) for p in pollutants if p in row]
    valid_indices = [i for i in sub_indices if pd.notna(i)]
    return max(valid_indices) if valid_indices else np.nan


def run_feature_pipeline():
    """
    Connects to Hopsworks, reads raw data, transforms it, 
    and saves it to a new ML-ready feature group.
    """
    
    print("Connecting to Hopsworks...")
    project = hopsworks.login(project=os.environ.get("HOPSWORKS_PROJECT_NAME"))
    fs = project.get_feature_store()

    print("Extracting raw data from 'aqi_weather_data_hourly'...")
    fg_raw = fs.get_feature_group(name="aqi_weather_data_hourly", version=1)
    df = fg_raw.read()
    df = df.sort_values(by="timestamp_int")

    print("Transforming data (calculating AQI and features)...")
    
    df['calculated_aqi'] = df.apply(calculate_overall_aqi, axis=1)
    
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp_utc']):
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
    
    df['hour_of_day'] = df['timestamp_utc'].dt.hour
    df['day_of_month'] = df['timestamp_utc'].dt.day 
    
    final_ml_features = [
        'timestamp_int',  
        'pm2_5',
        'pm10',
        'o3',
        'temp',
        'hour_of_day',
        'day_of_month',
        'calculated_aqi' 
    ]
    
    ml_df = df[final_ml_features].copy()
    ml_df = ml_df.dropna()
    
    print(f"Created ML-ready dataset with {ml_df.shape[0]} rows.")

    
    new_fg_name = "aqi_ml_training_features"
    print(f"Loading data into new ML feature group '{new_fg_name}'...")
    
    fg_ml = fs.get_or_create_feature_group(
        name=new_fg_name,
        version=1,
        description="Cleaned, ML-ready features (pm25, pm10, o3, temp, hour, day) for AQI training.",
        primary_key=['timestamp_int'],
        online_enabled=True,
    )

    fg_ml.insert(ml_df, write_options={"wait_for_job": True})
    
    print(f"\n✅ Successfully created and populated '{new_fg_name}'!")
    print("\nPreview of data in new feature group:")
    print(ml_df.head())


if __name__ == "__main__":
    
    if not os.environ.get("HOPSWORKS_PROJECT_NAME") or not os.environ.get("HOPSWORKS_API_KEY"):
        print("Error: HOPSWORKS_PROJECT_NAME or HOPSWORKS_API_KEY not set in .env file.")
    else:
        run_feature_pipeline()