import streamlit as st
import pandas as pd
import hopsworks
import altair as alt
import os
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import numpy as np 

# --- AQI HELPER FUNCTIONS ---
# These are needed to process the raw data
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
# --- END AQI HELPER FUNCTIONS ---


# --- Function to connect to Hopsworks and get the Feature Store ---
@st.cache_resource
def get_feature_store():
    """Connects to Hopsworks and returns the Feature Store handle."""
    with st.spinner("Connecting to Hopsworks Feature Store..."):
        try:
            load_dotenv()
            project = hopsworks.login(project=os.environ.get("HOPSWORKS_PROJECT_NAME"))
            fs = project.get_feature_store()
            return fs
        except Exception as e:
            st.error(f"Error connecting to Hopsworks: {e}")
            return None

# --- Function to load and process historical data ---
@st.cache_data
def load_eda_data(_fs):
    """
    Fetches all raw data from Hopsworks and calculates AQI for EDA.
    """
    with st.spinner("Loading all historical data for analysis..."):
        try:
            fg_raw = _fs.get_feature_group(name="aqi_weather_data_hourly", version=1)
            df_raw = fg_raw.read()
            
            # Calculate AQI
            df_raw['calculated_aqi'] = df_raw.apply(calculate_overall_aqi, axis=1)
            # Engineer time features for plotting
            df_raw['timestamp_utc'] = pd.to_datetime(df_raw['timestamp_utc'])
            df_raw['hour_of_day'] = df_raw['timestamp_utc'].dt.hour
            
            st.success("Historical data loaded.")
            return df_raw
        except Exception as e:
            st.error(f"Error loading EDA data: {e}")
            return pd.DataFrame()

# --- Page UI ---
st.set_page_config(
    page_title="Historical Data Explorer",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Historical Data Explorer")
st.write("This page analyzes all historical data from the `aqi_weather_data_hourly` feature group.")

fs = get_feature_store()

if fs:
    eda_df = load_eda_data(fs) # Load the raw data
    
    if not eda_df.empty:
        
        # --- 1. Correlation Heatmap ---
        st.subheader("Feature Correlation Heatmap")
        st.write("This shows how features relate to the calculated AQI and each other.")
        
        # Define columns for correlation
        eda_corr_cols = [
            'pm2_5', 'pm10', 'o3', 'temp', 'hour_of_day', 
            'calculated_aqi', 'wind_speed', 'humidity', 'pressure', 'clouds'
        ]
        # Filter for columns that actually exist in the df
        existing_cols = [col for col in eda_corr_cols if col in eda_df.columns]
        eda_corr = eda_df[existing_cols].corr()
        
        # Plot with Matplotlib and Seaborn
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(eda_corr, annot=True, fmt=".2f", cmap="vlag", linewidths=0.5, ax=ax)
        st.pyplot(fig)
        st.divider()

        # --- 2. Historical AQI over time ---
        st.subheader("Historical AQI over Time")
        st_df = eda_df.set_index('timestamp_utc')['calculated_aqi']
        st.line_chart(st_df)
        st.divider()

        # --- 3. NEW: Pollutant Trends Over Time ---
        st.subheader("Historical Pollutant Trends (Raw Concentrations)")
        st.write("See how all raw pollutant levels (in µg/m³) trend together over time.")
        pollutant_cols = ['pm2_5', 'pm10', 'o3', 'no2', 'co', 'so2']
        pollutant_df = eda_df.set_index('timestamp_utc')[pollutant_cols]
        st.line_chart(pollutant_df)
        st.divider()

        # --- 4. NEW: Average Pollutant Levels by Hour ---
        st.subheader("Average Pollutant Levels by Hour of Day")
        st.write("This helps identify daily patterns, like rush hour traffic (NO2/CO spikes) or ozone (O3) spikes in the afternoon.")
        
        # Group by hour and calculate the mean for each pollutant
        # We must 'melt' the dataframe to a long format for Altair to use
        hourly_avg_df = eda_df.groupby('hour_of_day')[pollutant_cols].mean().reset_index()
        hourly_avg_melted = hourly_avg_df.melt('hour_of_day', var_name='Pollutant', value_name='Average Concentration (µg/m³)')

        # Create the grouped bar chart
        hourly_chart = alt.Chart(hourly_avg_melted).mark_bar().encode(
            x=alt.X('hour_of_day:O', title='Hour of Day'), # 'O' treats it as a category
            y=alt.Y('Average Concentration (µg/m³):Q'), # 'Q' treats it as a quantity
            color='Pollutant:N', # 'N' treats it as a nominal (string) category
            tooltip=['hour_of_day', 'Pollutant', 'Average Concentration (µg/m³)']
        ) # --- REMOVED .interactive() ---
        st.altair_chart(hourly_chart, use_container_width=True)
        st.divider()

    
    else:
        st.error("Could not load EDA data.")
else:
    st.error("Failed to connect to Hopsworks. Check your .env file.")