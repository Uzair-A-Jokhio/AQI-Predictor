# Save this as app.py

import streamlit as st
import pandas as pd
import os
import requests
import datetime
import joblib
import hopsworks
from dotenv import load_dotenv

# --- 1. Function to Download Model from Registry ---
@st.cache_resource
def load_model_from_registry():
    """
    Connects to Hopsworks and downloads the model and scaler.
    """
    with st.spinner("Connecting to Model Registry..."):
        try:
            load_dotenv()
            project = hopsworks.login(project=os.environ.get("HOPSWORKS_PROJECT_NAME"))
            mr = project.get_model_registry()
            
            model_obj = mr.get_model(name="aqi_predictor_gradientboosting", version=1)
            model_dir = model_obj.download()
            
            model = joblib.load(os.path.join(model_dir, "model.pkl"))
            scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
            
            return model, scaler
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, None

# --- 2. Function to Get 3-Day Forecast Data ---
@st.cache_data(ttl=3600) # Cache the API call for 1 hour
def get_3_day_forecast_data(owm_api_key):
    """
    Fetches 72 hours of future pollutant and weather data from OpenWeather.
    """
    LAT = 24.8607
    LON = 67.0011
    
    try:
        # === API Call 1: Get Future Pollutants (Hourly) ===
        pollution_url = (
            f"http://api.openweathermap.org/data/2.5/air_pollution/forecast"
            f"?lat={LAT}&lon={LON}&appid={owm_api_key}"
        )
        response_poll = requests.get(pollution_url)
        response_poll.raise_for_status()
        poll_data = response_poll.json()['list']

        # === API Call 2: Get Future Weather (3-Hourly) ===
        weather_url = (
            f"http://api.openweathermap.org/data/2.5/forecast"
            f"?lat={LAT}&lon={LON}&appid={owm_api_key}&units=metric"
        )
        response_weather = requests.get(weather_url)
        response_weather.raise_for_status()
        weather_data_list = response_weather.json()['list']
        
        # === 3. Process all 72 hours ===
        all_features = []
        display_data = []
        
        for i in range(72):
            poll_forecast = poll_data[i]
            poll_dt = poll_forecast['dt']
            forecast_time = datetime.datetime.fromtimestamp(poll_dt, tz=datetime.timezone.utc)
            
            closest_weather = min(
                weather_data_list, 
                key=lambda x: abs(x['dt'] - poll_dt)
            )
            
            feature_row = [
                poll_forecast['components']['pm2_5'],
                poll_forecast['components']['pm10'],
                poll_forecast['components']['o3'],
                closest_weather['main']['temp'],
                forecast_time.hour,
                forecast_time.day
            ]
            all_features.append(feature_row)
            
            display_data.append({
                "Forecast Time": forecast_time, # Keep as datetime for charting
                "pm2_5": poll_forecast['components']['pm2_5'],
                "pm10": poll_forecast['components']['pm10'],
                "o3": poll_forecast['components']['o3'],
                "temp": closest_weather['main']['temp']
            })
            
        return all_features, pd.DataFrame(display_data)

    except Exception as e:
        st.error(f"Error fetching data from OpenWeather: {e}")
        return None, None

# --- 3. Streamlit UI ---

# Page config (must be the first st command)
st.set_page_config(
    page_title="AQI 3-Day Forecaster",
    page_icon="☁️",
    layout="wide"
)

# Load the OpenWeather API key
try:
    load_dotenv()
    OWM_API_KEY = os.environ["OPENWEATHER_API_KEY"]
except KeyError:
    st.error("OPENWEATHER_API_KEY not found in .env file. Please add it.")
    OWM_API_KEY = None

# Load the model and scaler
model, scaler = load_model_from_registry()

# --- Sidebar ---
st.sidebar.title("Controls")
st.sidebar.info("This app predicts the AQI for Karachi, Pakistan over the next 3 days.")

if not OWM_API_KEY or not model or not scaler:
    st.sidebar.error("App is not configured. Missing API keys or failed model load.")
else:
    if st.sidebar.button("🚀 Generate 3-Day Forecast", type="primary"):
        
        # --- Main Page Content ---
        st.title("☁️ 3-Day Air Quality Forecast")
        
        with st.spinner("Fetching 72 hours of forecast data..."):
            feature_values_list, display_df = get_3_day_forecast_data(OWM_API_KEY)
        
        if feature_values_list and display_df is not None:
            st.success("Data fetched successfully!")
            
            with st.spinner("Scaling data and making 72 predictions..."):
                X_scaled = scaler.transform(feature_values_list)
                predictions = model.predict(X_scaled)
            
            # --- Display the Results ---
            display_df['Predicted AQI'] = predictions
            display_df['Forecast Time'] = pd.to_datetime(display_df['Forecast Time'])
            
            # Calculate summary metrics
            avg_aqi = round(display_df['Predicted AQI'].mean())
            max_aqi = round(display_df['Predicted AQI'].max())
            max_aqi_time = display_df.loc[display_df['Predicted AQI'].idxmax()]['Forecast Time'].strftime('%a, %b %d at %I %p')

            st.header("Forecast Summary")
            col1, col2 = st.columns(2)
            col1.metric("Average AQI (Next 3 Days)", f"{avg_aqi}")
            col2.metric(f"Peak AQI (on {max_aqi_time})", f"{max_aqi}")

            # --- Use Tabs for Chart and Data ---
            tab1, tab2 = st.tabs(["📈 Forecast Chart", "🗃️ Raw Data"])

            with tab1:
                st.subheader("Predicted AQI over Time")
                st.line_chart(display_df.set_index('Forecast Time')['Predicted AQI'])

            with tab2:
                st.subheader("Forecast Data Table")
                # Format the time for better display in the table
                display_df['Forecast Time'] = display_df['Forecast Time'].dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(display_df)
            
        else:
            st.error("Failed to get forecast data.")