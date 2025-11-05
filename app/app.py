# Save this as app.py

import streamlit as st
import pandas as pd
import os
import requests
import datetime
import joblib
import hopsworks
import numpy as np # Make sure numpy is imported
from dotenv import load_dotenv

# --- 1. Function to Download ALL Models and Metrics ---
@st.cache_resource
def load_all_models_and_metrics():
    """
    Connects to Hopsworks and downloads all 3 models, the scaler,
    and all their metrics.
    """
    with st.spinner("Connecting to Model Registry..."):
        try:
            load_dotenv()
            project = hopsworks.login(project=os.environ.get("HOPSWORKS_PROJECT_NAME"))
            mr = project.get_model_registry()
            
            # This list now includes all 3 models
            model_names = ["ridge", "randomforest", "gradientboosting"]
            models = {}
            all_metrics = {}

            st.write("Downloading all 3 models...")
            
            for name in model_names:
                model_full_name = f"aqi_predictor_{name.lower()}"
                try:
                    model_obj = mr.get_model(name=model_full_name) # Get latest version
                    model_dir = model_obj.download()
                    
                    models[name] = joblib.load(os.path.join(model_dir, "model.pkl"))
                    all_metrics[name] = model_obj.training_metrics
                except Exception as e:
                    st.warning(f"Could not load model '{model_full_name}'. Skipping. Error: {e}")
            
            # The scaler is the same for all, so we just grab one
            scaler_dir = mr.get_model("aqi_predictor_gradientboosting").download()
            scaler = joblib.load(os.path.join(scaler_dir, "scaler.pkl"))
            
            st.success("All models and scaler loaded successfully.")
            return models, scaler, all_metrics
        except Exception as e:
            st.error(f"Fatal error loading models: {e}")
            return None, None, None

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
        
        for i in range(72): # Get all 72 hours
            poll_forecast = poll_data[i]
            poll_dt = poll_forecast['dt']
            forecast_time = datetime.datetime.fromtimestamp(poll_dt, tz=datetime.timezone.utc)
            
            closest_weather = min(
                weather_data_list, 
                key=lambda x: abs(x['dt'] - poll_dt)
            )
            
            # This is the feature order the model expects
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

st.set_page_config(
    page_title="AQI Model Comparison",
    page_icon="☁️",
    layout="wide"
)

# This list must match the *exact* order of features in the function above
FEATURE_NAMES = [
    'pm2_5',
    'pm10',
    'o3',
    'temp',
    'hour_of_day',
    'day_of_month'
]

# Load the OpenWeather API key
try:
    load_dotenv()
    OWM_API_KEY = os.environ["OPENWEATHER_API_KEY"]
except KeyError:
    st.error("OPENWEATHER_API_KEY not found in .env file. Please add it.")
    OWM_API_KEY = None

# Load all models, scaler, AND metrics
models, scaler, all_metrics = load_all_models_and_metrics()

# --- Sidebar ---
st.sidebar.title("Controls")
st.sidebar.info("This app predicts the AQI for Karachi, Pakistan over the next 3 days using 3 different ML models.")
st.sidebar.divider()

# --- Main App Logic ---
if not OWM_API_KEY or not models or not scaler:
    st.sidebar.error("App is not configured. Check keys/models.")
else:
    if st.sidebar.button("🚀 Generate 3-Day Forecast", type="primary"):
        
        # --- Main Page Content ---
        st.title("☁️ 3-Day AQI Forecast & Model Comparison")
        
        with st.spinner("Fetching 72 hours of forecast data..."):
            feature_values_list, display_df = get_3_day_forecast_data(OWM_API_KEY)
        
        if feature_values_list and display_df is not None:
            st.success("Data fetched successfully!")
            
            with st.spinner("Scaling data and making predictions with 3 models..."):
                X_scaled = scaler.transform(feature_values_list)
                
                prediction_df = pd.DataFrame()
                for name, model in models.items():
                    # Capitalize model name for display
                    prediction_df[name.capitalize()] = model.predict(X_scaled)
                
                prediction_df['Forecast Time'] = pd.to_datetime(display_df['Forecast Time'])

            # --- Display the Results ---
            st.header("Forecast Summary")
            
            # --- START OF MODIFIED SECTION ---
            
            st.subheader("Model Prediction Summary (Next 72 Hours)")
            
            # 1. Group by date and get stats for ALL models at once
            daily_groups = prediction_df.groupby(prediction_df['Forecast Time'].dt.date)
            # This creates a multi-index DataFrame
            daily_stats_df = daily_groups[['Gradientboosting', 'Randomforest', 'Ridge']].agg(['mean', 'max']).head(3)

            col1, col2, col3 = st.columns(3)

            # --- GradientBoosting Column ---
            with col1:
                st.write("##### 🥇 GradientBoosting (Best)")
                # Loop through the 3 days for this model
                for date, stats in daily_stats_df.iterrows():
                    day_str = date.strftime('%a, %b %d')
                    st.metric(f"Avg ({day_str})", f"{stats[('Gradientboosting', 'mean')]:.0f}")
                    st.metric(f"Peak ({day_str})", f"{stats[('Gradientboosting', 'max')]:.0f}")
                    st.divider() # Add a small divider between days

            # --- RandomForest Column ---
            with col2:
                st.write("##### 🥈 RandomForest")
                for date, stats in daily_stats_df.iterrows():
                    day_str = date.strftime('%a, %b %d')
                    st.metric(f"Avg ({day_str})", f"{stats[('Randomforest', 'mean')]:.0f}")
                    st.metric(f"Peak ({day_str})", f"{stats[('Randomforest', 'max')]:.0f}")
                    st.divider()

            # --- Ridge Column ---
            with col3:
                st.write("##### 🥉 Ridge")
                for date, stats in daily_stats_df.iterrows():
                    day_str = date.strftime('%a, %b %d')
                    st.metric(f"Avg ({day_str})", f"{stats[('Ridge', 'mean')]:.0f}")
                    st.metric(f"Peak ({day_str})", f"{stats[('Ridge', 'max')]:.0f}")
                    st.divider()

            # --- END OF MODIFIED SECTION ---

            # --- Use Tabs for Chart and Data ---
            tab1, tab2, tab3 = st.tabs(["📈 Model Comparison Chart", "📊 Model Performance", "🗃️ Raw Data"])

            with tab1:
                st.subheader("Predicted AQI over Time (All Models)")
                st.line_chart(prediction_df.set_index('Forecast Time'))

            with tab2:
                st.subheader("Model Performance (from Training)")
                
                perf_data = []
                for name, metrics in all_metrics.items():
                    perf_data.append({
                        "Model": name.capitalize(),
                        "R² Score": metrics.get('r2_score', 0),
                        "MAE (Avg. Error)": metrics.get('mae', 0)
                    })
                perf_df = pd.DataFrame(perf_data).sort_values(by="R² Score", ascending=False)
                
                st.dataframe(perf_df, hide_index=True)
                
                st.divider()
                st.subheader("Model Feature Importance")
                
                col1_fi, col2_fi, col3_fi = st.columns(3)

                # --- GradientBoosting Importance ---
                with col1_fi:
                    st.write("##### GradientBoosting")
                    try:
                        model = models['gradientboosting']
                        importances = model.feature_importances_
                        df_fi = pd.DataFrame({'Feature': FEATURE_NAMES, 'Importance': importances})
                        df_fi = df_fi.sort_values(by='Importance', ascending=False)
                        st.bar_chart(df_fi.set_index('Feature'))
                    except Exception as e:
                        st.error(f"Could not load GB importance: {e}")

                # --- RandomForest Importance ---
                with col2_fi:
                    st.write("##### RandomForest")
                    try:
                        model = models['randomforest']
                        importances = model.feature_importances_
                        df_fi = pd.DataFrame({'Feature': FEATURE_NAMES, 'Importance': importances})
                        df_fi = df_fi.sort_values(by='Importance', ascending=False)
                        st.bar_chart(df_fi.set_index('Feature'))
                    except Exception as e:
                        st.error(f"Could not load RF importance: {e}")

                # --- Ridge Importance (Coefficients) ---
                with col3_fi:
                    st.write("##### Ridge (Coefficient Magnitude)")
                    try:
                        model = models['ridge']
                        importances = np.abs(model.coef_)
                        df_fi = pd.DataFrame({'Feature': FEATURE_NAMES, 'Importance': importances})
                        df_fi = df_fi.sort_values(by='Importance', ascending=False)
                        st.bar_chart(df_fi.set_index('Feature'))
                        st.caption("Note: Shows the absolute value of the feature coefficient.")
                    except Exception as e:
                        st.error(f"Could not load Ridge importance: {e}")

            with tab3:
                st.subheader("Raw Forecast Data")
                # Join predictions with the raw data for the table
                table_df = display_df.join(prediction_df.drop(columns='Forecast Time'))
                table_df['Forecast Time'] = table_df['Forecast Time'].dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(table_df)
            
        else:
            st.error("Failed to get forecast data.")