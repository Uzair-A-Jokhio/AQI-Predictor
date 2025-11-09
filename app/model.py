import streamlit as st
import pandas as pd
import os
import requests
import datetime
import joblib
import hopsworks
import numpy as np 
from dotenv import load_dotenv

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
            
            model_names = ["ridge", "randomforest", "gradientboosting"]
            models = {}
            all_metrics = {}

            st.write("Downloading all 3 models...")
            
            for name in model_names:
                model_full_name = f"aqi_predictor_{name.lower()}"
                try:
                    model_obj = mr.get_model(name=model_full_name) 
                    model_dir = model_obj.download()
                    
                    models[name] = joblib.load(os.path.join(model_dir, "model.pkl"))
                    all_metrics[name] = model_obj.training_metrics
                except Exception as e:
                    st.warning(f"Could not load model '{model_full_name}'. Skipping. Error: {e}")
            
            
            scaler_dir = mr.get_model("aqi_predictor_gradientboosting").download()
            scaler = joblib.load(os.path.join(scaler_dir, "scaler.pkl"))
            
            st.success("All models and scaler loaded successfully.")
            return models, scaler, all_metrics
        except Exception as e:
            st.error(f"Fatal error loading models: {e}")
            return None, None, None

@st.cache_data(ttl=3600)
def get_3_day_forecast_data(owm_api_key):
    """
    Fetches 72 hours of future pollutant and weather data from OpenWeather.
    """
    LAT = 24.8607
    LON = 67.0011
    
    try:
        pollution_url = (
            f"http://api.openweathermap.org/data/2.5/air_pollution/forecast"
            f"?lat={LAT}&lon={LON}&appid={owm_api_key}"
        )
        response_poll = requests.get(pollution_url)
        response_poll.raise_for_status()
        poll_data = response_poll.json()['list']

        weather_url = (
            f"http://api.openweathermap.org/data/2.5/forecast"
            f"?lat={LAT}&lon={LON}&appid={owm_api_key}&units=metric"
        )
        response_weather = requests.get(weather_url)
        response_weather.raise_for_status()
        weather_data_list = response_weather.json()['list']
        
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
                "Forecast Time": forecast_time, 
                "pm2_5": poll_forecast['components']['pm2_5'],
                "pm10": poll_forecast['components']['pm10'],
                "o3": poll_forecast['components']['o3'],
                "temp": closest_weather['main']['temp']
            })
            
        return all_features, pd.DataFrame(display_data)

    except Exception as e:
        st.error(f"Error fetching data from OpenWeather: {e}")
        return None, None


st.set_page_config(
    page_title="AQI Model Comparison",
    page_icon="☁️",
    layout="wide"
)

FEATURE_NAMES = [
    'pm2_5',
    'pm10',
    'o3',
    'temp',
    'hour_of_day',
    'day_of_month'
]

try:
    load_dotenv()
    OWM_API_KEY = os.environ["OPENWEATHER_API_KEY"]
except KeyError:
    st.error("OPENWEATHER_API_KEY not found in .env file. Please add it.")
    OWM_API_KEY = None

models, scaler, all_metrics = load_all_models_and_metrics()

st.sidebar.title("Controls")
st.sidebar.info("This app predicts the AQI for Karachi, Pakistan over the next 3 days using 3 different ML models.")
st.sidebar.divider()

if not OWM_API_KEY or not models or not scaler:
    st.sidebar.error("App is not configured. Check keys/models.")
else:
    if st.sidebar.button("🚀 Generate 3-Day Forecast", type="primary"):
        
        st.title("☁️ 3-Day AQI Forecast & Model Comparison")
        
        with st.spinner("Fetching 72 hours of forecast data..."):
            feature_values_list, display_df = get_3_day_forecast_data(OWM_API_KEY)
        
        if feature_values_list and display_df is not None:
            st.success("Data fetched successfully!")
            
            with st.spinner("Scaling data and making predictions with 3 models..."):
                X_scaled = scaler.transform(feature_values_list)
                
                prediction_df = pd.DataFrame()
                for name, model in models.items():
                    prediction_df[name.capitalize()] = model.predict(X_scaled)
                
                prediction_df['Forecast Time'] = pd.to_datetime(display_df['Forecast Time'])

            st.header("Forecast Summary")
            
            
            st.subheader("Model Prediction Summary (Next 72 Hours)")
            
            daily_groups = prediction_df.groupby(prediction_df['Forecast Time'].dt.date)
            daily_stats_df = daily_groups[['Gradientboosting', 'Randomforest', 'Ridge']].agg(['mean', 'max']).head(3)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("##### 🥇 GradientBoosting (Best)")
                # Loop through the 3 days for this model
                for date, stats in daily_stats_df.iterrows():
                    day_str = date.strftime('%a, %b %d')
                    st.metric(f"Avg ({day_str})", f"{stats[('Gradientboosting', 'mean')]:.0f}")
                    st.metric(f"Peak ({day_str})", f"{stats[('Gradientboosting', 'max')]:.0f}")
                    st.divider() # Add a small divider between days

            with col2:
                st.write("##### 🥈 RandomForest")
                for date, stats in daily_stats_df.iterrows():
                    day_str = date.strftime('%a, %b %d')
                    st.metric(f"Avg ({day_str})", f"{stats[('Randomforest', 'mean')]:.0f}")
                    st.metric(f"Peak ({day_str})", f"{stats[('Randomforest', 'max')]:.0f}")
                    st.divider()

            with col3:
                st.write("##### 🥉 Ridge")
                for date, stats in daily_stats_df.iterrows():
                    day_str = date.strftime('%a, %b %d')
                    st.metric(f"Avg ({day_str})", f"{stats[('Ridge', 'mean')]:.0f}")
                    st.metric(f"Peak ({day_str})", f"{stats[('Ridge', 'max')]:.0f}")
                    st.divider()


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
                table_df = display_df.join(prediction_df.drop(columns='Forecast Time'))
                table_df['Forecast Time'] = table_df['Forecast Time'].dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(table_df)
            
        else:
            st.error("Failed to get forecast data.")