import os
import hopsworks
import pandas as pd
import numpy as np
import joblib  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def train_and_save_model():
    """
    Connects to Hopsworks, reads ML features, trains 3 models,
    and saves ALL of them to the Model Registry.
    """
    
    print("Connecting to Hopsworks...")
    project = hopsworks.login(project=os.environ.get("HOPSWORKS_PROJECT_NAME"))
    fs = project.get_feature_store()

    print("Reading data from 'aqi_ml_training_features'...")
    try:
        fg = fs.get_feature_group(name="aqi_ml_training_features", version=1)
    except Exception as e:
        print(f"Error: Could not find feature group. Run your feature pipeline first. {e}")
        return

    df = fg.read()
    df = df.dropna()

    target = "calculated_aqi"
    features = [
        'pm2_5',
        'pm10',
        'o3',
        'temp',
        'hour_of_day',
        'day_of_month'
    ]
    
    X = df[features]
    y = df[target]

    print("Splitting and scaling data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training and saving models...")
    models = {
        'Ridge': Ridge(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    mr = project.get_model_registry()

    for name, model in models.items():
        print(f"--- Training {name} ---")
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        metrics = {'r2_score': r2, 'mae': mae}
        
        print(f"  - {name}: R² = {r2:.4f}, MAE = {mae:.4f}")

        print(f"Saving {name} to Hopsworks Model Registry...")
        
        model_dir = f"aqi_model_{name.lower()}"
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(model, os.path.join(model_dir, "model.pkl"))
        joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
        
        hopsworks_model_name = f"aqi_predictor_{name.lower()}"
        
        hopsworks_model = mr.sklearn.create_model(
            name=hopsworks_model_name,
            metrics=metrics,
            description=f"AQI predictor (demo model) using {name}."
        )
        
        hopsworks_model.save(model_dir)
        print(f"✅ Model {name} successfully saved as {hopsworks_model_name}!")

    print("\nAll models trained and saved to the registry.")

if __name__ == "__main__":
    train_and_save_model()