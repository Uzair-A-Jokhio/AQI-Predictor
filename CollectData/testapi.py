import requests
from dotenv import load_dotenv
import os

load_dotenv()

open_key = os.environ.get("OPEN_API")
lat=24.8608
lon=67.0104
url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={open_key}"
response = requests.get(url)
data = response.json()

print(data)