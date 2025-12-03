from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="House Price Prediction API")

# =====================
# ⭐ CORS FIX (IMPORTANT)
# =====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # or ["https://your-frontend.com"]
    allow_credentials=True,
    allow_methods=["*"],       # <--- FIXES OPTIONS 405
    allow_headers=["*"],
)

# Optional but safe: Manual OPTIONS handler for /predict
@app.options("/predict")
def options_handler():
    return {"message": "OK"}


# =====================
# Load Model
# =====================
model_path = Path(__file__).parent / "house_price_model.pkl"
model = joblib.load(model_path)


# =====================
# Request Model
# =====================
class HouseData(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float


# =====================
# Predict Route
# =====================
@app.post("/predict")
def predict_price(data: HouseData):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)
    return {"predicted_house_value": float(prediction[0])}


# =====================
# Root Route
# =====================
@app.get("/")
def read_root():
    return {"message": "Welcome to the House Price Prediction API!"}
