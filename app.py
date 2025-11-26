from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path  # for relative paths

app = FastAPI(title="House Price Prediction API")

# Load the saved model using a path relative to this file
model_path = Path(__file__).parent / "house_price_model.pkl"  # <-- use correct file name
model = joblib.load(model_path)

class HouseData(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float

@app.post("/predict")
def predict_price(data: HouseData):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])
    # Make prediction
    prediction = model.predict(input_df)
    return {"predicted_house_value": float(prediction[0])}

@app.get("/")
def read_root():
    return {"message": "Welcome to the House Price Prediction API!"}
