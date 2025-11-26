from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="House Price Prediction API")

# Load the saved pipeline
model = joblib.load("house_price_pipeline.pkl")

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
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)
    return {"predicted_house_value": float(prediction[0])}

@app.get("/")
def read_root():
    return {"message": "Welcome to the House Price Prediction API!"}
