import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.models import FertilizerInput
from src.ml_utils import FertilizerPredictionApp, train_and_save_model

# Ensure data and models directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Train model if not exists
train_and_save_model()

app = FastAPI(
    title="Fertilizer Recommendation API",
    description="ML API for predicting the most suitable fertilizer based on soil and crop conditions",
    version="1.0.0"
)

# Initialize prediction app
prediction_app = FertilizerPredictionApp()

@app.post("/predict")
async def predict_fertilizer(input_data: FertilizerInput):
    try:
        # Get prediction
        fertilizer_prediction = prediction_app.predict(input_data)
        
        return {
            "predicted_fertilizer": fertilizer_prediction,
            "input_data": dict(input_data)  # Use dict() instead of model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Fertilizer Prediction API",
        "endpoints": {
            "prediction": "/predict (POST)"
        }
    }