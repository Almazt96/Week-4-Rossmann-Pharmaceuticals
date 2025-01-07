# Task 3: Model Serving API
# 3.1 Create REST API using FastAPI

from fastapi import FastAPI
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
from fastapi.routing import APIRouter
api_router = APIRouter()

def start_application():
    app = FastAPI()
    app.include_router(api_router)
    return app

model = joblib.load('C:/Users/Almazt/OneDrive - Ethiopian Airlines/Desktop/10 Academy/Rossmann Pharmaceuticals/Week-4/notebooks/model-v1.pkl')

@app.post("/predict/")

def predict(data: dict):
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)
    return {"prediction": prediction[0]}
