import pathlib
import json
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, create_model
from typing import Any

from catboost import CatBoostClassifier

# Application directory
BASE_DIR = pathlib.Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Load feature schema
with open(MODEL_DIR / "feature_schema.json", "r") as f:
    schema = json.load(f)

feature_names = schema["feature_names"]
cat_feature_names = schema["categorical_features"]

# Load CatBoost models
model_paths = {
    "baseline": MODEL_DIR / "baseline.cbm",
    "raw_weighted": MODEL_DIR / "raw_amount_weighted.cbm",
    "sqrt_weighted": MODEL_DIR / "sqrt_amount_weighted.cbm"
}

models = {}

for name, path in model_paths.items():
    model = CatBoostClassifier()
    model.load_model(str(path))
    models[name] = model

# Preprocess incoming transaction data
def preprocess_transaction(df):
    missing_cols = set(feature_names) - set(df.columns)
    extra_cols = set(df.columns) - set(feature_names)

    if missing_cols:
        raise ValueError(f"Missing features: {missing_cols}")

    if extra_cols:
        raise ValueError(f"Unexpected features: {extra_cols}")

    df = df[feature_names].copy()

    for col in cat_feature_names:
        df[col] = df[col].fillna("__MISSING__").astype(str)

    return df

# Generate fraud predictions
def predict_transaction(df):
    X = preprocess_transaction(df)

    baseline_prob = models["baseline"].predict_proba(X)[:, 1]
    raw_weighted_prob = models["raw_weighted"].predict_proba(X)[:, 1]
    sqrt_weighted_prob = models["sqrt_weighted"].predict_proba(X)[:, 1]

    results = pd.DataFrame({
        "baseline_probability": baseline_prob,
        "probability_x_amount": baseline_prob * X["TransactionAmt"].values,
        "raw_weighted_probability": raw_weighted_prob,
        "sqrt_weighted_probability": sqrt_weighted_prob
    })

    return results

# Define transaction input schema
TransactionInput = create_model(
    "TransactionInput",
    **{
        feature: (Any, ...)
        for feature in feature_names
    }
)

# Create FastAPI application
app = FastAPI(
    title="IEEE-CIS Fraud Detection API",
    description="API serving three CatBoost fraud detection models and four scoring strategies.",
    version="1.0.0"
)


# Health-check endpoint
@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running!",
        "status": "healthy"
    }


# Fraud prediction endpoint
@app.post("/predict")
def predict_endpoint(transaction: TransactionInput):
    transaction_dict = transaction.model_dump()
    transaction_df = pd.DataFrame([transaction_dict])

    result = predict_transaction(transaction_df)

    return result.iloc[0].to_dict()

