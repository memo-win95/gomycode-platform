import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="GoMyCode Student Risk Prediction API")
# Téléchargement du modèle entraîné dans GP-4
MODEL_PATH = "models/student_risk_model.joblib"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None

# Définition de la structure des données d'entrée via Pydantic
class StudentFeatures(BaseModel):
    attendance_rate: float = Field(..., ge=0.0, le=1.0, example=0.85)
    assignments_completed_ratio: float = Field(..., ge=0.0, le=1.0, example=0.90)
    platform_time_hours: float = Field(..., ge=0.0, example=45.0)

# Point de terminaison (Endpoint) pour la prédiction
@app.post("/predict")
def predict_dropout(features: StudentFeatures):
    if model is None:
        raise HTTPException(
            status_code=500, detail="Modèle non chargé ou introuvable"
        )

    # Conversion des données d'entrée en structures de données (DataFrames) compatibles avec le modèle
    data = pd.DataFrame([features.dict()])

    # Pour prédire la probabilité d'une interruption
    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[:, 1][0])

    return {
        "dropout_prediction": prediction,
        "dropout_probability": round(probability, 4),
        "status": "At Risk" if prediction == 1 else "Normal",
    } 