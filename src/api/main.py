# src/api/main.py
import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="GoMyCode Dropout Risk API", version="1.0.0")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/student_risk_model.joblib")

def get_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modèle introuvable à l'emplacement : {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

class StudentMetrics(BaseModel):
    attendance_rate: float = Field(..., ge=0.0, le=1.0, description="Taux de présence (0.0 à 1.0)")
    assignments_completed_ratio: float = Field(..., ge=0.0, le=1.0, description="Ratio devoirs rendus (0.0 à 1.0)")
    platform_time_hours: float = Field(..., ge=0.0, description="Heures passées sur la plateforme")

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "GoMyCode Analytics"}

@app.post("/predict-dropout-risk")
def predict_risk(metrics: StudentMetrics):
    try:
        model = get_model()
        features = np.array([[
            metrics.attendance_rate,
            metrics.assignments_completed_ratio,
            metrics.platform_time_hours
        ]])
        probability = float(model.predict_proba(features)[0][1])

        return {
            "dropout_risk_score": round(probability, 3),
            "status": "High Risk" if probability > 0.60 else "Normal"
        }
    except FileNotFoundError as fnf_err:
        raise HTTPException(status_code=500, detail=str(fnf_err))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))