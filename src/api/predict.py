import joblib

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

@app.get("/dashboard/alerts")
def get_alerts():
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    
    # عينة بيانات لمحاكاة طلاب المنصة
    students_data = [
        {"student_id": 101, "name": "Yassine B.", "attendance_rate": 0.45, "assignments_completed_ratio": 0.40, "platform_time_hours": 12.0},
        {"student_id": 102, "name": "Sarra M.", "attendance_rate": 0.95, "assignments_completed_ratio": 0.92, "platform_time_hours": 65.0},
        {"student_id": 103, "name": "Ahmed T.", "attendance_rate": 0.55, "assignments_completed_ratio": 0.50, "platform_time_hours": 18.0},
    ]
    
    alerts = []
    for s in students_data:
        features = pd.DataFrame([{
            "attendance_rate": s["attendance_rate"],
            "assignments_completed_ratio": s["assignments_completed_ratio"],
            "platform_time_hours": s["platform_time_hours"]
        }])
        
        prob = float(model.predict_proba(features)[:, 1][0])
        pred = int(model.predict(features)[0])
        
        alerts.append({
            "student_id": s["student_id"],
            "name": s["name"],
            "dropout_probability": round(prob, 4),
            "risk_level": "HIGH" if prob >= 0.6 else "LOW",
            "status": "At Risk" if pred == 1 else "Normal"
        })
        
    return alerts 
alerts = []
@app.get("/dashboard/alerts") 

def get_alerts():
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")

    students_data = [
        {
            "student_id": 101,
            "name": "Yassine B.",
            "attendance_rate": 0.45,
            "assignments_completed_ratio": 0.40,
            "platform_time_hours": 12.0,
        },
        {
            "student_id": 102,
            "name": "Sarra M.",
            "attendance_rate": 0.95,
            "assignments_completed_ratio": 0.92,
            "platform_time_hours": 65.0,
        },
        {
            "student_id": 103,
            "name": "Ahmed T.",
            "attendance_rate": 0.55,
            "assignments_completed_ratio": 0.50,
            "platform_time_hours": 18.0,
        },
    ]

    alerts = []
    for s in students_data:
        features = [[
            s["attendance_rate"],
            s["assignments_completed_ratio"],
            s["platform_time_hours"],
        ]]

        prob = float(model.predict_proba(features)[:, 1][0])
        pred = int(model.predict(features)[0])

        alerts.append({
            "student_id": s["student_id"],
            "name": s["name"],
            "dropout_probability": round(prob, 4),
            "risk_level": "HIGH" if prob >= 0.6 else "LOW",
            "status": "At Risk" if pred == 1 else "Normal",
        })

    return alerts