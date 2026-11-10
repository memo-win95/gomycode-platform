# tests/test_model.py
import os
import pytest
import numpy as np
import joblib

MODEL_PATH = "models/student_risk_model.joblib"

@pytest.fixture(scope="session")
def trained_model():
    assert os.path.exists(MODEL_PATH), f"L'artefact {MODEL_PATH} doit être généré"
    return joblib.load(MODEL_PATH)

def test_model_prediction_shape(trained_model):
    sample = np.array([[0.8, 0.9, 40.0]])
    proba = trained_model.predict_proba(sample)
    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0][1] <= 1.0

def test_model_risk_logic(trained_model):
    diligent_student = np.array([[1.0, 1.0, 60.0]])
    inactive_student = np.array([[0.1, 0.0, 1.0]])

    risk_diligent = trained_model.predict_proba(diligent_student)[0][1]
    risk_inactive = trained_model.predict_proba(inactive_student)[0][1]

    assert risk_diligent < risk_inactive, "L'étudiant assidu doit avoir un risque inférieur à l'étudiant inactif"