# src/ml/train.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def train_dropout_model():
    # Exemple de données synthétiques représentatives des cohortes GoMyCode
    data = {
        "attendance_rate": [0.95, 0.40, 0.85, 0.30, 0.70, 0.20, 0.90, 0.50, 0.15, 0.80],
        "assignments_completed_ratio": [1.0, 0.30, 0.90, 0.20, 0.75, 0.10, 0.95, 0.40, 0.05, 0.85],
        "platform_time_hours": [60.0, 10.5, 45.0, 8.0, 35.0, 4.0, 52.0, 18.0, 2.0, 40.0],
        "dropout_status": [0, 1, 0, 1, 0, 1, 0, 1, 1, 0]  # 0: Réussite, 1: Abandon
    }
    df = pd.DataFrame(data)

    X = df[["attendance_rate", "assignments_completed_ratio", "platform_time_hours"]]
    y = df["dropout_status"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
    clf.fit(X_train, y_train)

    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Modèle entraîné avec succès. ROC-AUC: {auc:.2f}")

    os.makedirs("models", exist_ok=True)
    output_path = "models/student_risk_model.joblib"
    joblib.dump(clf, output_path)
    print(f"Modèle sérialisé sous : {output_path}")

if __name__ == "__main__":
    train_dropout_model()
