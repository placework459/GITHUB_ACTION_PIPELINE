import pytest
from fastapi.testclient import TestClient
from app import app  # Make sure your main FastAPI app is in app.py or adjust this import

client = TestClient(app)

# ✅ Sample valid payload based on actual model input features
valid_payload = {
    "features": [
        {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "No",
            "MultipleLines": "No phone service",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 29.85,
            "TotalCharges": 29.85
        },
        {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 34,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "Yes",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "One year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 56.95,
            "TotalCharges": 1889.5
        }
    ]
}

# ❌ Sample empty payload
empty_payload = {
    "features": []
}

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Churn Prediction API is up."}

def test_predict_valid():
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert isinstance(response.json()["predictions"], list)

def test_predict_empty_features():
    response = client.post("/predict", json=empty_payload)
    assert response.status_code == 500
    assert "Prediction failed" in response.json()["detail"]

def test_predict_invalid_payload():
    response = client.post("/predict", json={})
    assert response.status_code == 422  # Pydantic validation error
