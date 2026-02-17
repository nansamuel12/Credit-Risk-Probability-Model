# 🚀 Credit Risk Prediction API Documentation

## Overview

This FastAPI application provides a production-ready REST API for predicting credit risk probability using Machine Learning models trained on customer RFM (Recency, Frequency, Monetary) features.

---

## 🎯 Features

- ✅ **MLflow Model Registry Integration** - Automatically loads production models
- ✅ **Model Versioning** - Tracks model and preprocessor versions
- ✅ **Batch Predictions** - Support for single and batch inference
- ✅ **Input Validation** - Pydantic schemas for data validation
- ✅ **Health Checks** - Monitor API and model status
- ✅ **Error Handling** - Comprehensive error responses
- ✅ **OpenAPI Documentation** - Auto-generated interactive docs

---

## 📦 Installation

### Prerequisites

```bash
pip install fastapi uvicorn mlflow scikit-learn pandas numpy pydantic
```

All dependencies are already in `requirements.txt`.

---

## 🚀 Running the API

### Method 1: Using Uvicorn directly

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Method 2: Using Python

```bash
python -m uvicorn app.main:app --reload
```

### Method 3: Running the script directly

```bash
python app/main.py
```

The API will be available at:
- **Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

---

## 📚 API Endpoints

### 1. Root Endpoint

**GET /** 

Get API information and status.

**Response:**
```json
{
  "message": "Credit Risk Probability Prediction API",
  "version": "1.0.0",
  "model": "credit_risk_randomforest",
  "model_version": "1",
  "status": "online"
}
```

---

### 2. Health Check

**GET /health**

Check if the API and models are loaded and healthy.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "credit_risk_randomforest",
  "model_version": "1",
  "preprocessor_version": "1"
}
```

**Status Codes:**
- `200 OK` - API is healthy
- `503 Service Unavailable` - Models not loaded

---

### 3. Model Information

**GET /model/info**

Get detailed information about the loaded model.

**Response:**
```json
{
  "model_name": "credit_risk_randomforest",
  "model_version": "1",
  "preprocessor_version": "1",
  "stage": "Production",
  "features": ["Recency", "Frequency", "Monetary"],
  "output": "Risk probability (0 = Low Risk, 1 = High Risk)"
}
```

---

### 4. Single Prediction

**POST /predict/single**

Predict credit risk for a single customer.

**Request Body:**
```json
{
  "Recency": 30,
  "Frequency": 15,
  "Monetary": 5000.0
}
```

**Response:**
```json
{
  "prediction": {
    "risk_probability": 0.15,
    "low_risk_probability": 0.85,
    "risk_label": 0
  },
  "model_name": "credit_risk_randomforest",
  "model_version": "1"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/predict/single" \
  -H "Content-Type: application/json" \
  -d '{
    "Recency": 30,
    "Frequency": 15,
    "Monetary": 5000.0
  }'
```

---

### 5. Batch Prediction

**POST /predict**

Predict credit risk for multiple customers in one request.

**Request Body:**
```json
{
  "features": [
    {
      "Recency": 5,
      "Frequency": 25,
      "Monetary": 15000.0
    },
    {
      "Recency": 180,
      "Frequency": 2,
      "Monetary": 300.0
    },
    {
      "Recency": 60,
      "Frequency": 10,
      "Monetary": 5000.0
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "risk_probability": 0.08,
      "low_risk_probability": 0.92,
      "risk_label": 0
    },
    {
      "risk_probability": 0.95,
      "low_risk_probability": 0.05,
      "risk_label": 1
    },
    {
      "risk_probability": 0.35,
      "low_risk_probability": 0.65,
      "risk_label": 0
    }
  ],
  "model_name": "credit_risk_randomforest",
  "model_version": "1",
  "preprocessor_version": "1"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      {"Recency": 5, "Frequency": 25, "Monetary": 15000.0},
      {"Recency": 180, "Frequency": 2, "Monetary": 300.0}
    ]
  }'
```

---

## 📊 Feature Descriptions

### Input Features (RFM)

| Feature | Description | Type | Constraints |
|---------|-------------|------|-------------|
| **Recency** | Days since last transaction | float | >= 0 |
| **Frequency** | Total number of transactions | float | >= 1 |
| **Monetary** | Total amount spent | float | >= 0 |

### Output Fields

| Field | Description | Type | Range |
|-------|-------------|------|-------|
| **risk_probability** | Probability of high risk | float | 0.0 - 1.0 |
| **low_risk_probability** | Probability of low risk | float | 0.0 - 1.0 |
| **risk_label** | Binary prediction | int | 0 or 1 |

**Risk Labels:**
- `0` = Low Risk (safe customer)
- `1` = High Risk (risky customer)

---

## 🧪 Testing the API

### Option 1: Using the Test Script

```bash
python test_api.py
```

This will run comprehensive tests including:
- ✅ Health checks
- ✅ Single prediction
- ✅ Batch prediction
- ✅ Error handling

### Option 2: Interactive API Documentation

Visit http://localhost:8000/docs for Swagger UI where you can:
- View all endpoints
- Test requests interactively
- See request/response schemas

### Option 3: Using Python Requests

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict/single",
    json={
        "Recency": 30,
        "Frequency": 15,
        "Monetary": 5000.0
    }
)

print(response.json())
```

---

## 🔧 Configuration

### Environment Variables

You can configure the API using environment variables:

```bash
# MLflow tracking URI (optional)
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"

# Model registry URI (optional)
export MLFLOW_REGISTRY_URI="sqlite:///mlflow.db"
```

### Model Loading Strategy

The API uses a **cascading fallback** strategy:

1. **Primary**: Load from MLflow Model Registry (Production stage)
2. **Fallback**: Load from local `models/` directory

This ensures the API works even if MLflow registry is unavailable.

---

## 📈 Production Deployment

### Using Docker (Recommended)

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

### Using Production ASGI Server

```bash
# Using Gunicorn with Uvicorn workers
gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

---

## 🔒 Security Considerations

### For Production:

1. **Add Authentication**: Implement API key or OAuth2
   ```python
   from fastapi.security import APIKeyHeader
   ```

2. **Rate Limiting**: Prevent abuse
   ```python
   from slowapi import Limiter
   ```

3. **HTTPS**: Use SSL/TLS certificates

4. **Input Sanitization**: Already handled by Pydantic

5. **CORS**: Configure allowed origins
   ```python
   from fastapi.middleware.cors import CORSMiddleware
   ```

---

## 📊 Monitoring

### Logging

The API logs all predictions and errors. View logs:

```bash
# Redirect logs to file
uvicorn app.main:app --log-config logging.conf > api.log 2>&1
```

### Metrics

Track these metrics in production:
- Request count
- Response time
- Prediction accuracy (with feedback loop)
- Error rate
- Model version usage

---

## 🐛 Troubleshooting

### Issue: "Models not loaded" error

**Solution:**
1. Ensure models exist in `models/` directory
2. Run training first: `python -m src.train`
3. Check MLflow registry: `mlflow ui`

### Issue: "Connection refused"

**Solution:**
- Ensure API is running: `uvicorn app.main:app`
- Check port 8000 is not in use

### Issue: Prediction errors

**Solution:**
1. Verify input format matches schema
2. Check feature constraints (non-negative values)
3. Review API logs for details

---

## 📚 Example Use Cases

### Use Case 1: Customer Risk Screening

Screen new customers before approval:

```python
customer_data = {
    "Recency": 10,
    "Frequency": 5,
    "Monetary": 2000.0
}

response = requests.post(
    "http://localhost:8000/predict/single",
    json=customer_data
)

risk = response.json()["prediction"]["risk_label"]
if risk == 1:
    print("⚠️  High risk - require additional verification")
else:
    print("✅ Low risk - approve")
```

### Use Case 2: Batch Risk Assessment

Assess entire customer portfolio:

```python
customers = [
    {"Recency": r, "Frequency": f, "Monetary": m}
    for r, f, m in customer_database
]

response = requests.post(
    "http://localhost:8000/predict",
    json={"features": customers}
)

high_risk_count = sum(
    p["risk_label"] for p in response.json()["predictions"]
)

print(f"High risk customers: {high_risk_count}/{len(customers)}")
```

---

## 📞 Support

For issues or questions:
1. Check logs: API logs contain detailed error messages
2. Review documentation: http://localhost:8000/docs
3. Run tests: `python test_api.py`

---

## 🎉 Summary

Your Credit Risk Prediction API is production-ready with:
- ✅ FastAPI framework
- ✅ MLflow model registry integration
- ✅ Comprehensive validation
- ✅ Interactive documentation
- ✅ Health monitoring
- ✅ Test suite included

**Start the API:**
```bash
uvicorn app.main:app --reload
```

**Access Documentation:**
```
http://localhost:8000/docs
```

**Run Tests:**
```bash
python test_api.py
```
