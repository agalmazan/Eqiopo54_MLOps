# Student Performance Prediction API

REST API service built with FastAPI to serve the Student Performance prediction model.

## 📋 Overview

This API exposes the trained Decision Tree model for real-time predictions of student performance based on demographic, academic, and behavioral features.

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r src/api/requirements.txt
```

### Running the API

```bash
# From project root
cd Equipo54_MLOps

# Run with uvicorn
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs (Swagger)**: http://localhost:8000/docs
- **Alternative Docs (ReDoc)**: http://localhost:8000/redoc

## 📡 API Endpoints

### POST `/predict`
Make a prediction for a student's performance.

**Request Body:**
```json
{
  "gender": "Male",
  "caste": "General",
  "mathematics_marks": 85,
  "english_marks": 78,
  "science_marks": 82,
  "father_occupation": "Government Officer",
  "mother_occupation": "Teacher",
  "number_of_siblings": 2,
  "boarding": "No",
  "distance_from_home": "Near",
  "time": 5,
  "coaching": "Yes"
}
```

**Response:**
```json
{
  "prediction": "Good",
  "probability": 0.85,
  "model_version": "latest"
}
```

### GET `/health`
Check API and model health status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "latest"
}
```

### GET `/model/info`
Get information about the loaded model.

**Response:**
```json
{
  "model_version": "latest",
  "model_type": "DecisionTreeClassifier",
  "features": ["Gender", "Caste", "mathematics_marks", ...],
  "target_classes": ["Average", "Good", "Very Good", "Excellent"]
}
```

## 🔧 Testing the API

### Option 1: Postman Collection

Import the provided Postman collection for easy testing:

**File**: `src/api/postman_collection.json`

The collection includes:
- ✅ Health check endpoint
- ✅ Model info endpoint
- ✅ Prediction examples (Good, Excellent, Average students)
- ✅ Validation error test (invalid data)

**How to use:**
1. Open Postman
2. Click "Import" → Select `postman_collection.json`
3. Run requests from the collection

### Option 2: curl Commands

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "caste": "General",
    "mathematics_marks": 85,
    "english_marks": 78,
    "science_marks": 82,
    "father_occupation": "Government Officer",
    "mother_occupation": "Teacher",
    "number_of_siblings": 2,
    "boarding": "No",
    "distance_from_home": "Near",
    "time": 5,
    "coaching": "Yes"
  }'
```

### Option 3: Python Test Script

```bash
# Run automated tests
python src/api/test_api.py
```

This will validate all endpoints and return a test report.

## 🐍 Testing with Python

```python
import requests

# API base URL
BASE_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Make prediction
student_data = {
    "gender": "Female",
    "caste": "OBC",
    "mathematics_marks": 92,
    "english_marks": 88,
    "science_marks": 90,
    "father_occupation": "Business",
    "mother_occupation": "Teacher",
    "number_of_siblings": 1,
    "boarding": "Yes",
    "distance_from_home": "Near",
    "time": 6,
    "coaching": "Yes"
}

response = requests.post(f"{BASE_URL}/predict", json=student_data)
print(response.json())
# Output: {"prediction": "Excellent", "probability": 0.92, "model_version": "latest"}
```

## 📦 Model Artifact Information

The API loads the trained model from:
- **Path**: `models/latest/decision_tree_model.pkl`
- **Encoders**: `models/latest/label_encoders.pkl`
- **Version**: `latest` (or specific version from MLflow registry)
- **Registry Path**: `models:/student-performance-dt/<version>` (when using MLflow)

### Model Versioning

The model can be loaded from different sources:
1. **Local artifacts**: `models/latest/` (default, updated by pipeline)
2. **MLflow Model Registry**: `models:/student-performance-dt/1` (for production)

## 🔐 Input Validation

All inputs are validated using Pydantic schemas:

- **Gender**: Male, Female
- **Caste**: General, OBC, SC, ST
- **Marks**: 0-100 (integer)
- **Father/Mother Occupation**: Predefined categories
- **Number of Siblings**: 0-10 (integer)
- **Boarding**: Yes, No
- **Distance**: Near, Far, Very Far
- **Study Time**: 0-24 hours (integer)
- **Coaching**: Yes, No

Invalid inputs will return a `400 Bad Request` with error details.

## 🛠️ Development

### Project Structure

```
src/api/
├── __init__.py          # Package initialization
├── main.py              # FastAPI application and endpoints
├── model.py             # Model loading and prediction logic
├── schemas.py           # Pydantic models for validation
├── requirements.txt     # API-specific dependencies
└── README.md           # This file
```

### Adding New Endpoints

1. Define request/response schemas in `schemas.py`
2. Add endpoint logic in `main.py`
3. Update this README with endpoint documentation

## 🚢 Deployment

### Docker (Optional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/api /app/src/api
COPY models/latest /app/models/latest

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t student-performance-api .
docker run -p 8000:8000 student-performance-api
```

## 📊 Monitoring

- Access logs are printed to console
- Prediction requests and responses are logged
- Model loading status is logged on startup

## 🐛 Troubleshooting

**Model not found error:**
- Ensure model files exist in `models/latest/`
- Run the training pipeline first: `python src/pipeline/run_pipeline.py`

**Import errors:**
- Install all dependencies: `pip install -r src/api/requirements.txt`
- Ensure you're running from the project root

**Port already in use:**
- Change port: `uvicorn src.api.main:app --port 8001`

## 📝 License

Same as parent project (see root LICENSE file).
