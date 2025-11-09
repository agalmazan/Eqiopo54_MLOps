# Student Performance Prediction API

REST API service built with FastAPI to serve the Student Performance prediction model.

## 📋 Overview

This API exposes the trained Decision Tree model for real-time predictions of student performance based on demographic, academic, and behavioral features.

> **⚠️ Important**: All input values must match the exact format from the training data. The model expects:
> - **Categorical values in UPPERCASE** (e.g., `"MALE"`, not `"Male"`)
> - **Percentages as categories** (`"EXCELLENT"`, `"GOOD"`, `"AVERAGE"`, `"VG"`)
> - **Time as text** (`"FIVE"`, not `5`)
> - See [Input Validation](#-input-validation) section for complete details

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r src/api/requirements.txt

# Pull model artifacts from DVC (required!)
dvc pull
```

### Running the API

**Option 1: Using helper script (Recommended)**
```bash
# From project root
cd Equipo54_MLOps
python run_api.py
```

**Option 2: Using uvicorn directly**
```bash
# From project root
cd Equipo54_MLOps
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
  "gender": "MALE",
  "caste": "GENERAL",
  "coaching": "OA",
  "time": "FIVE",
  "Class_ten_education": "CBSE",
  "twelve_education": "CBSE",
  "medium": "ENGLISH",
  "Class_ X_Percentage": "EXCELLENT",
  "Class_XII_Percentage": "GOOD",
  "Father_occupation": "BANK_OFFICIAL",
  "Mother_occupation": "HOUSE_WIFE"
}
```

**Response:**
```json
{
  "prediction": "Excellent",
  "probability": 0.397,
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
  "features": [
    "Gender", "Caste", "coaching", "time", 
    "Class_ten_education", "twelve_education", "medium",
    "Class_ X_Percentage", "Class_XII_Percentage",
    "Father_occupation", "Mother_occupation"
  ],
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
    "gender": "MALE",
    "caste": "GENERAL",
    "coaching": "OA",
    "time": "FIVE",
    "Class_ten_education": "CBSE",
    "twelve_education": "CBSE",
    "medium": "ENGLISH",
    "Class_ X_Percentage": "EXCELLENT",
    "Class_XII_Percentage": "GOOD",
    "Father_occupation": "BANK_OFFICIAL",
    "Mother_occupation": "HOUSE_WIFE"
  }'
```

### Option 3: Python Test Script

**Quick Test:**
```bash
# From project root
python test_predict.py
```

**Full Test Suite:**
```bash
# Run automated tests (if available)
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
    "gender": "FEMALE",
    "caste": "OBC",
    "coaching": "WA",
    "time": "SEVEN",
    "Class_ten_education": "CBSE",
    "twelve_education": "CBSE",
    "medium": "ENGLISH",
    "Class_ X_Percentage": "EXCELLENT",
    "Class_XII_Percentage": "EXCELLENT",
    "Father_occupation": "ENGINEER",
    "Mother_occupation": "COLLEGE_TEACHER"
}

response = requests.post(f"{BASE_URL}/predict", json=student_data)
print(response.json())
# Output: {"prediction": "Excellent", "probability": 0.45, "model_version": "latest"}
```

## 📦 Model Artifact Information

The API loads the trained model from:
- **Path**: `models/latest/decision_tree_model.pkl`
- **Encoders**: `models/latest/label_encoders.pkl`
- **Version**: `latest` (or specific version from MLflow registry)
- **Registry Path**: `models:/student-performance-dt/<version>` (when using MLflow)

> **⚠️ Important**: Model artifacts are managed by **DVC** and are NOT in git.
> Before running the API, you must either:
> - Pull models from DVC: `dvc pull`
> - Or run the training pipeline: `python src/pipeline/run_pipeline.py`

### Model Versioning

The model can be loaded from different sources:
1. **DVC tracked artifacts**: `models/latest/` (managed by DVC, pull with `dvc pull`)
2. **Local artifacts**: `models/latest/` (generated by pipeline)
3. **MLflow Model Registry**: `models:/student-performance-dt/1` (for production)

## 🔐 Input Validation

All inputs are validated using Pydantic schemas. **Note**: All values match the training data format.

### Required Fields:

- **gender**: `"MALE"`, `"FEMALE"`, `"NAN"`
- **caste**: `"GENERAL"`, `"OBC"`, `"SC"`, `"ST"`
- **coaching**: `"NO"`, `"OA"` (Online/Offline Available), `"WA"` (Weekend Available)
- **time**: `"ONE"`, `"TWO"`, `"THREE"`, `"FOUR"`, `"FIVE"`, `"SEVEN"` (study hours as text)
- **Class_ten_education**: `"CBSE"`, `"SEBA"`, `"OTHERS"`
- **twelve_education**: `"CBSE"`, `"AHSEC"`, `"OTHERS"`, `"NAN"`
- **medium**: `"ENGLISH"`, `"ASSAMESE"`, `"OTHERS"`
- **Class_ X_Percentage**: `"AVERAGE"`, `"GOOD"`, `"VG"` (Very Good), `"EXCELLENT"`, `"NAN"`
- **Class_XII_Percentage**: `"AVERAGE"`, `"GOOD"`, `"VG"`, `"EXCELLENT"`, `"NAN"`
- **Father_occupation**: `"BANK_OFFICIAL"`, `"BUSINESS"`, `"COLLEGE_TEACHER"`, `"CULTIVATOR"`, `"DOCTOR"`, `"ENGINEER"`, `"SCHOOL_TEACHER"`, `"OTHERS"`, `"NAN"`
- **Mother_occupation**: `"BANK_OFFICIAL"`, `"BUSINESS"`, `"COLLEGE_TEACHER"`, `"CULTIVATOR"`, `"DOCTOR"`, `"ENGINEER"`, `"HOUSE_WIFE"`, `"SCHOOL_TEACHER"`, `"OTHERS"`, `"NAN"`

**Important Notes:**
- All categorical values are in **UPPERCASE**
- Percentages are **categorical** (not numeric values)
- Time is represented as **text** (not integers)
- Snake_case field names (e.g., `class_x_percentage`) are automatically mapped to model column names

Invalid inputs will return a `422 Unprocessable Entity` or `400 Bad Request` with validation error details.

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
- Pull data and models from DVC: `dvc pull`
- Or run the training pipeline: `python src/pipeline/run_pipeline.py`
- Model artifacts are managed by DVC (not in git)

**Import errors:**
- Install all dependencies: `pip install -r src/api/requirements.txt`
- Ensure you're running from the project root

**Port already in use:**
- Change port: `uvicorn src.api.main:app --port 8001`

## 📝 License

Same as parent project (see root LICENSE file).
