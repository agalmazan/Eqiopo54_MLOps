"""
FastAPI application for Student Performance Prediction
Exposes REST API endpoints for model predictions
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import logging
from contextlib import asynccontextmanager

from .schemas import (
    StudentFeatures, 
    PredictionResponse, 
    HealthResponse, 
    ErrorResponse
)
from .model import ModelPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model predictor instance
predictor: ModelPredictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for the FastAPI application
    Loads model on startup, cleans up on shutdown
    """
    # Startup: Load model
    global predictor
    logger.info("🚀 Starting FastAPI application...")
    
    # Determine model path (prioritize latest, fallback to root)
    project_root = Path(__file__).resolve().parents[3]
    model_dir = project_root / "models" / "latest"
    
    model_path = model_dir / "decision_tree_model.pkl"
    encoders_path = model_dir / "label_encoders.pkl"
    
    if not model_path.exists():
        logger.warning(f"Model not found at {model_path}, trying root...")
        model_path = project_root / "models" / "decision_tree_model.pkl"
        encoders_path = project_root / "models" / "label_encoders.pkl"
    
    predictor = ModelPredictor()
    
    if model_path.exists():
        success = predictor.load_model(model_path, encoders_path)
        if success:
            logger.info(f"✅ Model loaded successfully from {model_path}")
        else:
            logger.error("❌ Failed to load model")
    else:
        logger.error(f"❌ Model file not found: {model_path}")
    
    yield
    
    # Shutdown: Cleanup
    logger.info("🛑 Shutting down FastAPI application...")


# Create FastAPI app
app = FastAPI(
    title="Student Performance Prediction API",
    description="""
    REST API for predicting student performance based on various features.
    
    ## Features
    * **Predict**: Get performance prediction for a student
    * **Health Check**: Verify API and model status
    
    ## Model Information
    * **Model Type**: Decision Tree Classifier
    * **Target**: Student Performance (Average, Good, Very Good, Excellent)
    * **Features**: Demographics, academic scores, family background, study habits
    
    ## Usage
    Send a POST request to `/predict` with student features to get a prediction.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware (allow all origins for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/",
    response_model=dict,
    tags=["Root"],
    summary="Root endpoint"
)
async def root():
    """
    Root endpoint - provides API information
    """
    return {
        "message": "Student Performance Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint"
)
async def health_check():
    """
    Check API health and model status
    
    Returns:
        HealthResponse with status and model information
    """
    return HealthResponse(
        status="healthy" if predictor and predictor.is_loaded() else "unhealthy",
        model_loaded=predictor.is_loaded() if predictor else False,
        model_version=predictor.get_version() if predictor else None
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        200: {
            "description": "Successful prediction",
            "model": PredictionResponse
        },
        400: {
            "description": "Invalid input data",
            "model": ErrorResponse
        },
        500: {
            "description": "Model not loaded or prediction error",
            "model": ErrorResponse
        }
    },
    tags=["Prediction"],
    summary="Predict student performance"
)
async def predict(features: StudentFeatures):
    """
    Predict student performance based on input features
    
    Args:
        features: StudentFeatures object with all required fields
        
    Returns:
        PredictionResponse with prediction and confidence
        
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    # Check if model is loaded
    if not predictor or not predictor.is_loaded():
        logger.error("Prediction request received but model not loaded")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model not loaded. Please contact administrator."
        )
    
    try:
        # Convert Pydantic model to dict
        feature_dict = features.dict()
        
        logger.info(f"Prediction request received: {feature_dict}")
        
        # Make prediction
        prediction, probability = predictor.predict(feature_dict)
        
        # Create response
        response = PredictionResponse(
            prediction=prediction,
            probability=probability,
            model_version=predictor.get_version()
        )
        
        logger.info(f"Prediction successful: {prediction} (confidence: {probability})")
        
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get(
    "/model/info",
    tags=["Model"],
    summary="Get model information"
)
async def model_info():
    """
    Get information about the loaded model
    
    Returns:
        Dictionary with model metadata
    """
    if not predictor or not predictor.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model not loaded"
        )
    
    return {
        "model_version": predictor.get_version(),
        "model_type": "DecisionTreeClassifier",
        "features": predictor.feature_names,
        "target_classes": ["Average", "Good", "Very Good", "Excellent"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
