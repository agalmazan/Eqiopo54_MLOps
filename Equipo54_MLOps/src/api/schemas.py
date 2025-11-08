"""
Pydantic schemas for request/response validation
Defines the structure of input and output data for the API
"""

from pydantic import BaseModel, Field, validator
from typing import Literal, Optional


class StudentFeatures(BaseModel):
    """
    Schema for student features input
    
    Attributes match the features expected by the trained model
    """
    gender: Literal["Male", "Female"] = Field(
        ...,
        description="Student's gender",
        example="Male"
    )
    
    caste: Literal["General", "OBC", "SC", "ST"] = Field(
        ...,
        description="Student's caste category",
        example="General"
    )
    
    coaching: Literal["Yes", "No"] = Field(
        ...,
        description="Whether student takes coaching classes",
        example="Yes"
    )
    
    time: int = Field(
        ...,
        ge=0,
        le=24,
        description="Study time in hours per day",
        example=5
    )
    
    class_ten_education: Literal["CBSE", "ICSE", "State Board", "Others"] = Field(
        ...,
        alias="Class_ten_education",
        description="Type of education board for Class 10",
        example="CBSE"
    )
    
    twelve_education: Literal["CBSE", "ICSE", "State Board", "Others"] = Field(
        ...,
        description="Type of education board for Class 12",
        example="CBSE"
    )
    
    medium: Literal["English", "Hindi", "Regional"] = Field(
        ...,
        description="Medium of instruction",
        example="English"
    )
    
    class_x_percentage: float = Field(
        ...,
        alias="Class_ X_Percentage",
        ge=0,
        le=100,
        description="Percentage obtained in Class 10 (0-100)",
        example=85.5
    )
    
    class_xii_percentage: float = Field(
        ...,
        alias="Class_XII_Percentage",
        ge=0,
        le=100,
        description="Percentage obtained in Class 12 (0-100)",
        example=78.2
    )
    
    father_occupation: Literal[
        "Farmer", "Government Officer", "Private Job", "Business", "Others"
    ] = Field(
        ...,
        alias="Father_occupation",
        description="Father's occupation",
        example="Government Officer"
    )
    
    mother_occupation: Literal[
        "Housewife", "Teacher", "Government Officer", "Private Job", "Business", "Others"
    ] = Field(
        ...,
        alias="Mother_occupation",
        description="Mother's occupation",
        example="Teacher"
    )

    class Config:
        populate_by_name = True
        schema_extra = {
            "example": {
                "gender": "Male",
                "caste": "General",
                "coaching": "Yes",
                "time": 5,
                "Class_ten_education": "CBSE",
                "twelve_education": "CBSE",
                "medium": "English",
                "Class_ X_Percentage": 85.5,
                "Class_XII_Percentage": 78.2,
                "Father_occupation": "Government Officer",
                "Mother_occupation": "Teacher"
            }
        }


class PredictionResponse(BaseModel):
    """
    Schema for prediction response
    """
    prediction: Literal["Average", "Good", "Very Good", "Excellent"] = Field(
        ...,
        description="Predicted performance category",
        example="Good"
    )
    
    probability: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence probability for the prediction (0-1)",
        example=0.85
    )
    
    model_version: str = Field(
        ...,
        description="Version of the model used for prediction",
        example="1"
    )

    class Config:
        schema_extra = {
            "example": {
                "prediction": "Good",
                "probability": 0.85,
                "model_version": "1"
            }
        }


class HealthResponse(BaseModel):
    """Schema for health check response"""
    status: str = Field(..., description="API health status", example="healthy")
    model_loaded: bool = Field(..., description="Whether model is loaded", example=True)
    model_version: Optional[str] = Field(None, description="Loaded model version", example="1")


class ErrorResponse(BaseModel):
    """Schema for error responses"""
    detail: str = Field(..., description="Error message", example="Model not found")
