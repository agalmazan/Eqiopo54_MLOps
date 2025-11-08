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
    
    mathematics_marks: int = Field(
        ...,
        ge=0,
        le=100,
        description="Marks obtained in Mathematics (0-100)",
        example=85
    )
    
    english_marks: int = Field(
        ...,
        ge=0,
        le=100,
        description="Marks obtained in English (0-100)",
        example=78
    )
    
    science_marks: int = Field(
        ...,
        ge=0,
        le=100,
        description="Marks obtained in Science (0-100)",
        example=82
    )
    
    father_occupation: Literal[
        "Farmer", "Government Officer", "Private Job", "Business", "Others"
    ] = Field(
        ...,
        description="Father's occupation",
        example="Government Officer"
    )
    
    mother_occupation: Literal[
        "Housewife", "Teacher", "Government Officer", "Private Job", "Business", "Others"
    ] = Field(
        ...,
        description="Mother's occupation",
        example="Teacher"
    )
    
    number_of_siblings: int = Field(
        ...,
        ge=0,
        le=10,
        description="Number of siblings",
        example=2
    )
    
    boarding: Literal["Yes", "No"] = Field(
        ...,
        description="Whether student is a boarder",
        example="No"
    )
    
    distance_from_home: Literal["Near", "Far", "Very Far"] = Field(
        ...,
        description="Distance from home to school",
        example="Near"
    )
    
    time: int = Field(
        ...,
        ge=0,
        le=24,
        description="Study time in hours per day",
        example=5
    )
    
    coaching: Literal["Yes", "No"] = Field(
        ...,
        description="Whether student takes coaching classes",
        example="Yes"
    )

    class Config:
        schema_extra = {
            "example": {
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
