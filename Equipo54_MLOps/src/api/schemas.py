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
    gender: Literal["MALE", "FEMALE", "NAN"] = Field(
        ...,
        description="Student's gender",
        example="MALE"
    )
    
    caste: Literal["GENERAL", "OBC", "SC", "ST"] = Field(
        ...,
        description="Student's caste category",
        example="GENERAL"
    )
    
    coaching: Literal["NO", "OA", "WA"] = Field(
        ...,
        description="Coaching type (NO=None, OA=Online/Offline, WA=Weekend)",
        example="OA"
    )
    
    time: Literal["ONE", "TWO", "THREE", "FOUR", "FIVE", "SEVEN"] = Field(
        ...,
        description="Study time in hours per day (as text)",
        example="FIVE"
    )
    
    class_ten_education: Literal["CBSE", "SEBA", "OTHERS"] = Field(
        ...,
        alias="Class_ten_education",
        description="Type of education board for Class 10",
        example="CBSE"
    )
    
    twelve_education: Literal["CBSE", "AHSEC", "OTHERS", "NAN"] = Field(
        ...,
        description="Type of education board for Class 12",
        example="CBSE"
    )
    
    medium: Literal["ENGLISH", "ASSAMESE", "OTHERS"] = Field(
        ...,
        description="Medium of instruction",
        example="ENGLISH"
    )
    
    class_x_percentage: Literal["AVERAGE", "EXCELLENT", "GOOD", "VG", "NAN"] = Field(
        ...,
        alias="Class_ X_Percentage",
        description="Performance category in Class 10",
        example="EXCELLENT"
    )
    
    class_xii_percentage: Literal["AVERAGE", "EXCELLENT", "GOOD", "VG", "NAN"] = Field(
        ...,
        alias="Class_XII_Percentage",
        description="Performance category in Class 12",
        example="GOOD"
    )
    
    father_occupation: Literal[
        "BANK_OFFICIAL", "BUSINESS", "COLLEGE_TEACHER", "CULTIVATOR", 
        "DOCTOR", "ENGINEER", "SCHOOL_TEACHER", "OTHERS", "NAN"
    ] = Field(
        ...,
        alias="Father_occupation",
        description="Father's occupation",
        example="BANK_OFFICIAL"
    )
    
    mother_occupation: Literal[
        "BANK_OFFICIAL", "BUSINESS", "COLLEGE_TEACHER", "CULTIVATOR", 
        "DOCTOR", "ENGINEER", "HOUSE_WIFE", "SCHOOL_TEACHER", "OTHERS", "NAN"
    ] = Field(
        ...,
        alias="Mother_occupation",
        description="Mother's occupation",
        example="HOUSE_WIFE"
    )

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
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
        protected_namespaces = ()
        json_schema_extra = {
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
    
    class Config:
        protected_namespaces = ()


class ErrorResponse(BaseModel):
    """Schema for error responses"""
    detail: str = Field(..., description="Error message", example="Model not found")
