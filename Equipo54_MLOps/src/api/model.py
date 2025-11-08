"""
Model loader and predictor
Handles loading the trained model and making predictions
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Loads and manages the trained model for predictions
    """
    
    def __init__(self, model_path: Optional[Path] = None, encoders_path: Optional[Path] = None):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to the trained model file (.pkl)
            encoders_path: Path to the label encoders file (.pkl)
        """
        self.model = None
        self.encoders = None
        self.model_version = None
        self.feature_names = [
            'Gender', 'Caste', 'coaching', 'time', 'Class_ten_education',
            'twelve_education', 'medium', 'Class_ X_Percentage',
            'Class_XII_Percentage', 'Father_occupation', 'Mother_occupation'
        ]
        
        if model_path:
            self.load_model(model_path, encoders_path)
    
    def load_model(self, model_path: Path, encoders_path: Optional[Path] = None) -> bool:
        """
        Load the trained model and encoders
        
        Args:
            model_path: Path to model file
            encoders_path: Path to encoders file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load model
            logger.info(f"Loading model from: {model_path}")
            self.model = joblib.load(model_path)
            
            # Load encoders if provided
            if encoders_path and encoders_path.exists():
                logger.info(f"Loading encoders from: {encoders_path}")
                self.encoders = joblib.load(encoders_path)
            else:
                logger.warning("No encoders file provided or found")
                self.encoders = None
            
            # Extract version from path (e.g., models/latest or models:/model_name/1)
            self.model_version = self._extract_version(model_path)
            
            logger.info(f"Model loaded successfully (version: {self.model_version})")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def _extract_version(self, model_path: Path) -> str:
        """Extract version from model path"""
        path_str = str(model_path)
        if "latest" in path_str:
            return "latest"
        # Try to extract version number from path
        parts = path_str.split('/')
        for part in reversed(parts):
            if part.isdigit():
                return part
        return "unknown"
    
    def _preprocess_input(self, features: Dict) -> pd.DataFrame:
        """
        Preprocess input features to match model expectations
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            DataFrame ready for prediction
        """
        # Map API field names to model feature names
        feature_mapping = {
            'gender': 'Gender',
            'caste': 'Caste',
            'coaching': 'coaching',
            'time': 'time',
            'Class_ten_education': 'Class_ten_education',
            'twelve_education': 'twelve_education',
            'medium': 'medium',
            'Class_ X_Percentage': 'Class_ X_Percentage',
            'Class_XII_Percentage': 'Class_XII_Percentage',
            'Father_occupation': 'Father_occupation',
            'Mother_occupation': 'Mother_occupation'
        }
        
        # Create DataFrame with mapped feature names
        mapped_features = {
            feature_mapping[k]: v for k, v in features.items() 
            if k in feature_mapping
        }
        
        df = pd.DataFrame([mapped_features])
        
        # Apply label encoding if encoders are available
        if self.encoders:
            for col, encoder in self.encoders.items():
                if col in df.columns and col != 'Performance':
                    try:
                        df[col] = encoder.transform(df[col])
                    except ValueError as e:
                        logger.warning(f"Could not encode {col}: {e}")
        
        # Ensure correct column order
        df = df[self.feature_names]
        
        return df
    
    def predict(self, features: Dict) -> Tuple[str, float]:
        """
        Make a prediction for given features
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Tuple of (prediction, probability)
            
        Raises:
            ValueError: If model is not loaded
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess input
        X = self._preprocess_input(features)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        # Get probability if available
        try:
            probabilities = self.model.predict_proba(X)[0]
            max_prob = float(np.max(probabilities))
        except AttributeError:
            # Model doesn't support predict_proba
            max_prob = None
        
        # Decode prediction if encoders available
        if self.encoders and 'Performance' in self.encoders:
            try:
                prediction = self.encoders['Performance'].inverse_transform([prediction])[0]
            except Exception as e:
                logger.warning(f"Could not decode prediction: {e}")
        
        return str(prediction), max_prob
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def get_version(self) -> Optional[str]:
        """Get model version"""
        return self.model_version
