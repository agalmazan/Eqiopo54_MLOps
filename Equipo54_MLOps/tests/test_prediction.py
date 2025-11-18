"""
Pruebas unitarias para el módulo de predicción
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Agregar src al path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.predict_model import StudentPerformancePredictor


class TestStudentPerformancePredictor:
    """Pruebas para la clase StudentPerformancePredictor"""
    
    @pytest.fixture
    def mock_model_artifacts(self, tmp_path):
        """Crea artefactos de modelo mock para testing"""
        # Crear un modelo simple
        model = DecisionTreeClassifier(random_state=42, max_depth=3)
        X_train = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
        y_train = np.array([0, 1, 0, 1])
        model.fit(X_train, y_train)
        
        # Guardar modelo
        model_path = tmp_path / "decision_tree_model.pkl"
        joblib.dump(model, model_path)
        
        # Crear encoders
        le_feature1 = LabelEncoder()
        le_feature1.fit(['SCI', 'COM'])
        
        le_feature2 = LabelEncoder()
        le_feature2.fit(['M', 'F'])
        
        le_target = LabelEncoder()
        le_target.fit(['Good', 'Excellent'])
        
        encoders_data = {
            'feature_encoders': {
                'Degree': le_feature1,
                'Gender': le_feature2
            },
            'target_encoder': le_target,
            'feature_names': ['Degree', 'Gender']
        }
        
        # Guardar encoders
        encoders_path = tmp_path / "label_encoders.pkl"
        joblib.dump(encoders_data, encoders_path)
        
        return tmp_path
    
    def test_predictor_initialization(self, mock_model_artifacts):
        """Verifica que el predictor se inicializa correctamente"""
        predictor = StudentPerformancePredictor(str(mock_model_artifacts))
        
        assert predictor.model is not None
        assert predictor.encoders_data is not None
    
    def test_predictor_loads_artifacts(self, mock_model_artifacts):
        """Verifica que se cargan todos los artefactos"""
        predictor = StudentPerformancePredictor(str(mock_model_artifacts))
        
        assert 'feature_encoders' in predictor.encoders_data
        assert 'target_encoder' in predictor.encoders_data
        assert 'feature_names' in predictor.encoders_data
    
    def test_predictor_missing_model_raises_error(self, tmp_path):
        """Verifica que se lanza error si falta el modelo"""
        with pytest.raises(FileNotFoundError):
            StudentPerformancePredictor(str(tmp_path))
    
    def test_preprocess_input_cleans_text(self, mock_model_artifacts):
        """Verifica que preprocess_input limpia el texto"""
        predictor = StudentPerformancePredictor(str(mock_model_artifacts))
        
        df = pd.DataFrame({
            'Degree': ['  sci  ', 'com'],
            'Gender': ['m', 'f']
        })
        
        result = predictor.preprocess_input(df)
        
        # Verificar que los valores son numéricos (codificados)
        assert result['Degree'].dtype in ['int32', 'int64']
        assert result['Gender'].dtype in ['int32', 'int64']
    
    def test_preprocess_raises_error_missing_features(self, mock_model_artifacts):
        """Verifica que se lanza error si faltan features"""
        predictor = StudentPerformancePredictor(str(mock_model_artifacts))
        
        df = pd.DataFrame({
            'Degree': ['SCI']
            # Falta Gender
        })
        
        with pytest.raises(ValueError, match="Faltan las siguientes features"):
            predictor.preprocess_input(df)
    
    def test_predict_returns_correct_format(self, mock_model_artifacts):
        """Verifica que predict retorna el formato correcto"""
        predictor = StudentPerformancePredictor(str(mock_model_artifacts))
        
        df = pd.DataFrame({
            'Degree': ['SCI', 'COM'],
            'Gender': ['M', 'F']
        })
        
        predictions, probabilities = predictor.predict(df)
        
        # Verificar formato
        assert len(predictions) == len(df)
        assert probabilities.shape[0] == len(df)
        assert probabilities.shape[1] == 2  # 2 clases
        
        # Verificar que las predicciones son strings (decodificadas)
        assert isinstance(predictions[0], str)
    
    def test_predict_with_details_includes_probabilities(self, mock_model_artifacts):
        """Verifica que predict_with_details incluye probabilidades"""
        predictor = StudentPerformancePredictor(str(mock_model_artifacts))
        
        df = pd.DataFrame({
            'Degree': ['SCI'],
            'Gender': ['M']
        })
        
        result = predictor.predict_with_details(df)
        
        # Verificar columnas
        assert 'Predicted_Performance' in result.columns
        assert 'Confidence' in result.columns
        
        # Verificar que hay columnas de probabilidad
        prob_cols = [col for col in result.columns if col.startswith('Prob_')]
        assert len(prob_cols) > 0
        
        # Verificar que la confianza está entre 0 y 1
        assert 0 <= result['Confidence'].values[0] <= 1


class TestPredictorEdgeCases:
    """Pruebas de casos extremos"""
    
    @pytest.fixture
    def predictor_setup(self, tmp_path):
        """Setup básico del predictor"""
        # Crear modelo y encoders simples
        model = DecisionTreeClassifier(random_state=42)
        X = np.array([[0], [1]])
        y = np.array([0, 1])
        model.fit(X, y)
        
        joblib.dump(model, tmp_path / "decision_tree_model.pkl")
        
        le = LabelEncoder()
        le.fit(['A', 'B'])
        
        le_target = LabelEncoder()
        le_target.fit(['Low', 'High'])
        
        joblib.dump({
            'feature_encoders': {'Feature1': le},
            'target_encoder': le_target,
            'feature_names': ['Feature1']
        }, tmp_path / "label_encoders.pkl")
        
        return tmp_path
    
    def test_single_prediction(self, predictor_setup):
        """Verifica predicción de una sola muestra"""
        predictor = StudentPerformancePredictor(str(predictor_setup))
        
        df = pd.DataFrame({'Feature1': ['A']})
        predictions, probabilities = predictor.predict(df)
        
        assert len(predictions) == 1
        assert probabilities.shape == (1, 2)
    
    def test_multiple_predictions(self, predictor_setup):
        """Verifica predicción de múltiples muestras"""
        predictor = StudentPerformancePredictor(str(predictor_setup))
        
        df = pd.DataFrame({'Feature1': ['A', 'B', 'A', 'B']})
        predictions, probabilities = predictor.predict(df)
        
        assert len(predictions) == 4
        assert probabilities.shape == (4, 2)
