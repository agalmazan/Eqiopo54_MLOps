"""
Pruebas unitarias para el módulo de construcción de features
"""
import pytest
import pandas as pd
import sys
from pathlib import Path
import joblib

# Agregar src al path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features.build_features import build_features


class TestBuildFeatures:
    """Pruebas para la función build_features"""
    
    @pytest.fixture
    def sample_clean_data(self, tmp_path):
        """Crea datos limpios de ejemplo"""
        data = {
            'Degree': ['SCI', 'COM', 'SCI'],
            'Gender': ['M', 'F', 'M'],
            'Coaching': ['NO', 'YES', 'NO'],
            'Performance': ['Excellent', 'Good', 'Very Good']
        }
        
        df = pd.DataFrame(data)
        input_path = tmp_path / "clean_data.csv"
        df.to_csv(input_path, index=False)
        
        return input_path
    
    def test_build_features_returns_dataframe(self, sample_clean_data, tmp_path):
        """Verifica que retorna un DataFrame"""
        output_path = tmp_path / "features.csv"
        encoders_path = tmp_path / "encoders.pkl"
        
        result = build_features(
            str(sample_clean_data),
            str(output_path),
            str(encoders_path)
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    def test_build_features_creates_output_files(self, sample_clean_data, tmp_path):
        """Verifica que se crean los archivos de salida"""
        output_path = tmp_path / "features.csv"
        encoders_path = tmp_path / "encoders.pkl"
        
        build_features(
            str(sample_clean_data),
            str(output_path),
            str(encoders_path)
        )
        
        assert output_path.exists()
        assert encoders_path.exists()
    
    def test_build_features_encodes_correctly(self, sample_clean_data, tmp_path):
        """Verifica que las features se codifican correctamente"""
        output_path = tmp_path / "features.csv"
        encoders_path = tmp_path / "encoders.pkl"
        
        result = build_features(
            str(sample_clean_data),
            str(output_path),
            str(encoders_path)
        )
        
        # Verificar que todas las columnas son numéricas
        assert result.select_dtypes(include=['number']).shape[1] == result.shape[1]
        
        # Verificar que los valores son enteros (códigos del encoder)
        for col in result.columns:
            assert result[col].dtype in ['int32', 'int64']
    
    def test_build_features_saves_encoders(self, sample_clean_data, tmp_path):
        """Verifica que los encoders se guardan correctamente"""
        output_path = tmp_path / "features.csv"
        encoders_path = tmp_path / "encoders.pkl"
        
        build_features(
            str(sample_clean_data),
            str(output_path),
            str(encoders_path)
        )
        
        # Cargar encoders
        encoders_data = joblib.load(str(encoders_path))
        
        # Verificar estructura
        assert 'feature_encoders' in encoders_data
        assert 'target_encoder' in encoders_data
        assert 'feature_names' in encoders_data
        
        # Verificar que se guardaron encoders para cada feature
        assert len(encoders_data['feature_encoders']) > 0
        
        # Verificar que feature_names es una lista
        assert isinstance(encoders_data['feature_names'], list)
    
    def test_build_features_preserves_sample_count(self, sample_clean_data, tmp_path):
        """Verifica que no se pierden muestras en el proceso"""
        output_path = tmp_path / "features.csv"
        encoders_path = tmp_path / "encoders.pkl"
        
        # Contar filas originales
        original_df = pd.read_csv(sample_clean_data)
        original_count = len(original_df)
        
        result = build_features(
            str(sample_clean_data),
            str(output_path),
            str(encoders_path)
        )
        
        # Verificar que se mantiene el número de filas
        assert len(result) == original_count
