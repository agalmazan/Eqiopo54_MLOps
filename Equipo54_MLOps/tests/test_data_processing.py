"""
Pruebas unitarias para el módulo de procesamiento de datos
"""
import pytest
import pandas as pd
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.make_dataset import clean_text, process_data


class TestCleanText:
    """Pruebas para la función clean_text"""
    
    def test_clean_text_basic(self):
        """Verifica que el texto se limpia correctamente"""
        assert clean_text("  hello  ") == "HELLO"
        assert clean_text("World") == "WORLD"
    
    def test_clean_text_uppercase(self):
        """Verifica conversión a mayúsculas"""
        assert clean_text("lowercase") == "LOWERCASE"
        assert clean_text("MiXeD CaSe") == "MIXED CASE"
    
    def test_clean_text_strips_spaces(self):
        """Verifica que se eliminan espacios extra"""
        assert clean_text("  spaces  ") == "SPACES"
        assert clean_text("\ttabs\t") == "TABS"
    
    def test_clean_text_nan_handling(self):
        """Verifica manejo de valores NaN"""
        result = clean_text(pd.NA)
        assert pd.isna(result)
        
        result = clean_text(None)
        assert pd.isna(result)
    
    def test_clean_text_numbers(self):
        """Verifica que los números se convierten a string"""
        assert clean_text(123) == "123"
        assert clean_text(45.67) == "45.67"


class TestProcessData:
    """Pruebas para la función process_data"""
    
    @pytest.fixture
    def sample_data(self, tmp_path):
        """Crea un dataset de ejemplo para testing"""
        data = {
            'Degree': ['SCI', 'COM', 'SCI'],
            'Gender': ['M', 'F', 'M'],
            'Coaching': ['NO', 'YES', 'NO'],
            'Time': ['ONE', 'TWO', 'ONE'],
            'College_Tier': ['1', '2', '1'],
            'Class_X_Board': ['C', 'S', 'C'],
            'Class_X_Percentage': ['EXCELLENT', 'GOOD', 'VG'],
            'Class_XI_Subject_Taken': ['S', 'C', 'S'],
            'Class_XII_Board': ['C', 'S', 'C'],
            'Class_XII_Percentage': ['EXCELLENT', 'GOOD', 'AVERAGE'],
            'Entrance_Exam': ['2', '1', '2'],
            'Performance': ['EXCELLENT', 'GOOD', 'VG']
        }
        
        df = pd.DataFrame(data)
        input_path = tmp_path / "test_input.csv"
        df.to_csv(input_path, index=False)
        
        return input_path
    
    def test_process_data_returns_dataframe(self, sample_data, tmp_path):
        """Verifica que process_data retorna un DataFrame"""
        output_path = tmp_path / "test_output.csv"
        result = process_data(str(sample_data), str(output_path))
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    def test_process_data_creates_output_file(self, sample_data, tmp_path):
        """Verifica que se crea el archivo de salida"""
        output_path = tmp_path / "test_output.csv"
        process_data(str(sample_data), str(output_path))
        
        assert output_path.exists()
    
    def test_process_data_standardizes_performance(self, sample_data, tmp_path):
        """Verifica que Performance se estandariza correctamente"""
        output_path = tmp_path / "test_output.csv"
        result = process_data(str(sample_data), str(output_path))
        
        # Verificar que se aplicó el mapeo
        assert 'Excellent' in result['Performance'].values
        assert 'EXCELLENT' not in result['Performance'].values
    
    def test_process_data_removes_nulls(self, tmp_path):
        """Verifica que se eliminan filas con nulos"""
        # Crear datos con nulos
        data = {
            'Degree': ['SCI', None, 'COM'],
            'Gender': ['M', 'F', 'M'],
            'Performance': ['EXCELLENT', 'GOOD', 'VG']
        }
        df = pd.DataFrame(data)
        input_path = tmp_path / "test_with_nulls.csv"
        df.to_csv(input_path, index=False)
        
        output_path = tmp_path / "test_output.csv"
        result = process_data(str(input_path), str(output_path))
        
        # Verificar que no hay nulos
        assert result.isnull().sum().sum() == 0
        # Verificar que se eliminó la fila con nulo
        assert len(result) < len(df)
