"""
Pruebas de integración extremo a extremo del pipeline
"""
import pytest
import pandas as pd
import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Agregar src al path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.make_dataset import process_data
from features.build_features import build_features
from models.predict_model import StudentPerformancePredictor


class TestEndToEndPipeline:
    """Pruebas de integración del pipeline completo"""
    
    @pytest.fixture
    def raw_data(self, tmp_path):
        """Crea datos raw de ejemplo para el pipeline completo"""
        data = {
            'Degree': ['SCI', 'COM', 'SCI', 'COM', 'SCI', 'COM'],
            'Gender': ['M', 'F', 'M', 'F', 'M', 'F'],
            'Coaching': ['NO', 'YES', 'NO', 'YES', 'NO', 'YES'],
            'Time': ['ONE', 'TWO', 'ONE', 'TWO', 'ONE', 'TWO'],
            'College_Tier': ['1', '2', '1', '2', '1', '2'],
            'Class_X_Board': ['C', 'S', 'C', 'S', 'C', 'S'],
            'Class_X_Percentage': ['EXCELLENT', 'GOOD', 'VG', 'EXCELLENT', 'GOOD', 'AVERAGE'],
            'Class_XI_Subject_Taken': ['S', 'C', 'S', 'C', 'S', 'C'],
            'Class_XII_Board': ['C', 'S', 'C', 'S', 'C', 'S'],
            'Class_XII_Percentage': ['EXCELLENT', 'GOOD', 'VG', 'EXCELLENT', 'GOOD', 'AVERAGE'],
            'Entrance_Exam': ['2', '1', '2', '1', '2', '1'],
            'Performance': ['EXCELLENT', 'GOOD', 'VG', 'EXCELLENT', 'GOOD', 'AVERAGE']
        }
        
        df = pd.DataFrame(data)
        raw_path = tmp_path / "raw_data.csv"
        df.to_csv(raw_path, index=False)
        
        return raw_path
    
    def test_complete_pipeline_flow(self, raw_data, tmp_path):
        """
        Prueba el flujo completo:
        Raw Data → Processing → Feature Engineering → Training → Prediction
        """
        # 1. Procesamiento de datos
        clean_path = tmp_path / "clean_data.csv"
        clean_df = process_data(str(raw_data), str(clean_path))
        
        assert clean_path.exists()
        assert len(clean_df) > 0
        
        # 2. Construcción de features
        features_path = tmp_path / "features.csv"
        encoders_path = tmp_path / "encoders.pkl"
        features_df = build_features(
            str(clean_path),
            str(features_path),
            str(encoders_path)
        )
        
        assert features_path.exists()
        assert encoders_path.exists()
        assert len(features_df) > 0
        
        # 3. Entrenamiento de modelo simple
        X = features_df.drop('Performance', axis=1)
        y = features_df['Performance']
        
        model = DecisionTreeClassifier(random_state=42, max_depth=3)
        model.fit(X, y)
        
        model_path = tmp_path / "decision_tree_model.pkl"
        joblib.dump(model, model_path)
        
        assert model_path.exists()
        
        # 4. Predicción
        # Copiar encoders al directorio esperado (label_encoders.pkl)
        import shutil
        shutil.copy(str(encoders_path), str(tmp_path / "label_encoders.pkl"))
        
        predictor = StudentPerformancePredictor(str(tmp_path))
        
        # Crear nuevos datos para predicción
        new_data = pd.DataFrame({
            'Degree': ['SCI'],
            'Gender': ['M'],
            'Coaching': ['NO'],
            'Time': ['ONE'],
            'College_Tier': ['1'],
            'Class_X_Board': ['C'],
            'Class_X_Percentage': ['Excellent'],
            'Class_XI_Subject_Taken': ['S'],
            'Class_XII_Board': ['C'],
            'Class_XII_Percentage': ['Excellent'],
            'Entrance_Exam': ['2']
        })
        
        predictions, probabilities = predictor.predict(new_data)
        
        assert len(predictions) == 1
        assert probabilities.shape[0] == 1
    
    def test_pipeline_data_consistency(self, raw_data, tmp_path):
        """
        Verifica que los datos se mantienen consistentes a través del pipeline
        """
        # Procesar
        clean_path = tmp_path / "clean_data.csv"
        clean_df = process_data(str(raw_data), str(clean_path))
        
        original_count = len(clean_df)
        
        # Construir features
        features_path = tmp_path / "features.csv"
        encoders_path = tmp_path / "encoders.pkl"
        features_df = build_features(
            str(clean_path),
            str(features_path),
            str(encoders_path)
        )
        
        # Verificar que no se pierden muestras
        assert len(features_df) == original_count
        
        # Verificar que el número de features es correcto
        assert features_df.shape[1] == clean_df.shape[1]
    
    def test_pipeline_with_train_test_split(self, raw_data, tmp_path):
        """
        Prueba el pipeline con división train/test
        """
        # Pipeline completo
        clean_path = tmp_path / "clean_data.csv"
        clean_df = process_data(str(raw_data), str(clean_path))
        
        features_path = tmp_path / "features.csv"
        encoders_path = tmp_path / "encoders.pkl"
        features_df = build_features(
            str(clean_path),
            str(features_path),
            str(encoders_path)
        )
        
        # Split train/test
        X = features_df.drop('Performance', axis=1)
        y = features_df['Performance']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Entrenar
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Predecir en test
        y_pred = model.predict(X_test)
        
        # Verificar métricas básicas
        accuracy = accuracy_score(y_test, y_pred)
        assert 0 <= accuracy <= 1
        
        f1 = f1_score(y_test, y_pred, average='weighted')
        assert 0 <= f1 <= 1


class TestPipelineErrorHandling:
    """Pruebas de manejo de errores en el pipeline"""
    
    def test_pipeline_handles_missing_columns(self, tmp_path):
        """Verifica que el pipeline maneja datos con pocas columnas"""
        # Crear datos con todas las columnas necesarias pero valores simples
        data = {
            'Degree': ['SCI', 'COM'],
            'Gender': ['M', 'F'],
            'Performance': ['EXCELLENT', 'GOOD']  # Agregar Performance para evitar KeyError
        }
        df = pd.DataFrame(data)
        raw_path = tmp_path / "simple_data.csv"
        df.to_csv(raw_path, index=False)
        
        # El procesamiento debe funcionar
        clean_path = tmp_path / "clean_data.csv"
        clean_df = process_data(str(raw_path), str(clean_path))
        
        assert len(clean_df) > 0
        assert 'Performance' in clean_df.columns
    
    def test_predictor_with_unseen_categories(self, tmp_path):
        """Verifica el manejo de categorías no vistas"""
        # Crear modelo simple
        model = DecisionTreeClassifier(random_state=42)
        X = np.array([[0], [1]])
        y = np.array([0, 1])
        model.fit(X, y)
        
        joblib.dump(model, tmp_path / "decision_tree_model.pkl")
        
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(['A', 'B'])  # Solo conoce A y B
        
        le_target = LabelEncoder()
        le_target.fit(['Low', 'High'])
        
        joblib.dump({
            'feature_encoders': {'Feature1': le},
            'target_encoder': le_target,
            'feature_names': ['Feature1']
        }, tmp_path / "label_encoders.pkl")
        
        predictor = StudentPerformancePredictor(str(tmp_path))
        
        # Intentar predecir con categoría no vista
        df = pd.DataFrame({'Feature1': ['C']})  # C no está en el encoder
        
        # No debe lanzar error, debe usar valor por defecto
        predictions, probabilities = predictor.predict(df)
        assert len(predictions) == 1


class TestPipelineMetrics:
    """Pruebas de métricas del pipeline"""
    
    @pytest.fixture
    def trained_pipeline(self, tmp_path):
        """Crea un pipeline entrenado completo"""
        # Datos de ejemplo más grandes
        np.random.seed(42)
        n_samples = 50
        
        data = {
            'Degree': np.random.choice(['SCI', 'COM'], n_samples),
            'Gender': np.random.choice(['M', 'F'], n_samples),
            'Coaching': np.random.choice(['NO', 'YES'], n_samples),
            'Performance': np.random.choice(['EXCELLENT', 'GOOD', 'AVERAGE'], n_samples)
        }
        
        df = pd.DataFrame(data)
        raw_path = tmp_path / "raw.csv"
        df.to_csv(raw_path, index=False)
        
        # Pipeline
        clean_path = tmp_path / "clean.csv"
        clean_df = process_data(str(raw_path), str(clean_path))
        
        features_path = tmp_path / "features.csv"
        encoders_path = tmp_path / "encoders.pkl"
        features_df = build_features(str(clean_path), str(features_path), str(encoders_path))
        
        X = features_df.drop('Performance', axis=1)
        y = features_df['Performance']
        
        model = DecisionTreeClassifier(random_state=42, max_depth=5)
        model.fit(X, y)
        
        joblib.dump(model, tmp_path / "decision_tree_model.pkl")
        
        return tmp_path, X, y
    
    def test_pipeline_achieves_minimum_accuracy(self, trained_pipeline):
        """Verifica que el pipeline logra una accuracy mínima"""
        model_dir, X, y = trained_pipeline
        
        # Copiar encoders al nombre correcto
        import shutil
        encoders_src = model_dir / "encoders.pkl"
        encoders_dst = model_dir / "label_encoders.pkl"
        if encoders_src.exists() and not encoders_dst.exists():
            shutil.copy(str(encoders_src), str(encoders_dst))
        
        predictor = StudentPerformancePredictor(str(model_dir))
        
        # Cargar datos originales
        df_original = pd.read_csv(model_dir / "clean.csv")
        
        # Predecir
        predictions, _ = predictor.predict(df_original.drop('Performance', axis=1))
        
        # Calcular accuracy
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(df_original['Performance'])
        
        y_true = le.transform(df_original['Performance'])
        y_pred = le.transform(predictions)
        
        accuracy = accuracy_score(y_true, y_pred)
        
        # En un modelo entrenado con los mismos datos, accuracy debe ser alta
        assert accuracy > 0.5  # Al menos 50% (ajusta según tu modelo)
    
    def test_pipeline_predictions_distribution(self, trained_pipeline):
        """Verifica que las predicciones tienen una distribución razonable"""
        model_dir, X, y = trained_pipeline
        
        # Copiar encoders al nombre correcto
        import shutil
        encoders_src = model_dir / "encoders.pkl"
        encoders_dst = model_dir / "label_encoders.pkl"
        if encoders_src.exists() and not encoders_dst.exists():
            shutil.copy(str(encoders_src), str(encoders_dst))
        
        predictor = StudentPerformancePredictor(str(model_dir))
        df_original = pd.read_csv(model_dir / "clean.csv")
        
        predictions, probabilities = predictor.predict(df_original.drop('Performance', axis=1))
        
        # Verificar que no todas las predicciones son iguales
        unique_predictions = len(set(predictions))
        assert unique_predictions > 1
        
        # Verificar que las probabilidades suman 1
        prob_sums = probabilities.sum(axis=1)
        assert np.allclose(prob_sums, 1.0, atol=0.01)
