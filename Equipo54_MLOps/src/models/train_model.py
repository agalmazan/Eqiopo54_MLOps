"""
Script para entrenar modelo de árbol de decisión

Uso:
    python src/models/train_model.py data/processed/student_features.csv models/
"""

import os
import json
import logging
import argparse
import joblib
import pandas as pd
import mlflow, mlflow.sklearn
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import dvc.api

# ---------- utilidades nuevas ----------
from pathlib import Path
import subprocess
import yaml

def _flatten(d, parent="", sep="."):
    out = {}
    for k, v in d.items():
        kk = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            out.update(_flatten(v, kk, sep))
        else:
            out[kk] = v
    return out

def _git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"

try:
    from mlflow.models.signature import infer_signature
except Exception:
    infer_signature = None
# ---------------------------------------

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "student-performance"))

def load_features(data_path):
    """
    Carga features ya procesadas y codificadas
    """
    logger.info(f"Cargando features desde: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Features cargadas: {df.shape[0]} filas, {df.shape[1]} columnas")

    # Separar features y target
    X = df.drop('Performance', axis=1)
    y = df['Performance']

    logger.info(f"X: {X.shape}, y: {y.shape}")
    logger.info(f"Features: {list(X.columns)}")
    return X, y

def train_model(X_train, y_train, params):
    """
    Entrena el modelo de árbol de decisión
    """
    optimize = params['train']['optimize']

    if optimize:
        logger.info("🔍 Optimizando hiperparámetros con GridSearchCV...")

        param_grid = params['model']['param_grid']
        gs_cfg = params['model']['grid_search']

        grid_search = GridSearchCV(
            DecisionTreeClassifier(
                random_state=params['train']['random_state'],
                class_weight=params['model']['class_weight']
            ),
            param_grid,
            cv=gs_cfg['cv'],
            scoring=gs_cfg['scoring'],
            n_jobs=gs_cfg['n_jobs'],
            verbose=gs_cfg['verbose']
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"✅ Mejor CV Score: {grid_search.best_score_:.4f}")
        logger.info(f"✅ Mejores parámetros: {grid_search.best_params_}")

        # Devolvemos también cv_results_ por si queremos subirlo
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_, grid_search.cv_results_

    else:
        logger.info("Entrenando modelo con parámetros por defecto...")

        mp = params['model']
        model = DecisionTreeClassifier(
            random_state=params['train']['random_state'],
            max_depth=mp['max_depth'],
            min_samples_split=mp['min_samples_split'],
            min_samples_leaf=mp['min_samples_leaf'],
            criterion=mp['criterion'],
            class_weight=mp['class_weight']
        )

        model.fit(X_train, y_train)
        return model, None, None, None

def save_model_and_splits(model, X_train, X_test, y_train, y_test,
                          output_dir, best_params=None, cv_score=None):
    """
    Guarda el modelo y los splits de datos (para DVC/evaluate)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Guardar modelo
    model_path = os.path.join(output_dir, 'decision_tree_model.pkl')
    joblib.dump(model, model_path)
    logger.info(f"✅ Modelo guardado en: {model_path}")

    # Guardar splits (para evaluate_model.py)
    splits_path = os.path.join(output_dir, 'train_test_split.pkl')
    joblib.dump({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }, splits_path)
    logger.info(f"✅ Train/test splits guardados en: {splits_path}")

    # Guardar parámetros del modelo (si hubo búsqueda)
    params_path = None
    if best_params is not None:
        params_path = os.path.join(output_dir, 'model_params.pkl')
        joblib.dump({
            'best_params': best_params,
            'cv_score': cv_score
        }, params_path)
        logger.info(f"✅ Parámetros guardados en: {params_path}")

    return model_path, splits_path, params_path

def main():
    """Función principal"""

    # 1) Parámetros desde DVC
    params = dvc.api.params_show()
    logger.info("📊 Parámetros DVC cargados")

    parser = argparse.ArgumentParser(description='Entrenar modelo de árbol de decisión')
    parser.add_argument('data_path', type=str, help='Ruta del archivo CSV con features procesadas')
    parser.add_argument('output_dir', type=str, help='Directorio donde guardar el modelo')
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        logger.error(f"El archivo {args.data_path} no existe")
        return

    logger.info("=" * 60)
    logger.info("🚀 INICIANDO ENTRENAMIENTO DEL MODELO")
    logger.info("=" * 60)

    # 2) Cargar features
    X, y = load_features(args.data_path)

    # 3) Split train/test desde params
    test_size = params['train']['test_size']
    random_state = params['train']['random_state']
    logger.info(f"Dividiendo datos (test_size={test_size}, random_state={random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # 4) INTEGRACIÓN CON MLFLOW
    mlflow.set_experiment("student-performance")  # mismo nombre local/prod

    with mlflow.start_run(run_name="train"):
        # 4.1 Tags y params
        mlflow.set_tag("pipeline_stage", "train")
        mlflow.set_tag("git_rev", _git_rev())
        mlflow.set_tag("timestamp", datetime.now().isoformat())

        # Log params (aplana para tener llaves tipo train.test_size, model.max_depth, etc.)
        flat_params = _flatten(params)
        # Si no quieres enviar todo, filtra:
        keep = {k: v for k, v in flat_params.items() if k.startswith(("train.", "model."))}
        if keep:
            mlflow.log_params(keep)

        # 4.2 Entrenamiento
        model, best_params, cv_score, cv_results = train_model(X_train, y_train, params=params)

        # 4.3 Guardar artefactos para DVC/evaluate
        model_path, splits_path, model_params_path = save_model_and_splits(
            model, X_train, X_test, y_train, y_test, args.output_dir, best_params, cv_score
        )

        # 4.4 Log de “metadata” útil como artefactos
        # - columnas (esquema simple)
        cols_file = Path(args.output_dir) / "feature_columns.json"
        cols_file.write_text(json.dumps({"columns": list(X.columns)}, indent=2), encoding="utf-8")
        # Debug logging
        logger.info(f"🔍 MLflow artifact root: {os.getenv('MLFLOW_S3_ENDPOINT_URL', 'Not set')}")
        logger.info(f"🔍 ARTIFACT_ROOT env var: {os.getenv('ARTIFACT_ROOT', 'Not set')}")
        logger.info(f"🔍 MLflow tracking URI: {mlflow.get_tracking_uri()}")
        logger.info(f"🔍 MLflow artifact URI: {mlflow.get_artifact_uri()}")
        
        try:
            mlflow.log_artifact(str(cols_file), artifact_path="schema")
            logger.info("✅ Artifact logged successfully")
        except Exception as e:
            logger.error(f"❌ Failed to log artifact: {e}")
            logger.error(f"🔍 Trying to upload to bucket extracted from error: {str(e)}")
            raise

        # - si hubo GridSearch, sube cv_results_
        if cv_results is not None:
            cv_file = Path(args.output_dir) / "cv_results.json"
            with open(cv_file, "w", encoding="utf-8") as f:
                json.dump({k:list(map(lambda x: x if isinstance(x,(int,float,str,bool)) else str(x), v))
                           for k,v in cv_results.items()}, f, indent=2)
            mlflow.log_artifact(str(cv_file), artifact_path="cv")

        if model_params_path and Path(model_params_path).exists():
            mlflow.log_artifact(model_params_path, artifact_path="artifacts")

        # 4.5 Log del modelo a MLflow (con firma, si se puede)
        signature = None
        input_example = None
        if infer_signature is not None:
            try:
                signature = infer_signature(X_train, model.predict(X_train[:50]))
                input_example = X_train.head(2)
            except Exception as e:
                logger.warning(f"No se pudo inferir signature: {e}")

        registered_name = os.getenv("MLFLOW_REGISTERED_MODEL", "").strip() or None
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_name
        )

    # Resumen final
    logger.info("\n" + "=" * 60)
    logger.info("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    logger.info("=" * 60)
    logger.info(f"Modelo guardado en: {args.output_dir}")
    logger.info("Para evaluar el modelo, ejecuta:")
    logger.info(
        f"  python src/models/evaluate_model.py "
        f"{args.output_dir}/decision_tree_model.pkl "
        f"{args.output_dir}/train_test_split.pkl "
        f"models/label_encoders.pkl reports/metrics/"
    )

if __name__ == '__main__':
    main()