"""
Script para evaluar modelo entrenado

Uso:
    python src/models/evaluate_model.py \
        models/decision_tree_model.pkl \
        models/train_test_split.pkl \
        models/label_encoders.pkl \
        reports/metrics/
"""

import os
import json
import argparse
import logging
import subprocess

import joblib
import pandas as pd
import mlflow
from datetime import datetime

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

# ---------------- utilidades ----------------
def _git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"
# --------------------------------------------

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "student-performance"))


def load_artifacts(model_path, splits_path, encoders_path):
    """
    Carga modelo, splits y encoders
    
    Args:
        model_path: Ruta del modelo
        splits_path: Ruta de los splits
        encoders_path: Ruta de los encoders
        
    Returns:
        tuple: (model, X_train, X_test, y_train, y_test, encoders_data)
    """
    logger.info(f"Cargando modelo desde: {model_path}")
    model = joblib.load(model_path)

    logger.info(f"Cargando splits desde: {splits_path}")
    splits = joblib.load(splits_path)
    X_train = splits['X_train']
    X_test = splits['X_test']
    y_train = splits['y_train']
    y_test = splits['y_test']
    
    logger.info(f"Cargando encoders desde: {encoders_path}")
    encoders_data = joblib.load(encoders_path)

    logger.info("✅ Artefactos cargados correctamente")
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    return model, X_train, X_test, y_train, y_test, encoders_data


def evaluate_model(model, X_train, X_test, y_train, y_test, le_target):
    """
    Evalúa el modelo en train y test
    
    Args:
        model: Modelo entrenado
        X_train, X_test: Features
        y_train, y_test: Target
        le_target: LabelEncoder del target
        
    Returns:
        dict: Métricas del modelo
    """
    logger.info("\n=== EVALUANDO MODELO ===")

    # Predicciones
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Métricas principales
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    # Métricas macro (útiles en set desbalanceados)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred_test, average="macro", zero_division=0
    )

    logger.info(f"Accuracy (Train): {train_accuracy:.4f}")
    logger.info(f"Accuracy (Test):  {test_accuracy:.4f}")
    logger.info(f"Macro Precision:  {prec_macro:.4f}")
    logger.info(f"Macro Recall:     {rec_macro:.4f}")
    logger.info(f"Macro F1:         {f1_macro:.4f}")

    # Classification report (por clase)
    logger.info("\n=== CLASSIFICATION REPORT (Test) ===")
    report_txt = classification_report(
        y_test, y_pred_test, target_names=getattr(le_target, "classes_", None)
    )
    print(report_txt)
    report = classification_report(
        y_test,
        y_pred_test,
        target_names=getattr(le_target, "classes_", None),
        output_dict=True,
    )

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred_test)
    logger.info("\n=== MATRIZ DE CONFUSIÓN ===")
    logger.info(f"\n{cm}")

    # Importancias (si el modelo las expone)
    try:
        fi = pd.DataFrame(
            {"feature": X_train.columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)
        feature_importance = fi.to_dict("records")
        logger.info("\n=== TOP 10 CARACTERÍSTICAS MÁS IMPORTANTES ===")
        for _, row in fi.head(10).iterrows():
            logger.info(f"{row['feature']:20s}: {row['importance']:.4f}")
    except Exception:
        feature_importance = []
        fi = pd.DataFrame(columns=["feature", "importance"])

    metrics = {
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "feature_importance": feature_importance,
        "n_train": int(getattr(X_train, "shape", [0])[0]),
        "n_test": int(getattr(X_test, "shape", [0])[0]),
    }

    return metrics, fi, cm


def save_metrics(metrics, fi_df, cm_array, output_dir):
    """
    Guarda métricas en archivos (para DVC) y devuelve rutas para loguear a MLflow
    """
    os.makedirs(output_dir, exist_ok=True)

    # PKL completo
    metrics_pkl_path = os.path.join(output_dir, "metrics.pkl")
    joblib.dump(metrics, metrics_pkl_path)
    logger.info(f"✅ Métricas PKL guardadas en: {metrics_pkl_path}")

    # JSON legible (sin la lista completa de FI)
    metrics_json = dict(metrics)
    metrics_json["feature_importance_top10"] = {
        row["feature"]: row["importance"] for row in metrics["feature_importance"][:10]
    }
    metrics_json.pop("feature_importance", None)

    metrics_json_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    logger.info(f"✅ Métricas JSON guardadas en: {metrics_json_path}")

    # CSV de importancias
    fi_csv_path = os.path.join(output_dir, "feature_importance.csv")
    if fi_df is not None and not fi_df.empty:
        fi_df.to_csv(fi_csv_path, index=False)
        logger.info(f"✅ Feature importance CSV guardado en: {fi_csv_path}")
    else:
        fi_csv_path = None

    # Confusion matrix CSV/JSON (útil para dashboards)
    cm_csv_path = os.path.join(output_dir, "confusion_matrix.csv")
    cm_json_path = os.path.join(output_dir, "confusion_matrix.json")
    if cm_array is not None:
        pd.DataFrame(cm_array).to_csv(cm_csv_path, index=False)
        with open(cm_json_path, "w") as f:
            json.dump({"confusion_matrix": cm_array.tolist()}, f, indent=2)
        logger.info(f"✅ Confusion matrix guardada en: {cm_csv_path} y {cm_json_path}")
    else:
        cm_csv_path = cm_json_path = None

    return {
        "metrics_pkl": metrics_pkl_path,
        "metrics_json": metrics_json_path,
        "fi_csv": fi_csv_path,
        "cm_csv": cm_csv_path,
        "cm_json": cm_json_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluar modelo entrenado")
    parser.add_argument("model_path", type=str, help="Ruta del modelo entrenado (.pkl)")
    parser.add_argument("splits_path", type=str, help="Ruta del archivo con train/test splits (.pkl)")
    parser.add_argument("encoders_path", type=str, help="Ruta de los label encoders (.pkl)")
    parser.add_argument("output_dir", type=str, help="Directorio donde guardar las métricas")
    args = parser.parse_args()

    # Verificación
    for pth, label in [
        (args.model_path, "Modelo"),
        (args.splits_path, "Splits"),
        (args.encoders_path, "Encoders"),
    ]:
        if not os.path.exists(pth):
            logger.error(f"{label} no encontrado: {pth}")
            return

    logger.info("=" * 60)
    logger.info("🚀 INICIANDO EVALUACIÓN DEL MODELO")
    logger.info("=" * 60)

    # 1) Cargar artefactos
    model, X_train, X_test, y_train, y_test, encoders_data = load_artifacts(
        args.model_path, args.splits_path, args.encoders_path
    )
    le_target = encoders_data.get("target_encoder", None)

    # 2) Evaluar
    metrics, fi_df, cm_array = evaluate_model(model, X_train, X_test, y_train, y_test, le_target)

    # 3) Guardar (para DVC) y obtener rutas
    paths = save_metrics(metrics, fi_df, cm_array, args.output_dir)

    # 4) Log a MLflow
    with mlflow.start_run(run_name="evaluate"):
        mlflow.set_tag("pipeline_stage", "evaluate")
        mlflow.set_tag("git_rev", _git_rev())
        mlflow.set_tag("timestamp", datetime.now().isoformat())

        # métricas numéricas clave
        mlflow.log_metrics({
            "accuracy_train": metrics["train_accuracy"],
            "accuracy_test": metrics["test_accuracy"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
            "n_train": metrics["n_train"],
            "n_test": metrics["n_test"],
        })

        # artifacts
        if paths["metrics_json"] and os.path.exists(paths["metrics_json"]):
            mlflow.log_artifact(paths["metrics_json"], artifact_path="metrics")

        if paths["fi_csv"] and os.path.exists(paths["fi_csv"]):
            mlflow.log_artifact(paths["fi_csv"], artifact_path="metrics")

        # matriz de confusión (CSV y JSON)
        for cm_path in (paths["cm_csv"], paths["cm_json"]):
            if cm_path and os.path.exists(cm_path):
                mlflow.log_artifact(cm_path, artifact_path="metrics")

    logger.info("\n" + "=" * 60)
    logger.info("✅ EVALUACIÓN COMPLETADA EXITOSAMENTE")
    logger.info("=" * 60)
    logger.info(f"Métricas guardadas en: {args.output_dir}")


if __name__ == "__main__":
    main()
