# 🎓 Student Performance Prediction - MLOps Project

Proyecto de Machine Learning para predecir el rendimiento académico de estudiantes usando Árboles de Decisión.

**Equipo**: EQUIPO54_MLOps  
**Modelo**: Decision Tree Classifier

---

## 📋 Tabla de Contenidos

- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Uso Rápido](#uso-rápido)
- [Pipeline Completo](#pipeline-completo)
- [Hacer Predicciones](#hacer-predicciones)
- [Métricas del Modelo](#métricas-del-modelo)

---

## 📁 Estructura del Proyecto

```
EQUIPO54_MLOps/
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── make_dataset.py              # Procesamiento de datos
│   │   ├── raw/                         # Datos originales sin modificar
│   │   │   └── student_entry_performance_modified.csv
│   │   └── processed/                   # Datos procesados
│   │       └── student_performance_clean.csv
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py               # Entrenamiento
│   │   └── predict_model.py             # Predicciones
│   ├── pipeline/
│   │   ├── run_pipeline.py              # Orquestador del pipeline
│   │   ├── mlflow_runner.py             # Integración con MLflow
│   │   └── params.yaml                  # Configuración del pipeline
│   └── api/                             # 🆕 REST API con FastAPI
│       ├── main.py                      # Aplicación FastAPI
│       ├── model.py                     # Carga y predicción del modelo
│       ├── schemas.py                   # Validación con Pydantic
│       └── README.md                    # Documentación de la API
│
├── models/                              # Modelos entrenados
│   └── latest/                          # Última versión del modelo
│       ├── decision_tree_model.pkl      # Modelo principal
│       ├── label_encoders.pkl           # Encoders
│       ├── model_metrics.pkl            # Métricas (pickle)
│       └── model_metrics.json           # Métricas (JSON legible)
│
├── notebooks/                           # Jupyter notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 02_decision_tree_training.ipynb
│
├── requirements.txt                     # Dependencias
└── README.md                            # Este archivo
```

---

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/EQUIPO54_MLOps.git
cd EQUIPO54_MLOps
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv

# En Windows
venv\Scripts\activate

# En Mac/Linux
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## ⚡ Uso Rápido

### Opción 1: Usar modelo pre-entrenado

Si ya tienes el modelo entrenado en `models/`:

```bash
# Predicción interactiva (ingresa datos manualmente)
python src/models/predict_model.py models/ --interactive

# Predicción desde CSV
python src/models/predict_model.py models/ new_students.csv --output predictions.csv
```

### Opción 2: Entrenar desde cero

```bash
# Paso 1: Procesar datos
python src/data/make_dataset.py src/data/raw/student_entry_performance_modified.csv src/data/processed/student_performance_clean.csv

# Paso 2: Entrenar modelo
python src/models/train_model.py src/data/processed/student_performance_clean.csv models/ --no-optimize

# Paso 3: Hacer predicciones
python src/models/predict_model.py models/ --interactive
```

---

## 🔄 Pipeline Completo

Se puede ejecutar todo el pipeline con el siguiente comando. Este reproducirá automáticamente las etapas definidas en `dvc.yaml`, utilizando los parámetros establecidos en `params.yaml`:

```bash
dvc repro
```

### 1. Procesamiento de Datos

```bash
python src/data/make_dataset.py \
    src/data/raw/student_entry_performance_modified.csv \
    src/data/processed/student_performance_clean.csv
```

**Lo que hace:**
- ✅ Limpia y estandariza texto (mayúsculas, espacios)
- ✅ Normaliza categorías de Performance
- ✅ Elimina valores nulos
- ✅ Guarda datos limpios

### 2. Entrenamiento del Modelo

```bash
# Con optimización de hiperparámetros (recomendado, ~5-10 min)
python src/models/train_model.py \
    src/data/processed/student_performance_clean.csv \
    models/

# Sin optimización (más rápido, ~30 seg)
- Cambiar los parametros desde `params.yaml`
```

**Opciones adicionales:**

- Se pueden cambiar los hiperparámetros y la forma de generar los datos de entrenamiento desde `params.yaml`

**Lo que genera:**
- `decision_tree_model.pkl` - Modelo entrenado
- `label_encoders.pkl` - Encoders para variables categóricas
- `model_metrics.pkl` - Métricas del modelo (pickle)
- `model_metrics.json` - Métricas del modelo (JSON)

### 3. Hacer Predicciones

#### **Modo Interactivo** (una predicción a la vez)

```bash
python src/models/predict_model.py models/ --interactive
```

Te pedirá ingresar cada característica del estudiante:

```
Ingresa los datos del estudiante:

Gender
  Ejemplos válidos: MALE, FEMALE
  → Gender: MALE

Caste
  Ejemplos válidos: GENERAL, OBC, SC, ST
  → Caste: GENERAL

...

📊 RESULTADO DE LA PREDICCIÓN
=================================
🎯 Rendimiento Predicho: Excellent
📈 Confianza: 85.32%
```

#### **Modo Batch** (múltiples predicciones desde CSV)

```bash
python src/models/predict_model.py models/ new_students.csv --output predictions.csv
```

El CSV de salida incluirá:
- Todas las columnas originales
- `Predicted_Performance` - Predicción
- `Prob_Excellent`, `Prob_Very Good`, `Prob_Good`, `Prob_Average` - Probabilidades
- `Confidence` - Confianza de la predicción

---

## 🧪 Ejecutar Pruebas

El proyecto incluye **30 pruebas automatizadas** (unitarias e integración):

```bash
# Ejecutar todas las pruebas
pytest

# Modo silencioso
pytest -q

# Con cobertura de código
pytest --cov=src --cov-report=html
```

**Cobertura de tests:**
- ✅ Procesamiento de datos (`test_data_processing.py`)
- ✅ Construcción de features (`test_feature_engineering.py`)
- ✅ Predicción del modelo (`test_prediction.py`)
- ✅ Pipeline extremo a extremo (`test_integration.py`)

Ver documentación completa: [`tests/README.md`](tests/README.md)

---

## 📊 Métricas del Modelo

El modelo entrenado genera las siguientes métricas (guardadas en `models/model_metrics.json`):

### Ejemplo de métricas:

```json
{
  "train_accuracy": 0.9234,
  "test_accuracy": 0.8567,
  "cv_score": 0.8521,
  "best_params": {
    "max_depth": 15,
    "min_samples_split": 20,
    "min_samples_leaf": 10,
    "criterion": "gini"
  },
  "classification_report": {
    "Excellent": {
      "precision": 0.87,
      "recall": 0.89,
      "f1-score": 0.88
    },
    ...
  },
  "feature_importance_top10": {
    "Class_XII_Percentage": 0.3245,
    "Class_ X_Percentage": 0.2891,
    "coaching": 0.1456,
    ...
  }
}
```

---

## 🔧 Troubleshooting

### Error: "Modelo no encontrado"

Asegúrate de haber entrenado el modelo primero:

```bash
python src/models/train_model.py src/data/processed/student_performance_clean.csv models/
```

### Error: "Valores no vistos"

Si haces predicciones con valores que no estaban en el entrenamiento, el script usará un valor por defecto. Verifica que tus datos sean consistentes.

### Error de importación

Instala las dependencias:

```bash
pip install -r requirements.txt
```

---

## 🌐 API REST (FastAPI)

El proyecto incluye una API REST completa para servir el modelo en producción.

### Iniciar la API

```bash
# Instalar dependencias adicionales
pip install -r src/api/requirements.txt

# Ejecutar el servidor
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints Disponibles

- **Documentación interactiva**: http://localhost:8000/docs
- **POST `/predict`**: Hacer predicciones
- **GET `/health`**: Estado de la API y modelo
- **GET `/model/info`**: Información del modelo cargado

### Ejemplo de Uso

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Respuesta:**
```json
{
  "prediction": "Good",
  "probability": 0.85,
  "model_version": "latest"
}
```

📚 **Documentación completa**: Ver `src/api/README.md`

### Información del Modelo

- **Ruta del artefacto**: `models/latest/decision_tree_model.pkl`
- **Versión actual**: `latest` (actualizada por el pipeline)
- **Registro MLflow**: `models:/student-performance-dt/<version>`
- **Tipo de modelo**: DecisionTreeClassifier (scikit-learn)
- **Features de entrada**: 12 características (demográficas, académicas, hábitos de estudio)
- **Clases de salida**: Average, Good, Very Good, Excellent

---

## 📈 Próximos Pasos (Roadmap)

- [x] API REST con FastAPI ✅
- [ ] Dockerización del proyecto
- [ ] CI/CD con GitHub Actions
- [x] Monitoring con MLflow ✅
- [x] Pruebas unitarias con pytest ✅
- [ ] Dashboard interactivo con Streamlit

---

## 👥 Contribuidores

**EQUIPO54_MLOps**

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
