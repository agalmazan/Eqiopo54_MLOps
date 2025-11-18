# 🧪 Guía de Testing

## Descripción

Este proyecto implementa pruebas automatizadas usando **pytest** para validar componentes críticos del pipeline de ML.

## Estructura de Tests

```
tests/
├── __init__.py
├── test_data_processing.py      # Pruebas unitarias: procesamiento de datos
├── test_feature_engineering.py  # Pruebas unitarias: construcción de features
├── test_prediction.py            # Pruebas unitarias: modelo de predicción
└── test_integration.py           # Pruebas de integración: pipeline completo
```

## Tipos de Pruebas

### 1. Pruebas Unitarias

**test_data_processing.py**
- ✅ Limpieza de texto (`clean_text`)
- ✅ Procesamiento completo de datos
- ✅ Estandarización de categorías
- ✅ Manejo de valores nulos

**test_feature_engineering.py**
- ✅ Construcción de features
- ✅ Codificación de variables categóricas
- ✅ Guardado de encoders
- ✅ Preservación del número de muestras

**test_prediction.py**
- ✅ Inicialización del predictor
- ✅ Carga de artefactos del modelo
- ✅ Preprocesamiento de entrada
- ✅ Formato de predicciones
- ✅ Casos extremos (single/multiple predictions)

### 2. Pruebas de Integración

**test_integration.py**
- ✅ Pipeline extremo a extremo (raw → clean → features → train → predict)
- ✅ Consistencia de datos a través del pipeline
- ✅ División train/test
- ✅ Cálculo de métricas (accuracy, F1)
- ✅ Manejo de errores (columnas faltantes, categorías no vistas)
- ✅ Distribución de predicciones

## Instalación de Dependencias

```bash
# Activar entorno virtual
.\itesm_venv\Scripts\Activate.ps1

# Instalar pytest si no está instalado
pip install pytest pytest-cov
```

## Ejecución de Tests

### Ejecutar todos los tests

```bash
cd Equipo54_MLOps
pytest
```

### Ejecutar tests en modo silencioso (quiet)

```bash
pytest -q
```

### Ejecutar tests con verbose

```bash
pytest -v
```

### Ejecutar tests específicos

```bash
# Por archivo
pytest tests/test_data_processing.py

# Por clase
pytest tests/test_prediction.py::TestStudentPerformancePredictor

# Por función específica
pytest tests/test_data_processing.py::TestCleanText::test_clean_text_basic
```

### Ejecutar solo pruebas unitarias o de integración

```bash
# Solo unitarias
pytest tests/test_data_processing.py tests/test_feature_engineering.py tests/test_prediction.py

# Solo integración
pytest tests/test_integration.py
```

### Ejecutar con cobertura de código

```bash
pytest --cov=src --cov-report=html
```

Esto genera un reporte HTML en `htmlcov/index.html`

### Ver solo tests que fallaron

```bash
pytest --lf  # last failed
```

### Ejecutar tests en paralelo (más rápido)

```bash
pip install pytest-xdist
pytest -n auto
```

## Comandos Útiles

| Comando | Descripción |
|---------|-------------|
| `pytest` | Ejecuta todos los tests |
| `pytest -q` | Modo silencioso (quiet) |
| `pytest -v` | Modo verbose (detallado) |
| `pytest -x` | Detener al primer fallo |
| `pytest -k "text"` | Ejecutar tests que contengan "text" en el nombre |
| `pytest --collect-only` | Mostrar qué tests se ejecutarían sin ejecutarlos |
| `pytest --tb=short` | Tracebacks más cortos |
| `pytest --maxfail=3` | Detener después de 3 fallos |

## Interpretar Resultados

### Salida exitosa
```
======================== test session starts ========================
collected 35 items

tests/test_data_processing.py ........                        [ 22%]
tests/test_feature_engineering.py ......                      [ 40%]
tests/test_prediction.py .............                        [ 77%]
tests/test_integration.py ........                           [100%]

======================== 35 passed in 2.45s ========================
```

### Salida con fallos
```
======================== FAILURES ========================
_______________ TestCleanText.test_clean_text_basic _______________

    def test_clean_text_basic(self):
>       assert clean_text("  hello  ") == "HELLO"
E       AssertionError: assert 'hello' == 'HELLO'

tests/test_data_processing.py:18: AssertionError
```

## Mejores Prácticas

1. **Ejecutar tests antes de commit**
   ```bash
   pytest -q
   ```

2. **Mantener tests rápidos**: Las pruebas unitarias deben ejecutarse en < 1 segundo

3. **Tests aislados**: Cada test debe ser independiente y no depender de otros

4. **Usar fixtures**: Para compartir configuración común entre tests

5. **Nombres descriptivos**: `test_clean_text_removes_spaces` es mejor que `test1`

## Integración con CI/CD

Los tests se ejecutan automáticamente en cada push/PR mediante GitHub Actions.

Ver configuración en `.github/workflows/`

## Troubleshooting

### Error: "No module named 'src'"
```bash
# Asegurarse de ejecutar desde el directorio correcto
cd Equipo54_MLOps
pytest
```

### Error: "fixture not found"
```bash
# Verificar que pytest está actualizado
pip install --upgrade pytest
```

### Tests muy lentos
```bash
# Ejecutar solo tests rápidos primero
pytest -m "not slow"
```

## Métricas de Cobertura

Objetivo: **> 80% de cobertura** en módulos críticos:
- `src/data/make_dataset.py`
- `src/features/build_features.py`
- `src/models/predict_model.py`

Verificar con:
```bash
pytest --cov=src --cov-report=term-missing
```

## Siguiente Paso: Mejoras

- [ ] Agregar tests para API FastAPI
- [ ] Implementar tests de carga/performance
- [ ] Agregar tests de validación de datos
- [ ] Crear tests de regresión visual para gráficas

---

**Última actualización**: Noviembre 2025
