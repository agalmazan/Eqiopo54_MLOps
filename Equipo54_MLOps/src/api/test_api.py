"""
Script de prueba para la API de Student Performance Prediction
Valida que todos los endpoints funcionen correctamente
"""

import requests
import json
import sys

# Configuración
BASE_URL = "http://localhost:8000"

# Colores para output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'


def print_test(name, passed):
    """Imprime resultado de prueba con color"""
    status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
    print(f"{status} - {name}")


def test_root():
    """Test del endpoint raíz"""
    try:
        response = requests.get(f"{BASE_URL}/")
        passed = response.status_code == 200 and "message" in response.json()
        print_test("GET / (Root endpoint)", passed)
        return passed
    except Exception as e:
        print_test(f"GET / - Error: {e}", False)
        return False


def test_health():
    """Test del endpoint de health check"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        data = response.json()
        passed = (
            response.status_code == 200 and
            "status" in data and
            "model_loaded" in data
        )
        print_test("GET /health", passed)
        if passed:
            print(f"   Status: {data['status']}, Model Loaded: {data['model_loaded']}")
        return passed
    except Exception as e:
        print_test(f"GET /health - Error: {e}", False)
        return False


def test_model_info():
    """Test del endpoint de información del modelo"""
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        data = response.json()
        passed = (
            response.status_code == 200 and
            "model_version" in data and
            "features" in data
        )
        print_test("GET /model/info", passed)
        if passed:
            print(f"   Model Version: {data['model_version']}")
            print(f"   Features Count: {len(data['features'])}")
        return passed
    except Exception as e:
        print_test(f"GET /model/info - Error: {e}", False)
        return False


def test_predict_valid():
    """Test de predicción con datos válidos"""
    try:
        payload = {
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
        }
        
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        data = response.json()
        
        passed = (
            response.status_code == 200 and
            "prediction" in data and
            "model_version" in data
        )
        
        print_test("POST /predict (Valid data)", passed)
        if passed:
            print(f"   Prediction: {data['prediction']}")
            if data.get('probability'):
                print(f"   Probability: {data['probability']:.4f}")
            print(f"   Model Version: {data['model_version']}")
        return passed
    except Exception as e:
        print_test(f"POST /predict - Error: {e}", False)
        return False


def test_predict_invalid():
    """Test de predicción con datos inválidos (debe fallar)"""
    try:
        # Datos inválidos: marks fuera de rango
        payload = {
            "gender": "Male",
            "caste": "General",
            "mathematics_marks": 150,  # Invalid: > 100
            "english_marks": -10,       # Invalid: < 0
            "science_marks": 82,
            "father_occupation": "Government Officer",
            "mother_occupation": "Teacher",
            "number_of_siblings": 2,
            "boarding": "No",
            "distance_from_home": "Near",
            "time": 5,
            "coaching": "Yes"
        }
        
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        # Debe retornar error 422 (Validation Error)
        passed = response.status_code == 422
        
        print_test("POST /predict (Invalid data - should reject)", passed)
        if not passed:
            print(f"   Expected status 422, got {response.status_code}")
        return passed
    except Exception as e:
        print_test(f"POST /predict (Invalid) - Error: {e}", False)
        return False


def test_predict_excellent_student():
    """Test con un estudiante de alto rendimiento"""
    try:
        payload = {
            "gender": "Female",
            "caste": "General",
            "mathematics_marks": 95,
            "english_marks": 92,
            "science_marks": 94,
            "father_occupation": "Government Officer",
            "mother_occupation": "Teacher",
            "number_of_siblings": 1,
            "boarding": "Yes",
            "distance_from_home": "Near",
            "time": 7,
            "coaching": "Yes"
        }
        
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        data = response.json()
        
        passed = response.status_code == 200 and "prediction" in data
        
        print_test("POST /predict (Excellent student)", passed)
        if passed:
            print(f"   Prediction: {data['prediction']}")
            if data.get('probability'):
                print(f"   Confidence: {data['probability']:.4f}")
        return passed
    except Exception as e:
        print_test(f"POST /predict (Excellent) - Error: {e}", False)
        return False


def run_all_tests():
    """Ejecuta todas las pruebas"""
    print("\n" + "="*60)
    print("🧪 Testing Student Performance Prediction API")
    print("="*60 + "\n")
    
    # Verificar que la API esté corriendo
    try:
        requests.get(f"{BASE_URL}/", timeout=2)
    except requests.exceptions.RequestException:
        print(f"{RED}❌ API no está corriendo en {BASE_URL}{RESET}")
        print(f"\n{YELLOW}Inicia la API primero:{RESET}")
        print("   uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000\n")
        return False
    
    # Ejecutar pruebas
    results = []
    
    print(f"{YELLOW}Endpoints básicos:{RESET}")
    results.append(test_root())
    results.append(test_health())
    results.append(test_model_info())
    
    print(f"\n{YELLOW}Predicciones:{RESET}")
    results.append(test_predict_valid())
    results.append(test_predict_excellent_student())
    results.append(test_predict_invalid())
    
    # Resumen
    print("\n" + "="*60)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"{GREEN}✓ Todas las pruebas pasaron ({passed}/{total}){RESET}")
        print("="*60 + "\n")
        return True
    else:
        print(f"{RED}✗ Algunas pruebas fallaron ({passed}/{total}){RESET}")
        print("="*60 + "\n")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
