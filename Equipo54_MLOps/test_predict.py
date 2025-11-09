"""Quick script to test the API prediction endpoint"""
import requests
import json

url = "http://127.0.0.1:8000/predict"

payload = {
    "gender": "MALE",
    "caste": "GENERAL",
    "coaching": "OA",
    "time": "FIVE",
    "class_ten_education": "CBSE",
    "twelve_education": "CBSE",
    "medium": "ENGLISH",
    "class_x_percentage": "EXCELLENT",
    "class_xii_percentage": "GOOD",
    "father_occupation": "BANK_OFFICIAL",
    "mother_occupation": "HOUSE_WIFE"
}

print("Sending prediction request...")
print(f"URL: {url}")
print(f"Payload: {json.dumps(payload, indent=2)}")

try:
    response = requests.post(url, json=payload)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")
