"""
Helper script to run the FastAPI application
Run from Equipo54_MLOps directory: python run_api.py
"""

import sys
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Student Performance Prediction API...")
    print(f"📁 Working directory: {current_dir}")
    print(f"🌐 API will be available at: http://127.0.0.1:8000")
    print(f"📚 Interactive docs at: http://127.0.0.1:8000/docs")
    print("\nPress CTRL+C to stop the server\n")
    
    uvicorn.run(
        "src.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
