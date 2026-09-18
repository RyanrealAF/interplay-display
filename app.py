"""
Main application entry point for Uvicorn (app:app).
Re-exports the FastAPI app instance from api.py which defines the /api/v1/analyze route.
"""
from api import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
