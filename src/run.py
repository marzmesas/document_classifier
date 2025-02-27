import uvicorn
from src.app.main import app

if __name__ == "__main__":
    uvicorn.run(
        "src.app.main:app",  # Changed from "app.main:app" to match our package structure
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    ) 