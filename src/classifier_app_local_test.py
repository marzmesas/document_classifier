import subprocess
import sys
import time
import os
import webbrowser
from pathlib import Path

def run_apps():
    os.environ["TESTING"] = "True"
    # Get the project root directory
    root_dir = Path(__file__).parent.parent
    
    # Start FastAPI backend
    fastapi_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.app.api:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=root_dir
    )
    
    # Wait a bit for FastAPI to start
    time.sleep(2)
    
    # Start Streamlit frontend
    streamlit_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "src/frontend/streamlit_app.py"],
        cwd=root_dir
    )
    
    # Open the browser automatically
    # webbrowser.open("http://localhost:8501")
    
    try:
        # Keep the script running
        fastapi_process.wait()
        streamlit_process.wait()
    except KeyboardInterrupt:
        # Handle clean shutdown
        print("\nShutting down services...")
        fastapi_process.terminate()
        streamlit_process.terminate()
        fastapi_process.wait()
        streamlit_process.wait()
        print("Services stopped.")

if __name__ == "__main__":
    run_apps()