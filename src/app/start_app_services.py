import subprocess
import sys
import signal
import time
import logging
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("app_services")

def log_subprocess_output(process, service_name):
    """Read and log output from subprocess"""
    for line in iter(process.stdout.readline, b''):
        line_str = line.decode('utf-8').strip()
        if line_str:
            logger.info(f"[{service_name}] {line_str}")

def start_services():
    logger.info("Starting FastAPI and Streamlit services in app container...")
    
    # Start FastAPI backend in the background
    fastapi_process = subprocess.Popen(
        ["uvicorn", "src.app.api:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=False,
    )
    
    # Start thread to capture FastAPI logs
    fastapi_log_thread = threading.Thread(
        target=log_subprocess_output, 
        args=(fastapi_process, "FastAPI"),
        daemon=True
    )
    fastapi_log_thread.start()
    
    # Give FastAPI a moment to start
    time.sleep(2)
    
    # Start Streamlit frontend in the background
    streamlit_process = subprocess.Popen(
        ["streamlit", "run", "src/frontend/streamlit_app.py", 
         "--server.port", "8501", 
         "--server.address", "0.0.0.0",
         "--server.headless", "true"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=False,
    )
    
    # Start thread to capture Streamlit logs
    streamlit_log_thread = threading.Thread(
        target=log_subprocess_output, 
        args=(streamlit_process, "Streamlit"),
        daemon=True
    )
    streamlit_log_thread.start()
    
    # Log startup status
    logger.info("FastAPI running on http://0.0.0.0:8000")
    logger.info("Streamlit running on http://0.0.0.0:8501")
    
    # Function to handle signals and shut down gracefully
    def terminate_processes(signum, frame):
        logger.info("\nReceived signal to terminate. Shutting down services...")
        streamlit_process.terminate()
        fastapi_process.terminate()
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, terminate_processes)
    signal.signal(signal.SIGINT, terminate_processes)
    
    # Monitor processes and exit if either one exits
    while True:
        if fastapi_process.poll() is not None:
            logger.info("FastAPI process exited. Shutting down container.")
            streamlit_process.terminate()
            break
        if streamlit_process.poll() is not None:
            logger.info("Streamlit process exited. Shutting down container.")
            fastapi_process.terminate() 
            break
        time.sleep(1)

if __name__ == "__main__":
    start_services() 