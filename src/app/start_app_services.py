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
    """
    Read and log output from subprocess streams.
    
    This function continuously reads from the subprocess stdout pipe
    and logs each line with the service name for identification.
    
    Args:
        process (subprocess.Popen): Process object to monitor
        service_name (str): Name of the service for log identification
        
    Returns:
        None: Function runs until the process stream closes
    """
    for line in iter(process.stdout.readline, b''):
        line_str = line.decode('utf-8').strip()
        if line_str:
            logger.info(f"[{service_name}] {line_str}")

def start_services():
    """
    Start FastAPI and Streamlit services in parallel with logging.
    
    This function starts both the backend API and frontend Streamlit app
    as subprocesses, monitors their output, and handles graceful shutdown.
    
    Returns:
        None: Function runs until terminated by signal handler
    """
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
    
    logger.info("Both services started. Press Ctrl+C to stop.")
    
    # Set up signal handler for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received, stopping services...")
        # Terminate processes
        fastapi_process.terminate()
        streamlit_process.terminate()
        # Wait for processes to terminate
        fastapi_process.wait()
        streamlit_process.wait()
        logger.info("Services stopped. Exiting.")
        sys.exit(0)
    
    # Register signal handler for SIGINT (Ctrl+C) and SIGTERM
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Keep the script running
    try:
        while True:
            # Check if either process has unexpectedly terminated
            if fastapi_process.poll() is not None:
                logger.error("FastAPI process terminated unexpectedly")
                break
                
            if streamlit_process.poll() is not None:
                logger.error("Streamlit process terminated unexpectedly")
                break
                
            time.sleep(1)
            
    except Exception as e:
        logger.error(f"Error during service monitoring: {e}")
    finally:
        # Clean up
        logger.info("Shutting down...")
        fastapi_process.terminate()
        streamlit_process.terminate()
        fastapi_process.wait()
        streamlit_process.wait()

if __name__ == "__main__":
    start_services() 