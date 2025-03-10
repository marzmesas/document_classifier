# FastAPI App
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from enum import IntEnum
from typing import List
import logging
from contextlib import asynccontextmanager
from src.workflows.inference import initialize_model, predict
import os
import yaml

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Check if we're in testing mode
TESTING = os.environ.get("TESTING", "False").lower() == "true"

# Load configuration
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded successfully")
except Exception as e:
    logger.error(f"Error loading configuration: {str(e)}")
    # Fallback to defaults if config can't be loaded
    config = {
        "app": {
            "title": "Document Classification API",
            "description": "API for classifying documents into predefined categories",
            "version": "1.0.0"
        },
        "model": {
            "path": "src/models/final_model/roberta_mlp_best_model_torchscript.pt"
        }
    }
    logger.warning("Using default configuration")

# Define request and response models
class DocumentRequest(BaseModel):
    text: str

class HealthResponse(BaseModel):
    status: str

class PredictionResponse(BaseModel):
    predicted_class: int
    confidence: float
    probabilities: list[float]

# Only import and initialize OpenTelemetry if not in testing mode
if not TESTING:
    from opentelemetry import trace
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.trace import Status, StatusCode  
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    
    # Add these imports for logging
    from opentelemetry._logs import set_logger_provider
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
    
    # Import our custom logging setup
    from src.observability.logging import setup_otel_logging

    # Define resource with service name from config
    service_name = config.get("observability", {}).get("service_name", "document-classifier")
    resource = Resource.create({
        ResourceAttributes.SERVICE_NAME: service_name
    })

    # Create a meter instance
    meter = metrics.get_meter_provider().get_meter(service_name)

    # Define a custom counter metric to track successful inferences
    successful_inferences_counter = meter.create_counter(
        "successful_inferences_total", 
        description="Total number of successful predictions"
    )

    # Get OTel collector endpoint from config
    otel_endpoint = config.get("observability", {}).get(
        "otel_collector_endpoint", "http://otel-collector:4317"
    )

    # Initialize tracing with resource
    tracer_provider = TracerProvider(resource=resource)
    span_exporter = OTLPSpanExporter(endpoint=otel_endpoint, insecure=True)
    span_processor = SimpleSpanProcessor(span_exporter)
    tracer_provider.add_span_processor(span_processor)
    trace.set_tracer_provider(tracer_provider)

    # Initialize metrics with same resource
    metric_exporter = OTLPMetricExporter(endpoint=otel_endpoint, insecure=True)
    metric_reader = PeriodicExportingMetricReader(metric_exporter)
    metrics_provider = MeterProvider(
        resource=resource,
        metric_readers=[metric_reader]
    )
    metrics.set_meter_provider(metrics_provider)
    
    # Initialize logging with same resource
    logger_provider = LoggerProvider(resource=resource)
    log_exporter = OTLPLogExporter(endpoint=otel_endpoint, insecure=True)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
    set_logger_provider(logger_provider)
    
    # Set up OpenTelemetry logging with the same resource
    setup_otel_logging(resource=resource)
    
    # Create a logger for this module
    logger = logging.getLogger(__name__)
    logger.info("OpenTelemetry initialization complete")
else:
    # Create dummy counter for testing
    class DummyCounter:
        def add(self, value):
            pass
    
    successful_inferences_counter = DummyCounter()

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Manage application startup and shutdown lifecycle events.
    
    This async context manager initializes the model on startup and performs cleanup on shutdown.
    
    Args:
        _app (FastAPI): The FastAPI application instance
    
    Yields:
        None: Control is yielded back to FastAPI while the application is running
    
    Raises:
        Exception: If model initialization fails
    """
    # Startup: Initialize model
    try:
        # Get model path from config
        model_path = config["model"]["path"]
        initialize_model(model_path)
        logger.info(f"Model initialized successfully from {model_path}")
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise e
    
    yield  # Server is running and handling requests
    
    # Shutdown: Clean up resources if needed
    logger.info("Shutting down application")

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application instance.
    
    This function loads configuration, creates the FastAPI app with appropriate settings,
    and sets up API endpoints. It also configures OpenTelemetry instrumentation when not
    in testing mode.
    
    Returns:
        FastAPI: Configured FastAPI application instance ready to serve requests
    """
    # Get app config values
    app_config = config.get("app", {})
    app = FastAPI(
        title=app_config.get("title", "Document Classification API"),
        description=app_config.get("description", "API for classifying documents into predefined categories"),
        version=app_config.get("version", "1.0.0"),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Enable OpenTelemetry instrumentation only if not testing
    if not TESTING:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict_endpoint(request: DocumentRequest):
        """
        Process document classification requests.
        
        This endpoint accepts document text and returns classification predictions including
        the predicted class, confidence score, and probability distribution across all classes.
        
        Args:
            request (DocumentRequest): Request object containing the document text to classify
            
        Returns:
            PredictionResponse: Classification results including predicted class, 
                               confidence score, and probability distribution
                               
        Raises:
            HTTPException: If model prediction fails (status code 500)
        """
        try:
            result = predict(request.text)
            successful_inferences_counter.add(1)
            logger.info(f"Prediction successful: {result}")
            return result
        except RuntimeError as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Model prediction failed. Please try again later."
            )
    
    @app.get("/")
    async def root():
        """
        Root endpoint that confirms the API is running.
        
        Returns:
            dict: Simple message indicating the API is operational
        """
        return {"message": "Document Classification API is running"}
    
    @app.get("/health", response_model=HealthResponse)
    def health_check():
        """
        Health check endpoint to verify the API is running properly.
        
        This endpoint is used by monitoring systems to check if the API is operational.
        
        Returns:
            HealthResponse: Response containing status "healthy" if everything is working
        """
        return {"status": "healthy"}

    return app

# Create the application instance
app = create_app()
