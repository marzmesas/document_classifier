# FastAPI App
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from enum import IntEnum
from typing import List
import logging
from contextlib import asynccontextmanager
from src.workflows.inference import initialize_model, predict
import os

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Check if we're in testing mode
TESTING = os.environ.get("TESTING", "False").lower() == "true"

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

    # Define resource with service name
    resource = Resource.create({
        ResourceAttributes.SERVICE_NAME: "document-classifier"
    })

    # Create a meter instance
    meter = metrics.get_meter_provider().get_meter("document-classifier")

    # Define a custom counter metric to track successful inferences
    successful_inferences_counter = meter.create_counter(
        "successful_inferences_total", 
        description="Total number of successful predictions"
    )

    # Initialize tracing with resource
    tracer_provider = TracerProvider(resource=resource)
    span_exporter = OTLPSpanExporter(endpoint="http://otel-collector:4317", insecure=True)
    span_processor = SimpleSpanProcessor(span_exporter)
    tracer_provider.add_span_processor(span_processor)
    trace.set_tracer_provider(tracer_provider)

    # Initialize metrics with same resource
    metric_exporter = OTLPMetricExporter(endpoint="http://otel-collector:4317", insecure=True)
    metric_reader = PeriodicExportingMetricReader(metric_exporter)
    metrics_provider = MeterProvider(
        resource=resource,
        metric_readers=[metric_reader]
    )
    metrics.set_meter_provider(metrics_provider)
else:
    # Create dummy counter for testing
    class DummyCounter:
        def add(self, value):
            pass
    
    successful_inferences_counter = DummyCounter()

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Startup: Initialize model
    try:
        # TODO: Change to config.yaml value, not hardcoded path
        if not TESTING:  # Skip model initialization in testing
            initialize_model("src/models/final_model/roberta_mlp_best_model_torchscript.pt")
            logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise e
    
    yield  # Server is running and handling requests
    
    # Shutdown: Clean up resources if needed
    logger.info("Shutting down application")

def create_app() -> FastAPI:
    app = FastAPI(
        title="Document Classification API",
        description="API for classifying documents into predefined categories",
        version="1.0.0",
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
        try:
            result = predict(request.text)
            successful_inferences_counter.add(1)
            return result
        except RuntimeError as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Model prediction failed. Please try again later."
            )
    
    @app.get("/")
    async def root():
        return {"message": "Document Classification API is running"}
    
    @app.get("/health", response_model=HealthResponse)
    def health_check():
        """
        Health check endpoint to verify the API is running properly.
        Returns a 200 OK response with status: healthy if everything is working.
        """
        return {"status": "healthy"}

    return app

# Create the application instance
app = create_app()
