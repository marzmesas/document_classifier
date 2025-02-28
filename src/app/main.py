# FastAPI App
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from enum import IntEnum
from typing import List
import logging
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.trace import Status, StatusCode  
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from src.workflows.inference import initialize_model, predict
from contextlib import asynccontextmanager
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from prometheus_client import start_http_server
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes

LOCAL_TESTING = True  # Set to False to use collector

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define resource with service name
resource = Resource.create({
    ResourceAttributes.SERVICE_NAME: "document-classifier"
})

# Initialize tracing with resource
tracer_provider = TracerProvider(resource=resource)
if LOCAL_TESTING:
    span_exporter = ConsoleSpanExporter()
else:
    span_exporter = OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)
span_processor = SimpleSpanProcessor(span_exporter)
tracer_provider.add_span_processor(span_processor)
trace.set_tracer_provider(tracer_provider)

# Initialize metrics with same resource
if LOCAL_TESTING:
    metric_exporter = ConsoleMetricExporter()
else:
    metric_exporter = OTLPMetricExporter(endpoint="http://localhost:4317", insecure=True)
    start_http_server(port=9090)
metric_reader = PeriodicExportingMetricReader(metric_exporter)
metrics_provider = MeterProvider(
    resource=resource,
    metric_readers=[metric_reader]
)
metrics.set_meter_provider(metrics_provider)

# Define request and response models
class DocumentRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    predicted_class: int
    confidence: float
    probabilities: list[float]

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Startup: Initialize model
    try:
        # TODO: Change to config.yaml value, not hardcoded path
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
    
    # Enable OpenTelemetry instrumentation
    FastAPIInstrumentor.instrument_app(app)
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict_endpoint(request: DocumentRequest):
        try:
            result = predict(request.text)
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
    
    return app

# Create the application instance
app = create_app()
