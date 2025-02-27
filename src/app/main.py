# FastAPI App
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from enum import IntEnum
from typing import List
import logging
# from opentelemetry import trace
# from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
# from opentelemetry.trace import Status, StatusCode  
# from opentelemetry.sdk.trace import TracerProvider
# from opentelemetry.sdk.trace.export import SimpleSpanProcessor
# from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from src.workflows.inference import initialize_model, predict
from contextlib import asynccontextmanager

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize tracing
# tracer_provider = TracerProvider()
# otlp_exporter = OTLPSpanExporter()
# span_processor = SimpleSpanProcessor(otlp_exporter)
# tracer_provider.add_span_processor(span_processor)
# trace.set_tracer_provider(tracer_provider)

# Define request and response models
class DocumentRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    predicted_class: int
    confidence: float
    probabilities: list[float]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize model
    try:
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
    
    # Comment out the instrumentation
    # FastAPIInstrumentor.instrument_app(app)
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize model when the application starts"""
        try:
            # TODO: Change to config.yaml value, not hardcoded path
            initialize_model("src/models/final_model/roberta_mlp_best_model_torchscript.pt")
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            # In a production environment, you might want to raise an exception
            # to prevent the app from starting with a broken model
            raise e
    
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
