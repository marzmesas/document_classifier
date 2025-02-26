# FastAPI App
from fastapi import FastAPI, HTTPException
import logging
from pydantic import BaseModel
import uvicorn
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.trace import Status, StatusCode  
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Initialize Tracing
tracer_provider = TracerProvider()
span_processor = SimpleSpanProcessor(OTLPSpanExporter())
tracer_provider.add_span_processor(span_processor)
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer(__name__)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


app = FastAPI()
FastAPIInstrumentor().instrument_app(app)

# Request Model
class DocumentRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: DocumentRequest):
    with tracer.start_as_current_span("predict_request"):
        if not model:
            logger.error("Model not available")
            raise HTTPException(status_code=500, detail="Model not available")
        
        text = request.text
        prediction = model.predict([text])[0]
        logger.info(f"Predicted class: {prediction}")
        return {"prediction": prediction}

@app.get("/")
def root():
    return {"message": "Document Classification API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
