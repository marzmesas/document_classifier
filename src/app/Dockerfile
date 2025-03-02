# First stage - Builder
FROM python:3.12-slim AS builder

WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy only poetry files first to improve caching
COPY pyproject.toml poetry.lock ./

# Configure Poetry to install dependencies system-wide
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --without dev --no-root --no-interaction --verbose

# Second stage - Final runtime image
FROM python:3.12-slim

WORKDIR /app

# Copy installed dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.12/ /usr/local/lib/python3.12/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application source code
COPY . .

# Ensure Uvicorn and Streamlit are installed (as a fallback)
RUN pip install --no-cache-dir uvicorn streamlit

# Set environment variable to force PyTorch to use CPU
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# Expose both the FastAPI and Streamlit ports
EXPOSE 8000 8501

# Run the app services startup script
CMD ["python", "src/app/start_app_services.py"]
