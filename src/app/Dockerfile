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

# Ensure Uvicorn is installed (as a fallback)
RUN pip install --no-cache-dir uvicorn

# Set environment variable to force PyTorch to use CPU
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# Expose the FastAPI port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
