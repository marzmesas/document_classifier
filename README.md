# Document Classifier

A production-ready application for classifying text documents into predefined categories. Built with RoBERTa transformer models, FastAPI, Streamlit, and includes comprehensive observability using OpenTelemetry, MLflow, Prometheus, Loki, Jaeger, and Grafana.

## Features

- **ML Model**: Transformer-based document classification (RoBERTa + MLP)
- **API Backend**: FastAPI for efficient and type-safe API endpoints
- **User Interface**: Streamlit-based interactive UI for easy document classification
- **Experiment Tracking**: MLflow for tracking model training, parameters, and performance
- **Observability**: Complete telemetry with metrics, traces, and logs
- **Containerized**: Docker and Docker Compose setup for easy deployment
- **Testing**: Comprehensive test suite for all components

## Architecture

The application consists of:

- **FastAPI Backend**: Serves the ML model and provides prediction endpoints
- **Streamlit Frontend**: User interface for document submission and classification
- **MLflow**: Experiment tracking, model versioning, and registry
- **OpenTelemetry**: End-to-end tracing, metrics, and logging
- **Observability Stack**:
  - Prometheus for metrics collection
  - Loki for log aggregation
  - Jaeger for distributed tracing
  - Grafana for visualization

## Prerequisites

- Python 3.11+
- Poetry for dependency management
- Git LFS for model files
- Docker and Docker Compose (for containerized deployment)

## Installation

1. Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/yourusername/document-classifier.git
cd document-classifier
```

2. Install Git LFS and pull model files:

```bash
# Install Git LFS
# On macOS:
brew install git-lfs
# On Ubuntu:
sudo apt-get install git-lfs
# On Windows with Chocolatey:
choco install git-lfs

# Initialize Git LFS
git lfs install

# Pull the model files
git lfs pull
```

3. Install dependencies with Poetry:

```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install
```

## Running the Application

### Option 1: Docker Compose (Recommended for Production)

Run the full application stack with observability:

```bash
docker-compose up --build
```

Access:
- Streamlit UI: http://localhost:8501
- FastAPI Swagger UI: http://localhost:8000/docs
- Grafana Dashboard: http://localhost:3000
- Jaeger UI: http://localhost:16686
- Prometheus: http://localhost:9090

### Option 2: Local Testing Mode

For local development without the observability stack:

```bash
# Set TESTING environment variable
export TESTING=True

# Run the test script
python src/classifier_app_local_test.py
```

Access:
- Streamlit UI: http://localhost:8501
- FastAPI Swagger UI: http://localhost:8000/docs


## Project Structure

```
document-classifier/
├── notebooks/               # Jupyter notebooks
├── src/
│   ├── app/                 # FastAPI application
│   │   ├── api.py           # API endpoints
│   │   ├── Dockerfile       # Docker configuration for app
│   │   └── start_app_services.py  # Script to start classifier app
│   ├── frontend/            # Streamlit frontend
│   │   └── streamlit_app.py # Streamlit interface with FastAPI backend
│   ├── models/
│   │   ├── final_model/     # Saved model files
│   │   ├── base_model.py    # Model architecture
│   │   ├── evaluate.py      # Model evaluation utilities
│   │   ├── train.py         # Model training utilities
│   │   └── utils.py         # Model utility functions
│   ├── datasets/            # Dataset handling
│   ├── workflows/           # End-to-end workflows
│   │   ├── inference.py     # Inference pipeline
│   │   └── training.py      # Training pipeline
│   └── observability/       # Telemetry setup
│       └── logging.py       # OpenTelemetry logging
├── tests/                   # Test suite
├── docker-compose.yaml      # Docker Compose configuration
├── otel-collector-config.yaml  # OpenTelemetry Collector config
├── prometheus.yml           # Prometheus configuration
├── pyproject.toml           # Poetry dependencies
├── .gitattributes           # Git LFS configuration
├── .gitignore               # Git ignore configuration
├── .coveragerc              # Coverage configuration
├── config.yaml              # Application configuration
└── README.md                # Project documentation
```

## Running Tests

Execute the test suite:

```bash
# Run all tests
poetry run pytest

# Run tests with coverage report
poetry run pytest --cov=src tests/
```

## Training a New Model

To train a new classification model with experiment tracking:

```bash
# Run the training workflow
python -m src.workflows.training
```

All training runs are automatically tracked in MLflow, including:
- Model parameters
- Training and validation metrics
- Model artifacts

## MLflow Experiment Tracking

When you run training, MLflow will generate a local tracking database and artifacts directory (these are gitignored).

View experiment results and compare model runs:

```bash
# Start the MLflow UI
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

Access the MLflow UI at http://localhost:5000 to:
- Compare different training runs
- View learning curves
- Examine model parameters
- Download saved models and artifacts

### Setting Up a Shared MLflow Server (Optional)

For team collaboration, consider setting up a shared MLflow tracking server:

1. Update the `mlflow.tracking_uri` in `config.yaml` to point to your shared server
2. Make sure all team members use the same tracking URI

```yaml
# Example for a shared MLflow server in config.yaml
mlflow:
  tracking_uri: "http://shared-mlflow-server:5000"
  experiment_name: "document-classifier"
```

## License

[TBD]

## Contributing

[TBD]
