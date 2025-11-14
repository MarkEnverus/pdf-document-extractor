# PDF Document Extractor

A standalone FastAPI service for extracting structured data from PDF documents using [Docling](https://github.com/DS4SD/docling).

## Features

- **Document Extraction**: Extract text, tables, and figures from PDF documents
- **Multiple Strategies**: API-based and Docling-based extraction strategies
- **Image Analysis**: Optional AWS Bedrock integration for advanced image analysis
- **RESTful API**: FastAPI-based HTTP API with automatic OpenAPI documentation
- **Async Processing**: Built on FastAPI for high-performance async operations

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip

### Installation

```bash
# Using uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Or using pip
pip install -e ".[dev]"
```

### Running the Service

```bash
# Using uv
uv run uvicorn src.main:app --reload

# Or directly
python -m src.main
```

The service will be available at:
- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Extract Document

```bash
POST /api/v1/documents/extract
```

Extract structured data from a PDF document.

**Request Body:**
```json
{
  "file_id": "uuid",
  "file_path": "s3://bucket/path/to/document.pdf",
  "mime_type": "application/pdf"
}
```

**Response:**
```json
{
  "file_id": "uuid",
  "status": "completed",
  "pages": [
    {
      "page_number": 1,
      "text": "...",
      "tables": [...],
      "figures": [...]
    }
  ]
}
```

## Configuration

Configuration is managed through environment variables:

```bash
# Service
API_V1_STR="/api/v1"
ALLOWED_HOSTS="*"

# AWS (if using S3 or Bedrock)
AWS_REGION="us-east-1"
AWS_ACCESS_KEY_ID="..."
AWS_SECRET_ACCESS_KEY="..."

# Logging
LOG_LEVEL="INFO"

# OpenTelemetry (optional)
OTEL_TRACING_ENABLED="false"
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific tests
uv run pytest tests/services/
```

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint
uv run ruff check .

# Type checking
uv run mypy src/
```

## Project Structure

```
pdf-document-extractor/
├── src/
│   ├── api/           # FastAPI routes and endpoints
│   ├── configs/       # Configuration and settings
│   ├── services/      # Business logic and extraction services
│   ├── models/        # Pydantic models and DTOs
│   └── main.py        # FastAPI application entry point
├── lib/
│   ├── logger/        # Logging utilities
│   └── models/        # Shared data models
├── tests/             # Test suite
└── pyproject.toml     # Project dependencies and configuration
```

## Extraction Strategies

### Docling Strategy

The Docling strategy uses the Docling library for comprehensive document processing:

- Text extraction with layout preservation
- Table detection and extraction
- Figure/image extraction
- Bounding box coordinates
- Page-by-page processing

### API Strategy

Lightweight extraction strategy for simple use cases:

- Basic text extraction
- Fast processing
- Minimal dependencies

## Troubleshooting

### Common Issues

1. **Import errors for `idp_*` modules**: Some code may still reference the monorepo libraries. These need to be updated to use the local `lib/` modules.

2. **AWS credentials**: If using S3 or Bedrock features, ensure AWS credentials are properly configured.

3. **Docling GPU support**: Docling can use CUDA for GPU acceleration. Install appropriate PyTorch version for your system.

## Contributing

This is a standalone extraction of the extractor service from the genai-idp monorepo. Feel free to modify and adapt for your needs.

## License

Internal use only.

## Notes

This project was extracted from a larger monorepo and simplified for standalone use. Some features that depended heavily on the monorepo infrastructure (Kafka, complex auth) have been stubbed out or simplified. You can re-enable them as needed.
