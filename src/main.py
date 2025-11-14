import os

from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

from src.configs.settings import settings
from src.api.main import api_router
from lib.logger import Logger
from src.services.kafka_background_service import kafka_lifespan


logger = Logger.get_logger(os.path.basename(__file__))


def custom_generate_unique_id(route: APIRoute) -> str:
    if route.tags and len(route.tags) > 0:
        return f"{route.tags[0]}-{route.name}"
    else:
        return f"default-{route.name}"


def get_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    # Create FastAPI application
    fastapi_kwargs = {
        **settings.fastapi_kwargs,
        "title": "PDF Document Extractor",
        "description": "Standalone PDF document extraction service using Docling",
        "version": "1.0.0",
        "api_prefix": "",
        "root_path": "/",
        "generate_unique_id_function": custom_generate_unique_id,
        "swagger_ui_parameters": {"syntaxHighlight.theme": "obsidian"},
        "lifespan": kafka_lifespan,
    }
    application = FastAPI(**fastapi_kwargs)

    # OpenTelemetry tracing setup
    resource = Resource.create({"service.name": "genai-maestro-idp-extraction"})
    tracer_provider = TracerProvider(resource=resource)

    # Get OTEL endpoint from settings (already has environment fallback)
    endpoint = settings.OTEL_EXPORTER_OTLP_ENDPOINT

    # Configure OTLP exporter only if endpoint is provided and tracing is enabled
    if endpoint and settings.OTEL_TRACING_ENABLED:
        # Set environment variables for LangChain
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        span_processor = BatchSpanProcessor(exporter)
        tracer_provider.add_span_processor(span_processor)

    trace.set_tracer_provider(tracer_provider)
    RequestsInstrumentor().instrument()
    # PsycopgInstrumentor().instrument()
    # Botocore instrumentation is currently disabled because this service does not use AWS/botocore clients yet.
    # Re-enable when AWS integration is added or if botocore usage is introduced. (Review by 2024-12-01)
    # BotocoreInstrumentor().instrument()
    FastAPIInstrumentor.instrument_app(application, tracer_provider=tracer_provider)

    # Add middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_hosts,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_middleware(GZipMiddleware, minimum_size=1000)

    # Include API routers
    application.include_router(api_router, prefix=settings.API_V1_STR)

    logger.info("PDF Document Extractor application initialized successfully")
    return application


app = get_application()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", server_header=False)
