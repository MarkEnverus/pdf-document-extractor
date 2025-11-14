from typing import Any, Literal

from pydantic_settings import BaseSettings
from pydantic import BaseModel

from lib.aws import AWSHelper, get_s3_client
from lib.kafka import KafkaConfig
from lib.logger import Logger


# Stub for PostgresConfig (can be replaced with a proper implementation if needed)
class PostgresConfig(BaseModel):
    host: str
    port: int
    database: str
    user: str
    password: str
    db_schema: str = "public"
    min_connections: int = 1
    max_connections: int = 10
    connection_timeout: float = 10.0
    debug_queries: bool = False
    application_name: str = "pdf-document-extractor"
    server_options: dict[str, Any] = {}

logger = Logger.get_logger(__name__)


class Settings(BaseSettings):
    PROJECT_NAME: str = "pdf-document-extractor"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT_NAME: Literal["local", "test", "dev", "prod"] = "local"
    EXTRACTION_ENDPOINT: str = "https://genai-dev-ue1-mu-genai-mcp-document-extraction-dev"

    # FastAPI configuration
    allowed_hosts: list[str] = ["*"]
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str = "genai-proprietary-data-extract-dev"

    # Kafka/WARPSTREAM Configuration
    KAFKA_SECRET_NAME: str = "warpstream-maestro-sasl-dev"
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_SASL_USERNAME: str = ""
    KAFKA_SASL_PASSWORD: str = ""
    KAFKA_INGEST_TOPIC: str = "genai-proprietary-data-ingest-dev"
    KAFKA_OUTPUT_TOPIC: str = "genai-proprietary-data-extract-dev"
    # Kafka timeout configuration (increased for WarpStream latency)
    KAFKA_FETCH_TIMEOUT_MS: int = 120000  # 2 minutes (was 60000)
    KAFKA_SESSION_TIMEOUT_MS: int = 60000  # 1 minute (was 45000)
    KAFKA_MAX_POLL_INTERVAL_MS: int = 600000  # 10 minutes (was 300000)
    # Additional timeout configuration for WarpStream coordinator operations
    KAFKA_REQUEST_TIMEOUT_MS: int = 180000  # 3 minutes
    KAFKA_CONNECTIONS_MAX_IDLE_MS: int = 540000  # 9 minutes

    # Kafka Processing Configuration
    MAX_CONCURRENT_PROCESSING: int = 1

    # PostgreSQL Configuration (for ingestion status tracking)
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "maestro_db"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_SCHEMA: str = "public"

    # OpenTelemetry Configuration
    OTEL_EXPORTER_OTLP_ENDPOINT: str = ""
    OTEL_TRACING_ENABLED: bool = False

    # Docling Configuration
    DOCLING_DEFAULT_PROVIDER: str = "local"
    DOCLING_DEFAULT_OUTPUT_FORMAT: str = "markdown"
    DOCLING_ENABLE_OCR: bool = True
    DOCLING_ENABLE_TABLE_STRUCTURE: bool = True
    DOCLING_ENABLE_FIGURE_EXTRACTION: bool = True

    # Page extraction configuration
    DOCLING_SINGLE_PAGE_MODE: bool = False
    DOCLING_EXTRACT_PAGES: str = "all"

    # Bedrock Image Analysis Configuration
    BEDROCK_IMAGE_ANALYSIS_MODEL: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    BEDROCK_IMAGE_ANALYSIS_TIMEOUT: int = 30
    BEDROCK_IMAGE_ANALYSIS_MAX_TOKENS: int = 1024
    BEDROCK_IMAGE_ANALYSIS_TEMPERATURE: float = 0.1

    # Status Tracker Configuration
    STATUS_ENABLE_KAFKA_PUBLISHING: bool = True

    @property
    def fastapi_kwargs(self) -> dict[str, Any]:
        """Get FastAPI initialization kwargs."""
        return {}

    def get_kafka_config(self) -> KafkaConfig:
        """Get Kafka configuration with environment-appropriate authentication."""
        aws_helper = AWSHelper()
        env = self.ENVIRONMENT_NAME
        if env == "local" or env == "test":
            user = None
            password = None
            security_protocol = "PLAINTEXT"
            sasl_mechanism = None
        else:
            kafka_secret = aws_helper.get_secret(self.KAFKA_SECRET_NAME)
            user = kafka_secret["SASL_USERNAME"]
            password = kafka_secret["SASL_PASSWORD"]
            security_protocol = "SASL_PLAINTEXT"
            sasl_mechanism = "SCRAM-SHA-512"

        return KafkaConfig(
            bootstrap_servers=self.KAFKA_BOOTSTRAP_SERVERS,
            security_protocol=security_protocol,
            sasl_mechanism=sasl_mechanism,
            sasl_plain_username=user,
            sasl_plain_password=password,
            consumer_group_id=f"genai-maestro-idp-extraction-{self.ENVIRONMENT_NAME}",
            # Auto-commit configuration (TODO: Add DLQ for failed messages)
            enable_auto_commit=False,  # Manual commits for at-least-once delivery
            auto_offset_reset="earliest" if self.ENVIRONMENT_NAME == "local" else "latest",
            fetch_timeout_ms=self.KAFKA_FETCH_TIMEOUT_MS,
            session_timeout_ms=self.KAFKA_SESSION_TIMEOUT_MS,
            max_poll_interval_ms=self.KAFKA_MAX_POLL_INTERVAL_MS,
            request_timeout_ms=self.KAFKA_REQUEST_TIMEOUT_MS,
            connections_max_idle_ms=self.KAFKA_CONNECTIONS_MAX_IDLE_MS,
        )

    def get_s3_client(self) -> Any:
        """Get S3 client for file operations."""
        return get_s3_client()

    def get_postgres_config(self) -> PostgresConfig:
        """Get PostgreSQL configuration based on environment."""
        logger.info(
            "Retrieving PostgreSQL configuration",
            extra={"environment": self.ENVIRONMENT_NAME},
        )
        aws_helper = AWSHelper()
        if self.ENVIRONMENT_NAME not in ["local", "test"]:
            logger.info(
                "Fetching DB credentials from AWS Secrets Manager",
                extra={"secret_name": f"maestro-app-{self.ENVIRONMENT_NAME}"},
            )
            db_secret = aws_helper.get_secret(f"maestro-app-{self.ENVIRONMENT_NAME}")
            host = db_secret["host"]
            port = db_secret["port"]
            db = db_secret["db_name"]
            user = db_secret["username"]
            password = db_secret["password"]
        else:
            logger.info(
                "Using local PostgreSQL configuration",
                extra={
                    "database": self.POSTGRES_DB,
                    "schema": self.POSTGRES_SCHEMA,
                },
            )
            host = self.POSTGRES_HOST
            port = self.POSTGRES_PORT
            db = self.POSTGRES_DB
            user = self.POSTGRES_USER
            password = self.POSTGRES_PASSWORD

        logger.info(
            "PostgreSQL configuration retrieved successfully",
            extra={
                "database": db,
                "schema": self.POSTGRES_SCHEMA,
            },
        )
        return PostgresConfig(
            host=host,
            port=port,
            database=db,
            user=user,
            password=password,
            db_schema=self.POSTGRES_SCHEMA,
            min_connections=1,
            max_connections=10,
            connection_timeout=10.0,
            debug_queries=False,
            application_name="maestro-idp-extractor",
            server_options={
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 3,
                "autocommit": True,
                "prepare_threshold": 0,
            },
        )


try:
    settings = Settings()
except SystemExit as e:
    logger.error(f"Failed to load settings: {str(e)}")
    raise
