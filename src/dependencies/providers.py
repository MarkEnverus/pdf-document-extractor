"""Dependency injection providers for FastAPI."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from lib.logger import Logger
from src.configs.settings import settings
from src.interfaces.image_analysis import ImageAnalysisService
from src.interfaces.status_tracker import StatusTracker
from src.interfaces.storage import StorageService
from src.models.image_analysis_models import BedrockAnalysisConfig
from src.services.asset_storage_service import AssetStorageService
from src.services.bedrock_image_analysis_service import BedrockImageAnalysisService
from src.services.extraction_api_client import ExtractionApiClient as ConcreteExtractionApiClient
from src.services.extractors.api.api_extraction_processor import ApiExtractionProcessor
from src.services.extractors.docling.docling_strategy_processor import DoclingStrategyProcessor
from src.services.kafka_message_handler import KafkaMessageHandler
from src.services.processing_config_manager import ProcessingConfigManager
from src.services.processing_orchestrator import ProcessingOrchestrator
from src.services.s3_service import S3Service
from src.services.status_tracker import StatusTracker as ConcreteStatusTracker


def _is_test_environment() -> bool:
    """Check if we're running in test environment."""
    return settings.ENVIRONMENT_NAME.lower() == "test"


# Service factories - conditional caching based on environment
def get_logger() -> Logger:
    """Get logger instance."""
    if _is_test_environment():
        # Return new instance for tests to avoid caching issues
        return Logger.get_logger(__name__)
    else:
        # Use cached instance for production
        return _get_cached_logger()


@lru_cache()
def _get_cached_logger() -> Logger:
    """Get cached logger instance (production only)."""
    return Logger.get_logger(__name__)


def get_storage_service() -> StorageService:
    """Get storage service instance."""
    if _is_test_environment():
        # Return new instance for tests with no external dependencies
        return S3Service(s3_client=None, settings_provider=None)
    else:
        # Use cached instance for production
        return _get_cached_storage_service()


@lru_cache()
def _get_cached_storage_service() -> StorageService:
    """Get cached storage service instance (production only)."""
    return S3Service(s3_client=None, settings_provider=settings)


# get_document_processor removed - DoclingService functionality moved to DoclingStrategyProcessor


def get_status_tracker() -> ConcreteStatusTracker:
    """Get status tracker instance."""
    if _is_test_environment():
        # Return new instance for tests - Kafka always enabled but may fail gracefully in tests
        return ConcreteStatusTracker()
    else:
        # Use cached instance for production
        return _get_cached_status_tracker()


@lru_cache()
def _get_cached_status_tracker() -> ConcreteStatusTracker:
    """Get cached status tracker instance (production only)."""
    return ConcreteStatusTracker()


def get_extraction_api_client() -> ConcreteExtractionApiClient:
    """Get extraction API client instance."""
    if _is_test_environment():
        # Return new instance for tests
        return _create_extraction_api_client()
    else:
        # Use cached instance for production
        return _get_cached_extraction_api_client()


def _create_extraction_api_client() -> ConcreteExtractionApiClient:
    """Create extraction API client instance."""
    try:
        if settings.EXTRACTION_ENDPOINT is None or settings.EXTRACTION_ENDPOINT == "":
            raise ValueError("EXTRACTION_ENDPOINT is not set or is empty - configuration failed")

        return ConcreteExtractionApiClient(endpoint=settings.EXTRACTION_ENDPOINT, timeout=300)
    except Exception as e:
        # Import here to avoid circular dependencies
        from src.utils.critical_failure_handler import CriticalFailureHandler

        CriticalFailureHandler.handle_critical_failure(
            e, "ExtractionApiClient", "Failed to create extraction API client"
        )
        # This should never be reached due to sys.exit() in handle_critical_failure
        raise RuntimeError("Critical failure handler failed to exit application") from e


@lru_cache()
def _get_cached_extraction_api_client() -> ConcreteExtractionApiClient:
    """Get cached extraction API client instance (production only)."""
    return _create_extraction_api_client()


def create_processing_orchestrator() -> ProcessingOrchestrator:
    """Create processing orchestrator with regular dependency injection (non-FastAPI)."""
    # Create all required dependencies
    config_manager = get_processing_config_manager()
    status_tracker = get_status_tracker()
    storage_service = get_storage_service()
    extraction_client = get_extraction_api_client()

    # Create asset storage service (no longer needs DoclingService)
    image_analysis_service = get_image_analysis_service()
    asset_storage = AssetStorageService(storage_service=storage_service, image_analysis_service=image_analysis_service)

    # Create processing strategies
    api_processor = ApiExtractionProcessor(
        extraction_client=extraction_client, storage_service=storage_service, status_tracker=status_tracker
    )

    docling_processor = DoclingStrategyProcessor(
        storage_service=storage_service,
        status_tracker=status_tracker,
        config_manager=config_manager,
        asset_storage=asset_storage,
    )

    # Create orchestrator with status_tracker for heartbeat updates
    return ProcessingOrchestrator(
        docling_processor=docling_processor, api_processor=api_processor, status_tracker=status_tracker
    )


def get_kafka_message_handler() -> KafkaMessageHandler:
    """Get Kafka message handler instance."""
    status_tracker = get_status_tracker()
    orchestrator = create_processing_orchestrator()
    extraction_client = get_extraction_api_client()
    return KafkaMessageHandler(
        status_tracker=status_tracker, orchestrator=orchestrator, extraction_client=extraction_client
    )


# Note: Kafka message handler can be used by background services
# HTTP endpoints use ProcessingOrchestrator directly


def get_processing_config_manager() -> ProcessingConfigManager:
    """Get processing config manager instance."""
    if _is_test_environment():
        return ProcessingConfigManager()
    else:
        return _get_cached_processing_config_manager()


@lru_cache()
def _get_cached_processing_config_manager() -> ProcessingConfigManager:
    """Get cached processing config manager instance (production only)."""
    return ProcessingConfigManager()


def get_image_analysis_service() -> ImageAnalysisService:
    """Get Bedrock image analysis service instance."""
    config = BedrockAnalysisConfig(
        model_id=settings.BEDROCK_IMAGE_ANALYSIS_MODEL,
        timeout_seconds=settings.BEDROCK_IMAGE_ANALYSIS_TIMEOUT,
        max_tokens=settings.BEDROCK_IMAGE_ANALYSIS_MAX_TOKENS,
        temperature=settings.BEDROCK_IMAGE_ANALYSIS_TEMPERATURE,
    )

    if _is_test_environment():
        # In test environment, return service (may use mock Bedrock calls)
        return BedrockImageAnalysisService(config=config, aws_region=settings.AWS_REGION)
    else:
        return _get_cached_image_analysis_service(config, settings.AWS_REGION)


def _get_cached_image_analysis_service(config: BedrockAnalysisConfig, aws_region: str) -> ImageAnalysisService:
    """Get cached Bedrock image analysis service instance (production only)."""
    return BedrockImageAnalysisService(config=config, aws_region=aws_region)


def get_asset_storage_service(
    storage_service: StorageService = Depends(get_storage_service),
    image_analysis_service: ImageAnalysisService = Depends(get_image_analysis_service),
) -> AssetStorageService:
    """Get asset storage service instance."""
    if _is_test_environment():
        return AssetStorageService(storage_service=storage_service, image_analysis_service=image_analysis_service)
    else:
        return _get_cached_asset_storage_service(storage_service, image_analysis_service)


def _get_cached_asset_storage_service(
    storage_service: StorageService, image_analysis_service: ImageAnalysisService
) -> AssetStorageService:
    """Get asset storage service instance (caching handled by FastAPI DI)."""
    return AssetStorageService(storage_service=storage_service, image_analysis_service=image_analysis_service)


def get_api_extraction_processor(
    extraction_client: ConcreteExtractionApiClient = Depends(get_extraction_api_client),
    storage_service: StorageService = Depends(get_storage_service),
    status_tracker: ConcreteStatusTracker = Depends(get_status_tracker),
) -> ApiExtractionProcessor:
    """Get API extraction processor instance."""
    if _is_test_environment():
        return ApiExtractionProcessor(
            extraction_client=extraction_client, storage_service=storage_service, status_tracker=status_tracker
        )
    else:
        return _get_cached_api_extraction_processor(
            extraction_client,
            storage_service,
            status_tracker,
        )


def _get_cached_api_extraction_processor(
    extraction_client: ConcreteExtractionApiClient,
    storage_service: StorageService,
    status_tracker: ConcreteStatusTracker,
) -> ApiExtractionProcessor:
    """Get API extraction processor instance (caching handled by FastAPI DI)."""
    return ApiExtractionProcessor(
        extraction_client=extraction_client, storage_service=storage_service, status_tracker=status_tracker
    )


def get_docling_strategy_processor(
    storage_service: StorageService = Depends(get_storage_service),
    status_tracker: ConcreteStatusTracker = Depends(get_status_tracker),
    config_manager: ProcessingConfigManager = Depends(get_processing_config_manager),
    asset_storage: AssetStorageService = Depends(get_asset_storage_service),
) -> DoclingStrategyProcessor:
    """Get Docling strategy processor instance."""
    if _is_test_environment():
        return DoclingStrategyProcessor(
            storage_service=storage_service,
            status_tracker=status_tracker,
            config_manager=config_manager,
            asset_storage=asset_storage,
        )
    else:
        return _get_cached_docling_strategy_processor(
            storage_service,
            status_tracker,
            config_manager,
            asset_storage,
        )


def _get_cached_docling_strategy_processor(
    storage_service: StorageService,
    status_tracker: ConcreteStatusTracker,
    config_manager: ProcessingConfigManager,
    asset_storage: AssetStorageService,
) -> DoclingStrategyProcessor:
    """Get Docling strategy processor instance (caching handled by FastAPI DI)."""
    return DoclingStrategyProcessor(
        storage_service=storage_service,
        status_tracker=status_tracker,
        config_manager=config_manager,
        asset_storage=asset_storage,
    )


def get_processing_orchestrator(
    docling_processor: DoclingStrategyProcessor = Depends(get_docling_strategy_processor),
    api_processor: ApiExtractionProcessor = Depends(get_api_extraction_processor),
    status_tracker: ConcreteStatusTracker = Depends(get_status_tracker),
) -> ProcessingOrchestrator:
    """Get processing orchestrator instance."""
    if _is_test_environment():
        return ProcessingOrchestrator(
            docling_processor=docling_processor, api_processor=api_processor, status_tracker=status_tracker
        )
    else:
        return _get_cached_processing_orchestrator(docling_processor, api_processor, status_tracker)


def _get_cached_processing_orchestrator(
    docling_processor: DoclingStrategyProcessor,
    api_processor: ApiExtractionProcessor,
    status_tracker: ConcreteStatusTracker,
) -> ProcessingOrchestrator:
    """Get processing orchestrator instance (caching handled by FastAPI DI)."""
    return ProcessingOrchestrator(
        docling_processor=docling_processor, api_processor=api_processor, status_tracker=status_tracker
    )


# Type aliases for dependency injection (FastAPI only)
# Only include services that FastAPI endpoints actually need
ProcessingOrchestratorDep = Annotated[ProcessingOrchestrator, Depends(get_processing_orchestrator)]
LoggerDep = Annotated[Logger, Depends(get_logger)]


# Mock factories for testing
def get_mock_storage_service(mock_service: StorageService) -> StorageService:
    """Factory for injecting mock storage service in tests."""
    return mock_service


def get_mock_document_processor(mock_processor: DoclingStrategyProcessor) -> DoclingStrategyProcessor:
    """Factory for injecting mock document processor in tests."""
    return mock_processor


def get_mock_status_tracker(mock_tracker: StatusTracker) -> StatusTracker:
    """Factory for injecting mock status tracker in tests."""
    return mock_tracker
