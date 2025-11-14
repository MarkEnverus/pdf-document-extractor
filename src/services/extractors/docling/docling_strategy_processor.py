"""
Docling strategy processor implementing DocumentProcessingStrategy interface.

This contains the complete Docling processing implementation.
"""

import hashlib
import importlib.util
import json
import logging
import os
import tempfile
import threading
import time
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter

# Configure PyTorch environment variables
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # For macOS Metal Performance Shaders fallback

# Suppress PyTorch DataLoader pin_memory warning when no GPU is available
warnings.filterwarnings("ignore", message=".*pin_memory.*no accelerator.*", category=UserWarning)

# Docling imports
from docling.datamodel.pipeline_options import TableFormerMode, TableStructureOptions
from docling.document_converter import PowerpointFormatOption
from lib.models.mime_type import IngestionMimeType
from lib.logger import Logger
from lib.models.extraction_models import (
    ExtractionType,
    FigureReference,
    PageExtractionResult,
    PageMetadata,
    TableReference,
)
from lib.models.mime_type_utils import MimeTypeUtils
from src.interfaces.document_processor import AbstractProcessingStrategy
from src.interfaces.storage import StorageService
from src.models.docling_models import (
    DoclingBatchResult,
    DoclingConfig,
    DoclingDocumentResult,
    DoclingOutputFormatEnum,
    DoclingRequest,
)
from src.models.extraction_models import ExtractedFigure, ExtractedTable
from src.models.status_models import IngestionStepStatus
from src.services.asset_storage_service import AssetStorageService
from src.services.processing_config_manager import ProcessingConfigManager
from src.services.status_tracker import StatusTracker
from src.services.storage_utils import (
    generate_page_text_s3_path,
    generate_results_json_s3_path,
    store_text_file_to_s3,
)

logger = Logger.get_logger(__name__)


class DoclingNoiseFilter(logging.Filter):
    """Filter to suppress noisy Docling logs during processing."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Filter out torch-related warnings during Docling processing
        noisy_messages = [
            "You are using a model of type ",
            "Unable to get page dimensions",
            "Some weights of the model checkpoint",
            "pytorch_model.bin",
            "Loading checkpoint shards",
            "Converting Docling document",
            "pin_memory",
            "ColumnMergeMode",
            "docling_output",
            "model=test-model",
            ".bin",
            "Load pretrained classifier",
            "[WARNING]",
        ]

        return not any(msg in record.getMessage() for msg in noisy_messages)


def _configure_docling_logging() -> None:
    """Configure logging to suppress noisy Docling messages."""
    # Apply noise filter to relevant loggers
    loggers_to_filter = [
        "docling",
        "torch",
        "transformers",
        "huggingface_hub",
        "deepsearch_glm",
        "PIL",
        "accelerate",
    ]

    noise_filter = DoclingNoiseFilter()

    for logger_name in loggers_to_filter:
        target_logger = logging.getLogger(logger_name)
        if not any(isinstance(f, DoclingNoiseFilter) for f in target_logger.filters):
            target_logger.addFilter(noise_filter)
            target_logger.setLevel(logging.ERROR)  # Suppress INFO and DEBUG


# Configure logging once when module is loaded
_configure_docling_logging()


class DoclingStrategyProcessor(AbstractProcessingStrategy):
    """
    Docling-based document processing strategy.

    This strategy uses local Docling processing for document extraction,
    suitable for:
    - Structured documents (PDF, DOCX, PPTX)
    - Documents requiring detailed structure extraction
    - Local processing requirements
    - Advanced figure and table extraction
    """

    def __init__(
        self,
        storage_service: StorageService,
        status_tracker: StatusTracker,
        config_manager: ProcessingConfigManager,
        asset_storage: AssetStorageService,
    ):
        """
        Initialize DoclingStrategyProcessor.

        Args:
            storage_service: Storage service for storing results
            status_tracker: Service for updating processing status
            config_manager: Service for optimizing processing configurations
            asset_storage: Service for storing extracted figures and tables
        """
        super().__init__("docling")
        self.storage_service = storage_service
        self.status_tracker = status_tracker
        self.config_manager = config_manager
        self.asset_storage = asset_storage
        self.logger = logger

        # Shared DocumentConverter for model reuse across documents
        # This prevents loading 6GB of AI models for every document
        self._doc_converter: Optional["DocumentConverter"] = None
        self._converter_lock = threading.Lock()
        self._current_config_hash: Optional[str] = None

        # Configure PyTorch device (GPU if available, otherwise CPU)
        self._configure_pytorch_device()

    def _configure_pytorch_device(self) -> None:
        """
        Configure PyTorch to use GPU if available, otherwise fall back to CPU.

        This enables GPU acceleration for Docling's AI models (TableFormer, OCR, Vision)
        when CUDA-capable GPUs are present, providing 5-10x faster processing.
        """
        try:
            import torch
            from src.configs.settings import settings

            # Check if user explicitly disabled GPU via environment variable
            use_gpu = os.environ.get("USE_GPU", "true").lower() in ("true", "1", "yes")

            # Detect CUDA availability
            cuda_available = hasattr(torch, "cuda") and torch.cuda.is_available()

            if use_gpu and cuda_available:
                # GPU is available - use it!
                device = "cuda"
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "unknown"

                # Configure for GPU processing
                torch.set_default_device("cuda")

                # Enable pinned memory for faster CPU-GPU data transfers
                os.environ["TORCH_PIN_MEMORY"] = "true"

                # Get CUDA version safely
                cuda_version = "unknown"
                try:
                    if hasattr(torch, "version") and hasattr(torch.version, "cuda"):
                        version = torch.version.cuda
                        cuda_version = version if version is not None else "unknown"
                except Exception:
                    pass

                self.logger.info(
                    "🚀 PyTorch configured for GPU acceleration",
                    device=device,
                    torch_version=torch.__version__,
                    gpu_count=gpu_count,
                    gpu_name=gpu_name,
                    cuda_version=cuda_version,
                )
            else:
                # Fall back to CPU
                device = "cpu"
                torch.set_default_device("cpu")

                # Disable pinned memory for CPU-only processing
                os.environ["TORCH_PIN_MEMORY"] = "false"

                # Optimize for CPU inference
                torch.set_num_threads(4)  # Reasonable thread count for document processing

                # Disable MKL-DNN optimizations that can cause issues
                if hasattr(torch.backends, "mkldnn"):
                    torch.backends.mkldnn.enabled = False  # type: ignore[assignment]

                reason = "GPU disabled by USE_GPU=false" if not use_gpu else "No CUDA-capable GPU detected"
                self.logger.info(
                    "PyTorch configured for CPU processing",
                    device=device,
                    torch_version=torch.__version__,
                    reason=reason,
                )

            # Common configuration for both CPU and GPU
            torch.set_grad_enabled(False)  # Disable gradients for inference

        except ImportError:
            self.logger.warning("PyTorch not available - some Docling features may not work")
        except Exception as e:
            self.logger.warning("Failed to configure PyTorch device", error=str(e), error_type=type(e).__name__)

    def _get_or_create_converter(self, config: DoclingConfig) -> "DocumentConverter":
        """
        Get existing DocumentConverter or create new one if config changed.

        This ensures AI models are loaded ONCE and reused across all documents,
        drastically reducing memory usage from 16GB → 8GB.

        Thread-safe: Uses lock to prevent race conditions.
        Config-aware: Only reloads if processing config actually changes.

        Args:
            config: Docling configuration

        Returns:
            Cached or newly created DocumentConverter
        """
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        # Create hash of config to detect changes
        config_dict = config.model_dump()
        config_hash = hashlib.md5(str(config_dict).encode()).hexdigest()

        # Check if we need to create new converter
        with self._converter_lock:
            if self._doc_converter is None or self._current_config_hash != config_hash:
                self.logger.info(
                    "Creating DocumentConverter - models will be loaded into memory",
                    config_hash=config_hash,
                    previous_hash=self._current_config_hash,
                    enable_ocr=config.enable_ocr,
                    enable_table_structure=config.enable_table_structure,
                    enable_figure_extraction=config.enable_figure_extraction,
                )

                # Configure pipeline options
                pipeline_options = PdfPipelineOptions(
                    do_ocr=config.enable_ocr,
                    do_table_structure=config.enable_table_structure,
                    table_structure_options=TableStructureOptions(
                        do_cell_matching=True,
                    ),
                    generate_page_images=config.enable_figure_extraction,
                    generate_picture_images=config.enable_figure_extraction,
                )

                # Create new converter - this loads all AI models (6GB+)
                self._doc_converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                        InputFormat.PPTX: PowerpointFormatOption(pipeline_options=pipeline_options),
                    }
                )
                self._current_config_hash = config_hash

                self.logger.info(
                    "DocumentConverter created and cached for reuse across documents", config_hash=config_hash
                )
            else:
                self.logger.debug("Reusing existing DocumentConverter - models already loaded", config_hash=config_hash)

        # At this point, _doc_converter is guaranteed to be set
        assert self._doc_converter is not None, "DocumentConverter should have been created"
        return self._doc_converter

    async def process_documents(
        self,
        request_id: str,
        project_id: str,
        upload_id: str,
        document_urls: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process documents using Docling strategy.

        Args:
            request_id: Unique identifier for this processing request
            project_id: Project identifier (required for all logs)
            upload_id: Upload identifier - this is the key for tracking (required for all logs)
            document_urls: List of document URLs to process
            metadata: Optional metadata containing processing hints

        Returns:
            Dictionary containing Docling processing results
        """
        try:
            self.logger.info(
                "Starting Docling processing",
                request_id=request_id,
                project_id=project_id,
                upload_id=upload_id,
                document_count=len(document_urls),
                strategy="docling",
            )

            # Determine file type early for status updates
            provided_file_type = metadata.get("file_type") if metadata else None
            file_type = (
                MimeTypeUtils.determine_mime_type(document_urls[0], provided_file_type) if document_urls else "unknown"
            )

            # Basic validation - ensure we have documents
            if not document_urls:
                return self._create_failure_result(
                    request_id=request_id,
                    document_urls=document_urls,
                    error_message="No documents provided for processing",
                    project_id=project_id,
                    upload_id=upload_id,
                )

            # Update status to processing
            await self.status_tracker.update_status(
                upload_id=upload_id,
                status=IngestionStepStatus.PROCESSING,
                project_id=project_id,
                request_id=request_id,
                file_type=file_type,
                message="Starting Docling processing",
            )

            # Create optimized Docling request
            docling_request = self._create_docling_request(
                request_id=request_id,
                project_id=project_id,
                upload_id=upload_id,
                document_urls=document_urls,
                metadata=metadata,
            )

            # Process documents with Docling directly (no more wrapper service)
            docling_result = self._process_documents_batch(docling_request)

            # Process and store results
            return await self._handle_docling_results(
                request_id=request_id,
                document_urls=document_urls,
                docling_result=docling_result,
                project_id=project_id,
                upload_id=upload_id,
                file_type=file_type,
            )

        except Exception as e:
            self.logger.error(
                "Unexpected error in Docling processing",
                request_id=request_id,
                document_count=len(document_urls),
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )

            # project_id and upload_id already available as parameters

            # Determine file type for status update
            provided_file_type = metadata.get("file_type") if metadata else None
            file_type = (
                MimeTypeUtils.determine_mime_type(document_urls[0], provided_file_type) if document_urls else "unknown"
            )

            # Update status to failed with error metadata
            error_metadata = {
                "error_type": type(e).__name__,
                "error_details": str(e),
                "hard_failure": False,  # General exceptions are soft failures by default
            }

            await self.status_tracker.update_status(
                upload_id=upload_id,
                status=IngestionStepStatus.FAILURE,
                project_id=project_id,
                request_id=request_id,
                file_type=file_type,
                message=f"Docling processing failed: {str(e)}",
                metadata=error_metadata,
            )

            return self._create_failure_result(
                request_id=request_id,
                document_urls=document_urls,
                error_message=f"Processing failed: {str(e)}",
                project_id=project_id,
                upload_id=upload_id,
            )

    def _process_documents_batch(self, request: DoclingRequest) -> DoclingBatchResult:
        """
        Process documents using Docling with the specified configuration.

        Args:
            request: Docling processing request

        Returns:
            DoclingBatchResult with processing results
        """
        start_time = time.time()

        self.logger.info(
            "Starting Docling batch processing",
            request_id=request.request_id,
            project_id=request.project_id,
            upload_id=request.upload_id,
            document_count=len(request.document_urls),
            output_format=request.config.output_format.value,
            enable_ocr=request.config.enable_ocr,
            enable_table_structure=request.config.enable_table_structure,
            enable_figure_extraction=request.config.enable_figure_extraction,
            processing_mode="local",
        )

        results = []
        successful_count = 0
        failed_count = 0

        # Process each document using local Docling only
        for document_url in request.document_urls:
            try:
                result = self._process_document_local(
                    document_url,
                    request.config,
                    request.request_id,
                    request.upload_id,
                    request.project_id,
                )

                results.append(result)

                if result.success:
                    successful_count += 1
                else:
                    failed_count += 1

            except Exception as e:
                self.logger.error(
                    "Failed to process document",
                    document_url=document_url,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    request_id=request.request_id,
                    project_id=request.project_id,
                    upload_id=request.upload_id,
                )

                error_result = DoclingDocumentResult(document_url=document_url, success=False, error_message=str(e))
                results.append(error_result)
                failed_count += 1

        total_time = time.time() - start_time
        overall_success = successful_count > 0

        return DoclingBatchResult(
            request_id=request.request_id,
            success=overall_success,
            total_documents=len(request.document_urls),
            successful_documents=successful_count,
            failed_documents=failed_count,
            results=results,
            total_processing_time_seconds=total_time,
            config_used=request.config,
        )

    def _process_document_local(
        self, document_url: str, config: DoclingConfig, request_id: str, upload_id: str, project_id: str
    ) -> DoclingDocumentResult:
        """Process a document using local Docling."""
        start_time = time.time()

        try:
            # Download document to temporary file
            temp_file_path = self._download_document(document_url, request_id, upload_id, project_id)

            try:
                # Process with Docling directly and return result
                return self._docling_process_local(
                    temp_file_path, document_url, config, request_id, upload_id, project_id
                )

            finally:
                # Clean up temp file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        except Exception as e:
            processing_time = time.time() - start_time
            error_message = str(e)

            # Determine if this is a soft failure (transient errors that should retry)
            # or a hard failure (permanent errors that should commit and move on)
            from src.utils.image_validator import ImageValidationError
            from src.exceptions.domain_exceptions import StorageError

            # Check for SOFT failures (transient issues that should retry)
            # These are network/timeout issues that may succeed on retry
            is_soft_failure = (
                "timeout" in error_message.lower()
                or "connection" in error_message.lower()
                or "timed out" in error_message.lower()
            )

            # DEFAULT: All Docling processing errors are HARD failures (commit and move on)
            # Rationale: If Docling can't process a file (unsupported format, corrupted file,
            # validation errors, S3 errors, internal Docling errors), retrying won't help.
            # Only retry if it's a known transient/soft failure.
            is_hard_failure = not is_soft_failure

            # Check if this is already a formatted error from our detailed error handling
            if "DOCLING_ERROR:" in error_message:
                # Extract the readable part for user-facing error message
                readable_part = error_message.split("DOCLING_ERROR:")[-1].strip()
                self.logger.error(
                    "Document processing failed",
                    document_url=document_url,
                    error_details=readable_part,
                    error_full=error_message,
                    error_type=type(e).__name__,
                    project_id=project_id,
                    request_id=request_id,
                    hard_failure=is_hard_failure,
                    exc_info=True,
                )
                user_error = readable_part
            else:
                # Generic error handling for unexpected issues
                self.logger.error(
                    "Local Docling processing failed",
                    document_url=document_url,
                    error_message=error_message,
                    error_type=type(e).__name__,
                    project_id=project_id,
                    request_id=request_id,
                    hard_failure=is_hard_failure,
                    exc_info=True,
                )
                user_error = f"Document processing failed: {error_message}"

            return DoclingDocumentResult(
                document_url=document_url,
                success=False,
                error_message=user_error,
                processing_time_seconds=processing_time,
            )

    def _download_document(self, document_url: str, request_id: str, upload_id: str, project_id: str) -> str:
        """Download document to temporary file and validate if it's an image."""
        try:
            content = self.storage_service.download_content(
                document_url, request_id=request_id, upload_id=upload_id, project_id=project_id
            )

            # Create temporary file
            suffix = Path(document_url).suffix or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name

            # Validate image dimensions if this is an image file
            from src.utils.image_validator import ImageValidator, ImageValidationError

            if ImageValidator.should_validate_as_image(temp_path):
                try:
                    ImageValidator.validate_image_size(
                        temp_path, upload_id=upload_id, project_id=project_id, request_id=request_id
                    )
                    self.logger.info(
                        "Image validation passed",
                        document_url=document_url,
                        upload_id=upload_id,
                        project_id=project_id,
                        request_id=request_id,
                    )
                except ImageValidationError as val_error:
                    # Clean up temp file before re-raising
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    raise  # Re-raise to be caught by caller

            return temp_path

        except Exception as e:
            raise Exception(f"Failed to download document from {document_url}: {str(e)}")

    def _docling_process_local(
        self, file_path: str, document_url: str, config: DoclingConfig, request_id: str, upload_id: str, project_id: str
    ) -> DoclingDocumentResult:
        """Process document with local Docling (runs in thread pool)."""
        start_time = time.time()
        try:
            # Get or reuse existing DocumentConverter
            # This prevents loading 6GB of AI models for every document
            doc_converter = self._get_or_create_converter(config)

            # Convert document with detailed error handling
            try:
                self.logger.info(
                    "Starting docling conversion",
                    file_path=file_path,
                    upload_id=upload_id,
                    project_id=project_id,
                    request_id=request_id,
                )
                conversion_result = doc_converter.convert(file_path)
                doc = conversion_result.document
                self.logger.debug(
                    "Docling conversion completed successfully",
                    file_path=file_path,
                    upload_id=upload_id,
                    project_id=project_id,
                    request_id=request_id,
                )
            except Exception as conv_error:
                # Provide specific error handling for common issues
                error_msg = str(conv_error)

                if "BadZipFile" in error_msg or "File is not a zip file" in error_msg:
                    readable_error = (
                        f"PowerPoint file appears to be corrupted or invalid (not a valid ZIP format): {file_path}"
                    )
                    self.logger.error(
                        "DOCLING_ERROR: PowerPoint file corruption detected",
                        upload_id=upload_id,
                        readable_error=readable_error,
                        error_message=error_msg,
                        error_type=type(conv_error).__name__,
                        project_id=project_id,
                        request_id=request_id,
                        exc_info=True,
                    )
                    raise Exception(readable_error) from conv_error
                elif "zipfile" in error_msg.lower():
                    readable_error = f"Document file format is corrupted or unsupported: {file_path}"
                    self.logger.error(
                        "DOCLING_ERROR: Document file corruption detected",
                        upload_id=upload_id,
                        readable_error=readable_error,
                        error_message=error_msg,
                        error_type=type(conv_error).__name__,
                        project_id=project_id,
                        request_id=request_id,
                        exc_info=True,
                    )
                    raise Exception(readable_error) from conv_error
                elif "Presentation" in error_msg:
                    readable_error = f"PowerPoint file could not be opened - file may be corrupted or password protected: {file_path}"
                    self.logger.error(
                        "DOCLING_ERROR: PowerPoint access failure",
                        readable_error=readable_error,
                        error_message=error_msg,
                        error_type=type(conv_error).__name__,
                        upload_id=upload_id,
                        project_id=project_id,
                        request_id=request_id,
                        exc_info=True,
                    )
                    raise Exception(readable_error) from conv_error
                else:
                    # Generic docling conversion error
                    readable_error = f"Document conversion failed for: {file_path}"
                    self.logger.error(
                        "DOCLING_ERROR: Generic conversion failure",
                        readable_error=readable_error,
                        error_message=error_msg,
                        error_type=type(conv_error).__name__,
                        upload_id=upload_id,
                        project_id=project_id,
                        request_id=request_id,
                        exc_info=True,
                    )
                    raise Exception(f"{readable_error}. Details: {error_msg}") from conv_error

            # Handle page extraction based on configuration
            content, metadata = self._extract_content_with_pages(doc, config)

            # Extract elements with bounding boxes
            elements_with_bbox = self._extract_elements_with_bbox(doc)

            # Extract figures and tables if enabled
            extracted_figures = []
            extracted_tables = []

            if config.enable_figure_extraction:
                try:
                    extracted_figures = self._extract_figures_from_document(doc, request_id, upload_id, document_url)
                    self.logger.info(
                        "Extracted figures from document",
                        figure_count=len(extracted_figures),
                        upload_id=upload_id,
                        request_id=request_id,
                        project_id=project_id,
                    )
                except Exception as e:
                    self.logger.error(
                        "Failed to extract figures",
                        error=str(e),
                        error_type=type(e).__name__,
                        upload_id=upload_id,
                        request_id=request_id,
                        project_id=project_id,
                        exc_info=True,
                    )

            if config.enable_table_structure:
                try:
                    extracted_tables = self._extract_tables_from_document(doc, request_id, upload_id, document_url)
                    self.logger.info(
                        "Extracted tables from document",
                        table_count=len(extracted_tables),
                        upload_id=upload_id,
                        request_id=request_id,
                        project_id=project_id,
                    )
                except Exception as e:
                    self.logger.error(
                        "Failed to extract tables",
                        error=str(e),
                        error_type=type(e).__name__,
                        upload_id=upload_id,
                        request_id=request_id,
                        project_id=project_id,
                        exc_info=True,
                    )

            # Add extraction summary to metadata
            metadata["extraction_summary"] = {
                "figures_extracted": len(extracted_figures),
                "tables_extracted": len(extracted_tables),
                "extraction_enabled": {
                    "figures": config.enable_figure_extraction,
                    "tables": config.enable_table_structure,
                },
            }

            # Calculate processing time and return complete result
            processing_time = time.time() - start_time

            return DoclingDocumentResult(
                document_url=document_url,
                success=True,
                content=content,
                metadata=metadata,
                bounding_box=elements_with_bbox,
                processing_time_seconds=processing_time,
                page_count=metadata.get("page_count"),
                word_count=metadata.get("word_count"),
                doc=doc,
                extracted_figures=extracted_figures,
                extracted_tables=extracted_tables,
            )

        except ImportError as e:
            readable_error = "Docling library is not available or not properly installed"
            self.logger.error(
                "DOCLING_ERROR: Library not available",
                readable_error=readable_error,
                upload_id=upload_id,
                project_id=project_id,
                request_id=request_id,
            )
            self.logger.error(
                "Technical details",
                error_message=str(e),
                upload_id=upload_id,
                project_id=project_id,
                request_id=request_id,
            )
            raise Exception(readable_error)
        except Exception as e:
            # This catches any other errors not handled above
            if "DOCLING_ERROR:" not in str(e):
                self.logger.error(
                    "DOCLING_ERROR: Unexpected error during document processing",
                    error_message=str(e),
                    error_type=type(e).__name__,
                    upload_id=upload_id,
                    project_id=project_id,
                    request_id=request_id,
                    exc_info=True,
                )
            raise

    def _create_docling_request(
        self,
        request_id: str,
        project_id: str,
        upload_id: str,
        document_urls: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DoclingRequest:
        """Create optimized DoclingRequest based on document analysis."""
        # Get primary file type directly from first document
        provided_file_type = metadata.get("file_type") if metadata else None
        primary_file_type = (
            MimeTypeUtils.determine_mime_type(document_urls[0], provided_file_type) if document_urls else "unknown"
        )

        # Create optimized configuration for the primary file type
        config = self.config_manager.create_optimized_config(
            file_type=primary_file_type, custom_settings=metadata.get("docling_config") if metadata else None
        )

        return DoclingRequest(
            request_id=request_id,
            upload_id=upload_id,
            project_id=project_id,
            document_urls=document_urls,
            config=config,
        )

    async def _handle_docling_results(
        self,
        request_id: str,
        document_urls: List[str],
        docling_result: DoclingBatchResult,
        project_id: str,
        upload_id: str,
        file_type: str = "unknown",
    ) -> Dict[str, Any]:
        """Handle Docling processing results."""
        try:
            if docling_result.success:
                return await self._handle_successful_docling_processing(
                    request_id=request_id,
                    document_urls=document_urls,
                    docling_result=docling_result,
                    project_id=project_id,
                    upload_id=upload_id,
                    file_type=file_type,
                )
            else:
                return await self._handle_failed_docling_processing(
                    request_id=request_id,
                    document_urls=document_urls,
                    docling_result=docling_result,
                    project_id=project_id,
                    upload_id=upload_id,
                    file_type=file_type,
                )

        except Exception as e:
            self.logger.error("Failed to handle Docling results", request_id=request_id, error=str(e))
            return self._create_failure_result(
                request_id=request_id,
                document_urls=document_urls,
                error_message=f"Failed to process Docling results: {str(e)}",
                project_id=project_id,
                upload_id=upload_id,
            )

    async def _handle_successful_docling_processing(
        self,
        request_id: str,
        document_urls: List[str],
        docling_result: DoclingBatchResult,
        project_id: str,
        upload_id: str,
        file_type: str = "unknown",
    ) -> Dict[str, Any]:
        """Handle successful Docling processing results."""
        try:
            # Extract and store assets (figures and tables) FIRST - populates S3 paths on objects
            asset_paths = self._extract_and_store_assets(
                docling_result=docling_result, project_id=project_id, upload_id=upload_id
            )

            # Store main processing results to S3 (per-page) - now with S3 paths populated
            page_s3_paths = self._store_docling_results(
                docling_result=docling_result, project_id=project_id, upload_id=upload_id
            )

            # Construct base S3 location for status tracking
            from src.configs.settings import settings

            base_s3_location = f"s3://{settings.S3_BUCKET_NAME}/{project_id}/{upload_id}"

            # Build extractions array with relative paths for Kafka message
            # Format: ["0/results.json", "1/results.json", ...]
            extractions = [f"{page_num}/results.json" for page_num in sorted(page_s3_paths.keys())]

            # Update status to success with minimal metadata
            result_metadata = {
                "total_documents": docling_result.total_documents,
                "successful_documents": docling_result.successful_documents,
                "failed_documents": docling_result.failed_documents,
                "processing_time_seconds": docling_result.total_processing_time_seconds,
                "total_pages": len(page_s3_paths),
                "extractions": extractions,  # Array of results.json paths for Kafka
                "extraction_type": ExtractionType.DOCLING.value,
            }

            await self.status_tracker.update_status(
                upload_id=upload_id,
                status=IngestionStepStatus.SUCCESS,
                project_id=project_id,
                s3_location=base_s3_location,
                request_id=request_id,
                file_type=file_type,
                message=f"Successfully processed {len(document_urls)} documents ({len(page_s3_paths)} pages) via Docling",
                metadata=result_metadata,
            )

            self.logger.info(
                "Docling processing completed successfully",
                request_id=request_id,
                document_count=len(document_urls),
                total_pages=len(page_s3_paths),
                base_s3_location=base_s3_location,
                asset_count=len(asset_paths.get("figures", [])) + len(asset_paths.get("tables", [])),
            )

            return self._create_standard_result_format(
                request_id=request_id,
                success=True,
                document_urls=document_urls,
                processing_data={
                    "base_s3_location": base_s3_location,
                    "page_results": page_s3_paths,
                    "docling_result": docling_result,
                    "asset_paths": asset_paths,
                    "processing_method": "docling",
                    "total_pages": len(page_s3_paths),
                },
            )

        except Exception as e:
            self.logger.error("Failed to handle successful Docling processing", request_id=request_id, error=str(e))
            return self._create_failure_result(
                request_id=request_id,
                document_urls=document_urls,
                error_message=f"Failed to process successful results: {str(e)}",
                project_id=project_id,
                upload_id=upload_id,
            )

    async def _handle_failed_docling_processing(
        self,
        request_id: str,
        document_urls: List[str],
        docling_result: DoclingBatchResult,
        project_id: str,
        upload_id: str,
        file_type: str = "unknown",
    ) -> Dict[str, Any]:
        """Handle failed Docling processing results."""
        error_message = "Docling processing failed"
        if hasattr(docling_result, "error_message"):
            error_message = docling_result.error_message

        # DEFAULT: All Docling failures are HARD failures (commit and move on)
        # Only mark as soft failure if it's a known transient error
        # This matches the logic in _process_document_local()
        from src.utils.image_validator import ImageValidationError

        # Start with hard failure as default
        is_hard_failure = True
        error_type = "DoclingProcessingError"  # Default generic type
        failed_doc_url = None
        error_details = None

        for doc_result in docling_result.results:
            if doc_result.error_message:
                # Capture first failure details
                if not failed_doc_url:
                    failed_doc_url = doc_result.document_url
                    error_details = doc_result.error_message

                # Check if this is a SOFT failure (transient network/timeout issues)
                if (
                    "timeout" in doc_result.error_message.lower()
                    or "connection" in doc_result.error_message.lower()
                    or "timed out" in doc_result.error_message.lower()
                ):
                    is_hard_failure = False
                    error_type = "TransientNetworkError"
                    break  # First soft failure determines the result

                # Determine specific error type for hard failures
                if "exceeds" in doc_result.error_message:
                    error_type = "ImageValidationError"
                elif "File not found" in doc_result.error_message or "NoSuchKey" in doc_result.error_message:
                    error_type = "StorageError"
                elif "corrupted" in doc_result.error_message.lower() or "invalid" in doc_result.error_message.lower():
                    error_type = "CorruptedFileError"
                elif "BadZipFile" in doc_result.error_message or "not a zip file" in doc_result.error_message.lower():
                    error_type = "CorruptedFileError"
                # else: Keep default "DoclingProcessingError" for generic errors

        self.logger.error(
            "Docling processing failed",
            request_id=request_id,
            error_message=error_message,
            document_count=len(document_urls),
            hard_failure=is_hard_failure,
            error_type=error_type,
        )

        # Build detailed error metadata for status tracking
        error_metadata = {
            "error_type": error_type,
            "hard_failure": is_hard_failure,
            "total_documents": docling_result.total_documents,
            "successful_documents": docling_result.successful_documents,
            "failed_documents": docling_result.failed_documents,
        }

        # Add document-specific error details
        if failed_doc_url:
            error_metadata["failed_document_url"] = failed_doc_url
        if error_details:
            error_metadata["error_details"] = error_details

        # Update status to failed with detailed metadata
        await self.status_tracker.update_status(
            upload_id=upload_id,
            status=IngestionStepStatus.FAILURE,
            project_id=project_id,
            request_id=request_id,
            file_type=file_type,
            message=f"Docling processing failed: {error_message}",
            metadata=error_metadata,
        )

        result = self._create_failure_result(
            request_id=request_id,
            document_urls=document_urls,
            error_message=error_message,
            project_id=project_id,
            upload_id=upload_id,
            docling_result=docling_result,
        )

        # Add hard_failure flag to the result dict
        result["hard_failure"] = is_hard_failure

        return result

    async def _handle_empty_documents(
        self, request_id: str, project_id: str, upload_id: str, file_type: str = "unknown"
    ) -> Dict[str, Any]:
        """Handle the case of empty document list."""
        self.logger.info(
            "No documents to process via Docling", request_id=request_id, project_id=project_id, upload_id=upload_id
        )

        # Create empty result for consistency
        empty_result_data = {
            "results": [],
            "total_documents": 0,
            "successful_documents": 0,
            "failed_documents": 0,
        }

        # Update status to success (empty processing is still successful)
        await self.status_tracker.update_status(
            upload_id=upload_id,
            status=IngestionStepStatus.SUCCESS,
            project_id=project_id,
            request_id=request_id,
            file_type=file_type,
            message="Successfully processed 0 documents via Docling",
            metadata={"docling_result": empty_result_data},
        )

        return self._create_standard_result_format(
            request_id=request_id,
            success=True,
            document_urls=[],
            processing_data={"docling_result": empty_result_data, "processing_method": "docling"},
        )

    def _store_docling_results(
        self, docling_result: DoclingBatchResult, project_id: str, upload_id: str
    ) -> Dict[int, str]:
        """
        Store Docling results to S3 per-page and return mapping of page numbers to S3 paths.

        Args:
            docling_result: Batch result containing document processing results
            project_id: Project identifier
            upload_id: Upload identifier

        Returns:
            Dictionary mapping page_number to S3 path for that page's results
        """
        from src.configs.settings import settings

        page_paths: Dict[int, str] = {}
        bucket = settings.S3_BUCKET_NAME

        # Process each document result
        for doc_result in docling_result.results:
            if not doc_result.success or not doc_result.doc:
                continue

            # Get total page count from the document
            doc = doc_result.doc
            total_pages = len(doc.pages) if hasattr(doc, "pages") else 1

            self.logger.info(
                "Processing document pages",
                total_pages=total_pages,
                document_url=doc_result.document_url,
                project_id=project_id,
                upload_id=upload_id,
            )

            # Process each page separately (1-indexed to match Docling provenance)
            for page_number in range(1, total_pages + 1):
                self.logger.debug(
                    "Processing page",
                    page_number=page_number,
                    total_pages=total_pages,
                    project_id=project_id,
                    upload_id=upload_id,
                )

                # Extract page-specific content (Docling uses 1-indexed page numbers in provenance)
                page_content = self._extract_page_content(doc, page_number, docling_result.config_used)

                # Filter figures for this page (Docling uses 1-indexed page numbers)
                page_figures = [fig for fig in (doc_result.extracted_figures or []) if fig.page_number == page_number]

                # Filter tables for this page (Docling uses 1-indexed page numbers)
                page_tables = [
                    table for table in (doc_result.extracted_tables or []) if table.page_number == page_number
                ]

                # Filter bounding boxes for this page (Docling uses 1-indexed page numbers)
                page_bboxes = [bbox for bbox in (doc_result.bounding_box or []) if bbox.get("page_no") == page_number]

                # Create page metadata using Pydantic model
                page_metadata = PageMetadata(
                    page_count=total_pages,
                    word_count=len(page_content.split()) if page_content else 0,
                    figure_count=len(page_figures),
                    table_count=len(page_tables),
                    extraction_config=docling_result.config_used.model_dump(),
                )

                # Create figure references using Pydantic model
                figure_references = [
                    FigureReference(
                        figure_id=fig.figure_id,
                        s3_image_path=getattr(fig, "s3_image_path", None),
                        s3_metadata_path=getattr(fig, "s3_metadata_path", None),
                        page_number=getattr(fig, "page_number", None),
                    )
                    for fig in page_figures
                ]

                # Create table references using Pydantic model
                table_references = [
                    TableReference(
                        table_id=table.table_id,
                        s3_csv_path=getattr(table, "s3_csv_path", None),
                        s3_metadata_path=getattr(table, "s3_metadata_path", None),
                        title=getattr(table, "title", None),
                        row_count=getattr(table, "row_count", None),
                        column_count=getattr(table, "column_count", None),
                        page_number=getattr(table, "page_number", None),
                    )
                    for table in page_tables
                ]

                # Create page-specific result using Pydantic model
                page_result = PageExtractionResult(
                    page_number=page_number,
                    document_url=doc_result.document_url,
                    content=page_content,
                    metadata=page_metadata,
                    bounding_boxes=page_bboxes,
                    figures=figure_references,
                    tables=table_references,
                    processing_time_seconds=doc_result.processing_time_seconds,
                    extraction_type=ExtractionType.DOCLING,
                )

                # Generate S3 path for this page
                key = generate_results_json_s3_path(project_id, upload_id, page_number)

                # Serialize page results using Pydantic
                content = page_result.model_dump_json(indent=2).encode("utf-8")

                # Upload to S3
                s3_path = self.storage_service.upload_content(
                    content, bucket, key, upload_id=upload_id, project_id=project_id
                )

                page_paths[page_number] = s3_path

                self.logger.debug(
                    "Stored page results to S3",
                    s3_path=s3_path,
                    page_number=page_number,
                    project_id=project_id,
                    upload_id=upload_id,
                )

                # Store page content as separate text file for easy downstream consumption
                # while also maintaining the full PageExtractionResult in results.json for
                # consistency and compatibility with existing pipeline consumers.
                text_key = generate_page_text_s3_path(project_id, upload_id, page_number)
                store_text_file_to_s3(
                    storage_service=self.storage_service,
                    logger=self.logger,
                    text_content=page_content,
                    bucket=bucket,
                    s3_key=text_key,
                    project_id=project_id,
                    upload_id=upload_id,
                    log_context={"page_number": page_number},
                )

        self.logger.info(
            "Stored all page results to S3", total_pages=len(page_paths), project_id=project_id, upload_id=upload_id
        )

        return page_paths

    def _extract_page_content(self, doc: Any, page_number: int, config: DoclingConfig) -> str:
        """
        Extract content for a specific page from the document.

        Args:
            doc: Docling document object
            page_number: Page number to extract (1-indexed, matches Docling provenance)
            config: Docling configuration for output format

        Returns:
            Page content as string
        """
        try:
            # Filter text elements for this page (both page_number and prov.page_no are 1-indexed)
            if hasattr(doc, "texts"):
                page_elements = [elem for elem in doc.texts if any(prov.page_no == page_number for prov in elem.prov)]

                # Build content based on output format (page_number is already 1-indexed)
                if config.output_format == DoclingOutputFormatEnum.MARKDOWN:
                    content_parts = [f"# Page {page_number}\n\n"]
                    for elem in page_elements:
                        content_parts.append(elem.text + "\n")
                    return "".join(content_parts)
                elif config.output_format == DoclingOutputFormatEnum.TEXT:
                    content_parts = [f"Page {page_number}\n\n"]
                    for elem in page_elements:
                        content_parts.append(elem.text + "\n")
                    return "".join(content_parts)
                else:
                    # For other formats, just concatenate text
                    return "\n".join(elem.text for elem in page_elements)

            return f"Page {page_number} content not available"

        except Exception as e:
            self.logger.warning("Failed to extract page content", page_number=page_number, error=str(e))
            return f"Page {page_number}: Content extraction failed"

    def _extract_and_store_assets(
        self, docling_result: DoclingBatchResult, project_id: str, upload_id: str
    ) -> Dict[str, List[str]]:
        """
        Extract and store figures and tables from Docling results.

        This method groups figures and tables by their actual page numbers
        (from docling provenance data) rather than using document index.
        """
        asset_paths: Dict[str, List[str]] = {"figures": [], "tables": []}

        try:
            # Group figures and tables by page number across all documents
            figures_by_page: Dict[int, List[ExtractedFigure]] = {}
            tables_by_page: Dict[int, List[ExtractedTable]] = {}
            doc_by_page: Dict[int, Any] = {}

            # Collect all figures and tables from all documents
            for result in docling_result.results:
                if not result.success or not result.doc:
                    continue

                # Group figures by their actual page_number field
                if result.extracted_figures:
                    for figure in result.extracted_figures:
                        page_num = figure.page_number
                        if page_num not in figures_by_page:
                            figures_by_page[page_num] = []
                            doc_by_page[page_num] = result.doc
                        figures_by_page[page_num].append(figure)

                # Group tables by their actual page_number field
                if result.extracted_tables:
                    for table in result.extracted_tables:
                        page_num = table.page_number
                        if page_num not in tables_by_page:
                            tables_by_page[page_num] = []
                            if page_num not in doc_by_page:
                                doc_by_page[page_num] = result.doc
                        tables_by_page[page_num].append(table)

            # Store assets for each page
            all_page_numbers = sorted(set(figures_by_page.keys()) | set(tables_by_page.keys()))

            for page_number in all_page_numbers:
                doc = doc_by_page.get(page_number)

                # Store figures for this page
                if page_number in figures_by_page:
                    stored_figures = self.asset_storage.store_extracted_figures(
                        doc=doc,
                        extracted_figures=figures_by_page[page_number],
                        project_id=project_id,
                        upload_id=upload_id,
                        page_number=page_number,
                    )
                    # Collect S3 paths
                    figure_paths = [fig.s3_image_path for fig in stored_figures if fig.s3_image_path]
                    asset_paths["figures"].extend(figure_paths)

                # Store tables for this page
                if page_number in tables_by_page:
                    stored_tables = self.asset_storage.store_extracted_tables(
                        doc=doc,
                        extracted_tables=tables_by_page[page_number],
                        project_id=project_id,
                        upload_id=upload_id,
                        page_number=page_number,
                    )
                    # Collect S3 paths
                    table_paths = []
                    for table in stored_tables:
                        if table.s3_csv_path:
                            table_paths.append(table.s3_csv_path)
                    asset_paths["tables"].extend(table_paths)

            self.logger.info(
                "Asset extraction completed",
                project_id=project_id,
                upload_id=upload_id,
                pages_with_assets=len(all_page_numbers),
                figure_count=len(asset_paths["figures"]),
                table_count=len(asset_paths["tables"]),
            )

        except Exception as e:
            self.logger.error(
                "Failed to extract and store assets", project_id=project_id, upload_id=upload_id, error=str(e)
            )

        return asset_paths

    def _create_failure_result(
        self,
        request_id: str,
        document_urls: List[str],
        error_message: str,
        project_id: str,
        upload_id: str,
        docling_result: Optional[DoclingBatchResult] = None,
    ) -> Dict[str, Any]:
        """Create standardized failure result."""
        result_data: Dict[str, Any] = {"processing_method": "docling", "project_id": project_id, "upload_id": upload_id}

        if docling_result:
            result_data["docling_result"] = docling_result.model_dump()

        return self._create_standard_result_format(
            request_id=request_id,
            success=False,
            document_urls=document_urls,
            processing_data=result_data,
            error_message=error_message,
        )

    def _extract_content_with_pages(self, doc: Any, config: DoclingConfig) -> tuple[str, Dict[str, Any]]:
        """Extract content with page-specific handling based on configuration."""

        # Get total page count
        total_pages = len(doc.pages) if hasattr(doc, "pages") else 0

        if config.single_page_mode and total_pages > 1:
            # Extract each page as separate content
            page_contents = []

            for i, page in enumerate(doc.pages):
                # Create a temporary document with just this page
                # Note: This is a conceptual approach - actual implementation may vary based on Docling's API
                try:
                    # Extract content for this specific page
                    if config.output_format == DoclingOutputFormatEnum.MARKDOWN:
                        page_content = f"# Page {i + 1}\n\n"
                        # Extract page-specific content based on elements on this page
                        page_elements = [elem for elem in doc.texts if any(prov.page_no == i + 1 for prov in elem.prov)]
                        for elem in page_elements:
                            page_content += elem.text + "\n"
                    elif config.output_format == DoclingOutputFormatEnum.TEXT:
                        page_content = f"Page {i + 1}\n\n"
                        page_elements = [elem for elem in doc.texts if any(prov.page_no == i + 1 for prov in elem.prov)]
                        for elem in page_elements:
                            page_content += elem.text + "\n"
                    else:
                        # For HTML and JSON, fall back to full document extraction
                        page_content = f"Page {i + 1} content (format: {config.output_format})"

                    page_contents.append(page_content)
                except Exception as e:
                    self.logger.warning("Failed to extract page", page_number=i + 1, error=str(e))
                    page_contents.append(f"Page {i + 1}: Extraction failed")

            content = "\n\n---\n\n".join(page_contents)
            extraction_mode = "single_page"
        else:
            # Extract full document content (default behavior)
            if config.output_format == DoclingOutputFormatEnum.MARKDOWN:
                content = doc.export_to_markdown()
            elif config.output_format == DoclingOutputFormatEnum.TEXT:
                content = doc.export_to_text()
            elif config.output_format == DoclingOutputFormatEnum.HTML:
                content = doc.export_to_html()
            else:  # JSON
                content = doc.export_to_json()

            extraction_mode = "multipage"

        # Extract metadata
        metadata = {
            "page_count": total_pages,
            "extraction_mode": extraction_mode,
            "word_count": (len(content.split()) if isinstance(content, str) else None),
            "docling_version": doc.version if hasattr(doc, "version") else None,
            "format_detected": (str(doc.input_format) if hasattr(doc, "input_format") else None),
            "page_extraction_config": {
                "single_page_mode": config.single_page_mode,
                "extract_pages": config.extract_pages,
            },
        }

        return content, metadata

    def _extract_elements_with_bbox(self, doc: Any) -> List[Dict[str, Any]]:
        """Extract document elements with their bounding boxes and metadata."""
        elements = []

        # Determine if this is a presentation format
        is_presentation = doc.origin.filename.lower().endswith((".pptx", ".ppt")) if hasattr(doc, "origin") else False

        # Process all text items (includes headers, paragraphs, etc.)
        for text_item in doc.texts:
            for prov_item in text_item.prov:
                element = {
                    "text": text_item.text,
                    "label": (text_item.label.value if hasattr(text_item.label, "value") else str(text_item.label)),
                    "page_no": prov_item.page_no,
                    "slide_no": (prov_item.page_no if is_presentation else None),  # Add slide reference
                    "bbox": {
                        "left": prov_item.bbox.l,
                        "top": prov_item.bbox.t,
                        "right": prov_item.bbox.r,
                        "bottom": prov_item.bbox.b,
                        "coord_origin": prov_item.bbox.coord_origin.value,
                    },
                    "charspan": (prov_item.charspan if hasattr(prov_item, "charspan") else None),
                    "source_doc": (doc.name or doc.origin.filename if hasattr(doc, "origin") else None),
                    "document_type": "presentation" if is_presentation else "document",
                }
                elements.append(element)

        return elements

    def _extract_figures_from_document(
        self, doc: Any, request_id: str, upload_id: str, document_url: str
    ) -> List[ExtractedFigure]:
        """
        Extract figures/images from the processed document using Docling's get_image API.

        Args:
            doc: Processed Docling document
            request_id: Processing request ID
            document_url: Source document URL

        Returns:
            List of ExtractedFigure objects with metadata and S3 paths to be populated
        """
        extracted_figures: List[ExtractedFigure] = []

        try:
            # Check if document has pictures collection
            if not hasattr(doc, "pictures") or not doc.pictures:
                self.logger.info(
                    "No figures found in document",
                    document_url=document_url,
                    upload_id=upload_id,
                    request_id=request_id,
                )
                return extracted_figures

            self.logger.info(
                "Found figures in document",
                figure_count=len(doc.pictures),
                upload_id=upload_id,
                document_url=document_url,
                request_id=request_id,
            )

            for idx, picture in enumerate(doc.pictures):
                try:
                    # Generate unique figure ID
                    figure_id = f"{request_id}_figure_{idx}_{uuid.uuid4().hex[:8]}"

                    # Extract page number from provenance
                    page_number = 1  # default
                    if hasattr(picture, "prov") and picture.prov:
                        page_number = picture.prov[0].page_no

                    # Extract bounding box information
                    bbox = None
                    if hasattr(picture, "prov") and picture.prov:
                        prov = picture.prov[0]
                        if hasattr(prov, "bbox"):
                            bbox = {
                                "left": prov.bbox.l,
                                "top": prov.bbox.t,
                                "right": prov.bbox.r,
                                "bottom": prov.bbox.b,
                                "coord_origin": prov.bbox.coord_origin.value
                                if hasattr(prov.bbox, "coord_origin")
                                else "TOP_LEFT",
                            }

                    # Extract caption if available
                    caption = None
                    if hasattr(picture, "text") and picture.text:
                        caption = picture.text.strip()

                    # Get label type
                    label = (
                        (picture.label.value if hasattr(picture.label, "value") else str(picture.label))
                        if hasattr(picture, "label")
                        else "picture"
                    )

                    # Create ExtractedFigure object (S3 paths will be populated during storage)
                    extracted_figure = ExtractedFigure(
                        figure_id=figure_id,
                        page_number=page_number,
                        caption=caption,
                        alt_text=None,
                        label=label,
                        bbox=bbox,
                        s3_image_path=None,
                        s3_metadata_path="",  # Will be set during S3 upload
                        image_format=None,
                        image_size=None,
                        extraction_method="docling_get_image",
                        extracted_at=datetime.now(timezone.utc),
                    )

                    extracted_figures.append(extracted_figure)
                    self.logger.debug(
                        "Extracted figure metadata",
                        figure_id=figure_id,
                        upload_id=upload_id,
                        page_number=page_number,
                        request_id=request_id,
                    )

                except Exception as e:
                    self.logger.error(
                        "Failed to extract figure",
                        figure_index=idx,
                        upload_id=upload_id,
                        error=str(e),
                        request_id=request_id,
                    )
                    continue

        except Exception as e:
            self.logger.error(
                "Error extracting figures from document",
                error=str(e),
                upload_id=upload_id,
                document_url=document_url,
                request_id=request_id,
            )

        return extracted_figures

    def _extract_tables_from_document(
        self, doc: Any, request_id: str, upload_id: str, document_url: str
    ) -> List[ExtractedTable]:
        """
        Extract tables from the processed document using Docling's table export APIs.

        Args:
            doc: Processed Docling document
            request_id: Processing request ID
            document_url: Source document URL

        Returns:
            List of ExtractedTable objects with metadata and S3 paths to be populated
        """
        extracted_tables: List[ExtractedTable] = []

        try:
            # Check if document has tables collection
            if not hasattr(doc, "tables") or not doc.tables:
                self.logger.info(
                    "No tables found in document", document_url=document_url, upload_id=upload_id, request_id=request_id
                )
                return extracted_tables

            self.logger.info(
                "Found tables in document",
                table_count=len(doc.tables),
                upload_id=upload_id,
                document_url=document_url,
                request_id=request_id,
            )

            for idx, table in enumerate(doc.tables):
                try:
                    # Generate unique table ID
                    table_id = f"{request_id}_table_{idx}_{uuid.uuid4().hex[:8]}"

                    # Extract page number from provenance
                    page_number = 1  # default
                    if hasattr(table, "prov") and table.prov:
                        page_number = table.prov[0].page_no

                    # Extract bounding box information
                    bbox = None
                    if hasattr(table, "prov") and table.prov:
                        prov = table.prov[0]
                        if hasattr(prov, "bbox"):
                            bbox = {
                                "left": prov.bbox.l,
                                "top": prov.bbox.t,
                                "right": prov.bbox.r,
                                "bottom": prov.bbox.b,
                                "coord_origin": prov.bbox.coord_origin.value
                                if hasattr(prov.bbox, "coord_origin")
                                else "TOP_LEFT",
                            }

                    # Extract caption if available
                    caption = None
                    if hasattr(table, "text") and table.text:
                        caption = table.text.strip()

                    # Get label type
                    label = (
                        (table.label.value if hasattr(table.label, "value") else str(table.label))
                        if hasattr(table, "label")
                        else "table"
                    )

                    # Extract table structure information
                    num_rows = 0
                    num_cols = 0
                    headers = []
                    cell_count = 0
                    has_merged_cells = False

                    # Try to get table structure from Docling table data
                    if hasattr(table, "data") and table.data:
                        table_data = table.data
                        if hasattr(table_data, "table"):
                            docling_table = table_data.table
                            if hasattr(docling_table, "num_rows"):
                                num_rows = docling_table.num_rows
                            if hasattr(docling_table, "num_cols"):
                                num_cols = docling_table.num_cols
                            if hasattr(docling_table, "table_cells"):
                                cell_count = len(docling_table.table_cells)

                            # Extract headers if available
                            if hasattr(docling_table, "table_cells"):
                                # Get first row as headers (common pattern)
                                first_row_cells = [
                                    cell for cell in docling_table.table_cells if cell.start_row_offset_idx == 0
                                ]
                                headers = [
                                    cell.text for cell in sorted(first_row_cells, key=lambda x: x.start_col_offset_idx)
                                ]

                    # Create ExtractedTable object (S3 paths will be populated during storage)
                    extracted_table = ExtractedTable(
                        table_id=table_id,
                        page_number=page_number,
                        caption=caption,
                        label=label,
                        num_rows=num_rows,
                        num_cols=num_cols,
                        headers=headers,
                        bbox=bbox,
                        s3_csv_path=None,
                        s3_metadata_path="",  # Will be set during S3 upload
                        has_merged_cells=has_merged_cells,
                        cell_count=cell_count,
                        extraction_method="docling_export",
                        extracted_at=datetime.now(timezone.utc),
                    )

                    extracted_tables.append(extracted_table)
                    self.logger.debug(
                        "Extracted table metadata",
                        table_id=table_id,
                        upload_id=upload_id,
                        page_number=page_number,
                        table_dimensions=f"{num_rows}x{num_cols}",
                        request_id=request_id,
                    )

                except Exception as e:
                    self.logger.error(
                        "Failed to extract table",
                        table_index=idx,
                        upload_id=upload_id,
                        error=str(e),
                        request_id=request_id,
                    )
                    continue

        except Exception as e:
            self.logger.error(
                "Error extracting tables from document",
                error=str(e),
                upload_id=upload_id,
                document_url=document_url,
                request_id=request_id,
            )

        return extracted_tables

    def health_check(self) -> Dict[str, str]:
        """
        Check the health of the Docling strategy processor.

        Returns:
            Dictionary with health status information
        """
        health = {"docling_strategy_processor": "healthy", "strategy_name": self.strategy_name}

        # Check local Docling import availability
        try:
            import importlib.util

            if importlib.util.find_spec("docling.document_converter") is not None:
                health["local_docling"] = "available"
            else:
                health["local_docling"] = "unavailable"
        except Exception:
            health["local_docling"] = "unavailable"

        # Check storage service health
        try:
            if hasattr(self.storage_service, "health_check"):
                storage_health = self.storage_service.health_check()
                health["storage_service"] = "healthy" if storage_health else "unhealthy"
            else:
                health["storage_service"] = "available"
        except Exception:
            health["storage_service"] = "unhealthy"

        # Check status tracker health
        try:
            status_health = self.status_tracker.health_check()
            health["status_tracker"] = "healthy" if status_health.get("status_tracker") == "healthy" else "unhealthy"
        except Exception:
            health["status_tracker"] = "unhealthy"

        return health

    def supports_file_type(self, file_type: str) -> bool:
        """
        Check if Docling strategy supports the given file type.

        Args:
            file_type: MIME type string to check

        Returns:
            True if Docling can process this file type effectively
        """
        # Docling excels at structured documents
        docling_optimal_types = {
            IngestionMimeType.PDF.value,
            IngestionMimeType.DOCX.value,
            IngestionMimeType.DOC.value,
            IngestionMimeType.PPTX.value,
        }

        # Docling can handle these but may not be optimal
        docling_supported_types = {
            IngestionMimeType.PNG.value,
            IngestionMimeType.JPG.value,
            IngestionMimeType.TXT.value,
            IngestionMimeType.RTF.value,
        }

        return file_type in docling_optimal_types or file_type in docling_supported_types

    def get_processing_capabilities(self) -> Dict[str, Any]:
        """
        Get information about Docling processor capabilities.

        Returns:
            Dictionary describing processing capabilities
        """
        return {
            "strategy_name": self.strategy_name,
            "supported_file_types": [
                IngestionMimeType.PDF.value,
                IngestionMimeType.DOCX.value,
                IngestionMimeType.DOC.value,
                IngestionMimeType.PPTX.value,
                IngestionMimeType.PNG.value,
                IngestionMimeType.JPG.value,
                IngestionMimeType.TXT.value,
                IngestionMimeType.RTF.value,
            ],
            "optimal_file_types": [
                IngestionMimeType.PDF.value,
                IngestionMimeType.DOCX.value,
                IngestionMimeType.DOC.value,
                IngestionMimeType.PPTX.value,
            ],
            "features": [
                "local_processing",
                "structure_extraction",
                "figure_extraction",
                "table_extraction",
                "ocr_capability",
                "detailed_metadata",
            ],
            "limitations": ["local_compute_requirements", "limited_batch_size", "processing_time_varies"],
            "performance_characteristics": {
                "batch_size": "small_to_medium",
                "concurrency": "medium",
                "processing_speed": "moderate",
                "resource_usage": "high_local",
            },
            "optimal_use_cases": [
                "structured_documents",
                "detailed_extraction",
                "figure_table_extraction",
                "local_processing_requirements",
            ],
        }
