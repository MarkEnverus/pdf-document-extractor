"""
Processing configuration management service.

This service extracts configuration optimization logic from DocumentProcessingService
to follow single responsibility principle.
"""

from typing import Dict, Any, Optional, List, cast

from lib.models.mime_type import IngestionMimeType
from lib.logger import Logger
from src.models.docling_models import DoclingConfig, DoclingProviderEnum, DoclingOutputFormatEnum

logger = Logger.get_logger(__name__)


class ProcessingConfigManager:
    """
    Service responsible for managing and optimizing processing configurations.

    This service handles:
    - File type-specific configuration optimizations
    - Default configuration creation and management
    - Processing parameter tuning based on document characteristics
    - Configuration validation and recommendations
    """

    def __init__(self) -> None:
        """Initialize ProcessingConfigManager."""
        self.logger = logger

    def create_optimized_config(
        self,
        file_type: str,
        base_config: Optional[DoclingConfig] = None,
        custom_settings: Optional[Dict[str, Any]] = None,
    ) -> DoclingConfig:
        """
        Create an optimized configuration for the given file type.

        Args:
            file_type: MIME type string from MimeType enum
            base_config: Optional base configuration to start from
            custom_settings: Optional custom settings to override defaults

        Returns:
            Optimized DoclingConfig for the file type
        """
        # Start with base config or create default
        config = base_config.model_copy() if base_config else self._create_default_config()

        # Apply file type-specific optimizations
        self._apply_file_type_optimizations(config, file_type)

        # Apply any custom settings
        if custom_settings:
            self._apply_custom_settings(config, custom_settings)

        self.logger.info(
            "Created optimized configuration",
            file_type=file_type,
            enable_ocr=config.enable_ocr,
            enable_table_structure=config.enable_table_structure,
            enable_figure_extraction=config.enable_figure_extraction,
        )

        return config

    def _create_default_config(self) -> DoclingConfig:
        """
        Create a default DoclingConfig with sensible defaults.

        Returns:
            Default DoclingConfig
        """
        return DoclingConfig(
            provider=DoclingProviderEnum.LOCAL,
            output_format=DoclingOutputFormatEnum.MARKDOWN,
            enable_ocr=True,
            enable_table_structure=True,
            enable_figure_extraction=True,
            single_page_mode=False,
            extract_pages="all",
            chunk_size=1000,
            overlap_size=100,
        )

    def _apply_file_type_optimizations(self, config: DoclingConfig, file_type: str) -> None:
        """
        Apply file type-specific optimizations to Docling configuration.

        Args:
            config: DoclingConfig to modify
            file_type: MIME type string from MimeType enum
        """
        self.logger.debug("Applying file type optimizations", file_type=file_type)

        # PDF-specific optimizations
        if file_type == IngestionMimeType.PDF.value:
            self._optimize_for_pdf(config)

        # DOCX/DOC-specific optimizations
        elif file_type in [IngestionMimeType.DOCX.value, IngestionMimeType.DOC.value]:
            self._optimize_for_word_document(config)

        # PowerPoint-specific optimizations
        elif file_type == IngestionMimeType.PPTX.value:
            self._optimize_for_powerpoint(config)

        # Image-specific optimizations
        elif file_type in [IngestionMimeType.PNG.value, IngestionMimeType.JPG.value]:
            self._optimize_for_image(config)

        # Excel-specific optimizations
        elif file_type in [IngestionMimeType.XLSX.value, IngestionMimeType.XLS.value]:
            self._optimize_for_excel(config)

        # Text file optimizations
        elif file_type in [IngestionMimeType.TXT.value, IngestionMimeType.RTF.value]:
            self._optimize_for_text(config)

        # Unknown or unsupported file types - use conservative defaults
        else:
            self._optimize_for_unknown(config, file_type)

    def _optimize_for_pdf(self, config: DoclingConfig) -> None:
        """Optimize configuration for PDF files."""
        # PDFs often benefit from OCR for scanned documents
        config.enable_ocr = True
        config.enable_table_structure = True
        config.enable_figure_extraction = True

        # PDFs can be processed in multipage mode for better structure
        config.single_page_mode = False

        # Moderate chunk size for PDFs
        config.chunk_size = 1000
        config.overlap_size = 100

        self.logger.debug("Applied PDF optimizations: OCR, table structure, and figure extraction enabled")

    def _optimize_for_word_document(self, config: DoclingConfig) -> None:
        """Optimize configuration for Word documents (DOCX/DOC)."""
        # Word documents have good structure, less need for OCR
        config.enable_ocr = False  # Word docs have embedded text
        config.enable_table_structure = True  # Tables are common in Word docs
        config.enable_figure_extraction = True

        # Word docs benefit from multipage processing
        config.single_page_mode = False

        # Larger chunk size for structured text
        config.chunk_size = 1500
        config.overlap_size = 150

        self.logger.debug("Applied Word document optimizations: OCR disabled, table/figure extraction enabled")

    def _optimize_for_powerpoint(self, config: DoclingConfig) -> None:
        """Optimize configuration for PowerPoint presentations."""
        # PowerPoint presentations focus on visual content
        config.enable_ocr = True  # Slides may have text as images
        config.enable_table_structure = False  # Less common in presentations
        config.enable_figure_extraction = True  # Important for slides

        # Single page mode can be useful for slide-by-slide processing
        config.single_page_mode = True

        # Smaller chunks for slide content
        config.chunk_size = 800
        config.overlap_size = 80

        self.logger.debug("Applied PowerPoint optimizations: OCR and figure extraction enabled, tables disabled")

    def _optimize_for_image(self, config: DoclingConfig) -> None:
        """Optimize configuration for image files."""
        # Images require OCR to extract text
        config.enable_ocr = True
        config.enable_table_structure = False  # Raw images unlikely to have structured tables
        config.enable_figure_extraction = False  # The whole image is the figure

        # Single page mode for individual images
        config.single_page_mode = True

        # Smaller chunks for OCR text
        config.chunk_size = 500
        config.overlap_size = 50

        self.logger.debug("Applied image optimizations: OCR enabled, structure detection disabled")

    def _optimize_for_excel(self, config: DoclingConfig) -> None:
        """Optimize configuration for Excel files."""
        # Excel files are primarily tabular data
        config.enable_ocr = False  # Excel has embedded text
        config.enable_table_structure = True  # Primary content type
        config.enable_figure_extraction = True  # Charts and graphs

        # Process sheet by sheet
        config.single_page_mode = True

        # Larger chunks for tabular data
        config.chunk_size = 2000
        config.overlap_size = 0  # Minimal overlap for structured data

        self.logger.debug("Applied Excel optimizations: table structure enabled, optimized for tabular data")

    def _optimize_for_text(self, config: DoclingConfig) -> None:
        """Optimize configuration for plain text files."""
        # Plain text files have no complex structure
        config.enable_ocr = False  # Already text
        config.enable_table_structure = False  # Plain text unlikely to have tables
        config.enable_figure_extraction = False  # No figures in plain text

        # Process as single document
        config.single_page_mode = False

        # Larger chunks for continuous text
        config.chunk_size = 2000
        config.overlap_size = 200

        self.logger.debug("Applied text file optimizations: minimal processing for plain text")

    def _optimize_for_unknown(self, config: DoclingConfig, file_type: str) -> None:
        """Optimize configuration for unknown file types."""
        self.logger.warning(f"Unknown file type {file_type}, using conservative configuration")

        # Use conservative defaults for unknown types
        config.enable_ocr = True  # Better to have OCR and not need it
        config.enable_table_structure = True
        config.enable_figure_extraction = True

        # Conservative processing mode
        config.single_page_mode = False

        # Standard chunk size
        config.chunk_size = 1000
        config.overlap_size = 100

    def _apply_custom_settings(self, config: DoclingConfig, custom_settings: Dict[str, Any]) -> None:
        """
        Apply custom settings to override default optimizations.

        Args:
            config: DoclingConfig to modify
            custom_settings: Dictionary of custom settings to apply
        """
        for key, value in custom_settings.items():
            if hasattr(config, key):
                setattr(config, key, value)
                self.logger.debug(f"Applied custom setting: {key} = {value}")
            else:
                self.logger.warning(f"Unknown config setting ignored: {key} = {value}")

    def get_optimization_recommendations(self, file_type: str) -> Dict[str, Any]:
        """
        Get optimization recommendations for a specific file type.

        Args:
            file_type: MIME type string

        Returns:
            Dictionary with optimization recommendations
        """
        recommendations = {"file_type": file_type, "recommended_settings": {}, "reasoning": [], "performance_tips": []}

        # Generate recommendations based on file type
        if file_type == IngestionMimeType.PDF.value:
            recommendations["recommended_settings"] = {
                "enable_ocr": True,
                "enable_table_structure": True,
                "enable_figure_extraction": True,
                "single_page_mode": False,
            }
            recommendations["reasoning"] = [
                "PDFs may contain scanned content requiring OCR",
                "Tables and figures are common in PDF documents",
                "Multipage processing preserves document structure",
            ]
            recommendations["performance_tips"] = [
                "Consider single-page mode for very large PDFs",
                "Disable OCR if all PDFs are text-based for faster processing",
            ]

        elif file_type in [IngestionMimeType.DOCX.value, IngestionMimeType.DOC.value]:
            recommendations["recommended_settings"] = {
                "enable_ocr": False,
                "enable_table_structure": True,
                "enable_figure_extraction": True,
                "single_page_mode": False,
            }
            recommendations["reasoning"] = [
                "Word documents have embedded text, OCR not needed",
                "Tables and embedded objects are common",
                "Document structure should be preserved",
            ]

        elif file_type in [IngestionMimeType.PNG.value, IngestionMimeType.JPG.value]:
            recommendations["recommended_settings"] = {
                "enable_ocr": True,
                "enable_table_structure": False,
                "enable_figure_extraction": False,
                "single_page_mode": True,
            }
            recommendations["reasoning"] = [
                "Images require OCR for text extraction",
                "Raw images rarely contain structured tables",
                "The entire image is the primary content",
            ]

        # Add more file type recommendations as needed...

        return recommendations

    def validate_config(self, config: DoclingConfig) -> Dict[str, Any]:
        """
        Validate a DoclingConfig and provide feedback.

        Args:
            config: DoclingConfig to validate

        Returns:
            Dictionary with validation results
        """
        validation: Dict[str, Any] = {"is_valid": True, "warnings": [], "errors": [], "suggestions": []}

        # Check for potential issues
        if config.chunk_size is not None and config.chunk_size < 100:
            cast(List[str], validation["warnings"]).append("Very small chunk size may result in fragmented text")

        if config.chunk_size is not None and config.chunk_size > 5000:
            cast(List[str], validation["warnings"]).append("Very large chunk size may impact processing performance")

        if (
            config.overlap_size is not None
            and config.chunk_size is not None
            and config.overlap_size >= config.chunk_size
        ):
            cast(List[str], validation["errors"]).append("Overlap size cannot be greater than or equal to chunk size")
            validation["is_valid"] = False

        if config.enable_ocr and config.single_page_mode:
            cast(List[str], validation["suggestions"]).append(
                "OCR with single-page mode may work well for image-heavy documents"
            )

        if not any([config.enable_ocr, config.enable_table_structure, config.enable_figure_extraction]):
            cast(List[str], validation["warnings"]).append(
                "All extraction features disabled - minimal processing will occur"
            )

        return validation

    def get_config_summary(self, config: DoclingConfig) -> Dict[str, Any]:
        """
        Get a human-readable summary of a configuration.

        Args:
            config: DoclingConfig to summarize

        Returns:
            Dictionary with configuration summary
        """
        return {
            "provider": config.provider.value,
            "output_format": config.output_format.value,
            "processing_features": {
                "ocr_enabled": config.enable_ocr,
                "table_detection": config.enable_table_structure,
                "figure_extraction": config.enable_figure_extraction,
            },
            "processing_mode": {"single_page_mode": config.single_page_mode, "pages_to_extract": config.extract_pages},
            "chunking_strategy": {
                "chunk_size": config.chunk_size,
                "overlap_size": config.overlap_size,
                "overlap_percentage": (
                    round((config.overlap_size / config.chunk_size) * 100, 1)
                    if config.chunk_size is not None and config.overlap_size is not None and config.chunk_size > 0
                    else 0
                ),
            },
        }

    def health_check(self) -> Dict[str, str]:
        """
        Check health of processing config manager.

        Returns:
            Dictionary with health status
        """
        return {"processing_config_manager": "healthy", "mime_type_support": "available"}
