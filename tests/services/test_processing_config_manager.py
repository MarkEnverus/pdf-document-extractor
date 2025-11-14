"""Unit tests for ProcessingConfigManager service."""

from unittest.mock import Mock, patch

import pytest

from idp_file_management.models.mime_type import IngestionMimeType
from src.models.docling_models import (
    DoclingConfig,
    DoclingOutputFormatEnum,
    DoclingProviderEnum,
)
from src.services.processing_config_manager import ProcessingConfigManager


class TestProcessingConfigManagerInit:
    """Test ProcessingConfigManager initialization."""

    def test_init_creates_logger(self) -> None:
        """Test that initialization sets up logger."""
        manager = ProcessingConfigManager()
        assert manager.logger is not None


class TestCreateDefaultConfig:
    """Test _create_default_config method."""

    def test_create_default_config_returns_config(self) -> None:
        """Test that default config is created with expected values."""
        manager = ProcessingConfigManager()
        config = manager._create_default_config()

        assert isinstance(config, DoclingConfig)
        assert config.provider == DoclingProviderEnum.LOCAL
        assert config.output_format == DoclingOutputFormatEnum.MARKDOWN
        assert config.enable_ocr is True
        assert config.enable_table_structure is True
        assert config.enable_figure_extraction is True
        assert config.single_page_mode is False
        assert config.extract_pages == "all"
        assert config.chunk_size == 1000
        assert config.overlap_size == 100


class TestOptimizeForPDF:
    """Test _optimize_for_pdf method."""

    def test_optimize_for_pdf(self) -> None:
        """Test PDF optimization settings."""
        manager = ProcessingConfigManager()
        config = DoclingConfig(
            provider=DoclingProviderEnum.LOCAL,
            output_format=DoclingOutputFormatEnum.MARKDOWN,
        )

        manager._optimize_for_pdf(config)

        assert config.enable_ocr is True
        assert config.enable_table_structure is True
        assert config.enable_figure_extraction is True
        assert config.single_page_mode is False
        assert config.chunk_size == 1000
        assert config.overlap_size == 100


class TestOptimizeForWordDocument:
    """Test _optimize_for_word_document method."""

    def test_optimize_for_word_document(self) -> None:
        """Test Word document optimization settings."""
        manager = ProcessingConfigManager()
        config = DoclingConfig(
            provider=DoclingProviderEnum.LOCAL,
            output_format=DoclingOutputFormatEnum.MARKDOWN,
        )

        manager._optimize_for_word_document(config)

        assert config.enable_ocr is False  # Word has embedded text
        assert config.enable_table_structure is True
        assert config.enable_figure_extraction is True
        assert config.single_page_mode is False
        assert config.chunk_size == 1500
        assert config.overlap_size == 150


class TestOptimizeForPowerpoint:
    """Test _optimize_for_powerpoint method."""

    def test_optimize_for_powerpoint(self) -> None:
        """Test PowerPoint optimization settings."""
        manager = ProcessingConfigManager()
        config = DoclingConfig(
            provider=DoclingProviderEnum.LOCAL,
            output_format=DoclingOutputFormatEnum.MARKDOWN,
        )

        manager._optimize_for_powerpoint(config)

        assert config.enable_ocr is True  # Slides may have text as images
        assert config.enable_table_structure is False  # Less common in presentations
        assert config.enable_figure_extraction is True
        assert config.single_page_mode is True
        assert config.chunk_size == 800
        assert config.overlap_size == 80


class TestOptimizeForImage:
    """Test _optimize_for_image method."""

    def test_optimize_for_image(self) -> None:
        """Test image optimization settings."""
        manager = ProcessingConfigManager()
        config = DoclingConfig(
            provider=DoclingProviderEnum.LOCAL,
            output_format=DoclingOutputFormatEnum.MARKDOWN,
        )

        manager._optimize_for_image(config)

        assert config.enable_ocr is True  # Images need OCR
        assert config.enable_table_structure is False
        assert config.enable_figure_extraction is False
        assert config.single_page_mode is True
        assert config.chunk_size == 500
        assert config.overlap_size == 50


class TestOptimizeForExcel:
    """Test _optimize_for_excel method."""

    def test_optimize_for_excel(self) -> None:
        """Test Excel optimization settings."""
        manager = ProcessingConfigManager()
        config = DoclingConfig(
            provider=DoclingProviderEnum.LOCAL,
            output_format=DoclingOutputFormatEnum.MARKDOWN,
        )

        manager._optimize_for_excel(config)

        assert config.enable_ocr is False  # Excel has embedded text
        assert config.enable_table_structure is True  # Primary content type
        assert config.enable_figure_extraction is True  # Charts
        assert config.single_page_mode is True
        assert config.chunk_size == 2000
        assert config.overlap_size == 0  # Minimal overlap for structured data


class TestOptimizeForText:
    """Test _optimize_for_text method."""

    def test_optimize_for_text(self) -> None:
        """Test text file optimization settings."""
        manager = ProcessingConfigManager()
        config = DoclingConfig(
            provider=DoclingProviderEnum.LOCAL,
            output_format=DoclingOutputFormatEnum.MARKDOWN,
        )

        manager._optimize_for_text(config)

        assert config.enable_ocr is False  # Already text
        assert config.enable_table_structure is False
        assert config.enable_figure_extraction is False
        assert config.single_page_mode is False
        assert config.chunk_size == 2000
        assert config.overlap_size == 200


class TestOptimizeForUnknown:
    """Test _optimize_for_unknown method."""

    @patch("src.services.processing_config_manager.logger")
    def test_optimize_for_unknown_logs_warning(self, mock_logger: Mock) -> None:
        """Test that unknown file type logs warning."""
        manager = ProcessingConfigManager()
        config = DoclingConfig(
            provider=DoclingProviderEnum.LOCAL,
            output_format=DoclingOutputFormatEnum.MARKDOWN,
        )

        manager._optimize_for_unknown(config, "application/unknown")

        # Check conservative defaults
        assert config.enable_ocr is True
        assert config.enable_table_structure is True
        assert config.enable_figure_extraction is True
        assert config.single_page_mode is False
        assert config.chunk_size == 1000
        assert config.overlap_size == 100


class TestCreateOptimizedConfig:
    """Test create_optimized_config method."""

    def test_create_optimized_config_pdf(self) -> None:
        """Test creating optimized config for PDF."""
        manager = ProcessingConfigManager()
        config = manager.create_optimized_config(IngestionMimeType.PDF.value)

        assert config.enable_ocr is True
        assert config.enable_table_structure is True
        assert config.chunk_size == 1000

    def test_create_optimized_config_docx(self) -> None:
        """Test creating optimized config for DOCX."""
        manager = ProcessingConfigManager()
        config = manager.create_optimized_config(IngestionMimeType.DOCX.value)

        assert config.enable_ocr is False
        assert config.enable_table_structure is True
        assert config.chunk_size == 1500

    def test_create_optimized_config_pptx(self) -> None:
        """Test creating optimized config for PPTX."""
        manager = ProcessingConfigManager()
        config = manager.create_optimized_config(IngestionMimeType.PPTX.value)

        assert config.enable_ocr is True
        assert config.enable_table_structure is False
        assert config.single_page_mode is True

    def test_create_optimized_config_image(self) -> None:
        """Test creating optimized config for PNG/JPG."""
        manager = ProcessingConfigManager()
        config = manager.create_optimized_config(IngestionMimeType.PNG.value)

        assert config.enable_ocr is True
        assert config.enable_table_structure is False
        assert config.single_page_mode is True

    def test_create_optimized_config_with_base_config(self) -> None:
        """Test that base config is used as starting point."""
        manager = ProcessingConfigManager()
        base_config = DoclingConfig(
            provider=DoclingProviderEnum.LOCAL,
            output_format=DoclingOutputFormatEnum.JSON,  # Custom format
            enable_ocr=False,
        )

        config = manager.create_optimized_config(IngestionMimeType.PDF.value, base_config=base_config)

        # Should start from base but apply optimizations
        assert config.output_format == DoclingOutputFormatEnum.JSON
        assert config.enable_ocr is True  # PDF optimization overrides

    def test_create_optimized_config_with_custom_settings(self) -> None:
        """Test that custom settings override optimizations."""
        manager = ProcessingConfigManager()
        custom_settings = {"chunk_size": 5000, "enable_ocr": False}

        config = manager.create_optimized_config(IngestionMimeType.PDF.value, custom_settings=custom_settings)

        # Custom settings should win
        assert config.chunk_size == 5000
        assert config.enable_ocr is False


class TestApplyCustomSettings:
    """Test _apply_custom_settings method."""

    def test_apply_custom_settings_valid_keys(self) -> None:
        """Test applying valid custom settings."""
        manager = ProcessingConfigManager()
        config = DoclingConfig(
            provider=DoclingProviderEnum.LOCAL,
            output_format=DoclingOutputFormatEnum.MARKDOWN,
            chunk_size=1000,
        )

        manager._apply_custom_settings(config, {"chunk_size": 2000, "enable_ocr": False})

        assert config.chunk_size == 2000
        assert config.enable_ocr is False

    @patch("src.services.processing_config_manager.logger")
    def test_apply_custom_settings_invalid_key(self, mock_logger: Mock) -> None:
        """Test that invalid setting keys log warning."""
        manager = ProcessingConfigManager()
        config = DoclingConfig(
            provider=DoclingProviderEnum.LOCAL,
            output_format=DoclingOutputFormatEnum.MARKDOWN,
        )

        manager._apply_custom_settings(config, {"invalid_key": "value"})

        # Config should be unchanged, but warning should be logged


class TestGetOptimizationRecommendations:
    """Test get_optimization_recommendations method."""

    def test_recommendations_for_pdf(self) -> None:
        """Test getting recommendations for PDF."""
        manager = ProcessingConfigManager()
        recommendations = manager.get_optimization_recommendations(IngestionMimeType.PDF.value)

        assert recommendations["file_type"] == IngestionMimeType.PDF.value
        assert "recommended_settings" in recommendations
        assert recommendations["recommended_settings"]["enable_ocr"] is True
        assert "reasoning" in recommendations
        assert len(recommendations["reasoning"]) > 0
        assert "performance_tips" in recommendations

    def test_recommendations_for_word(self) -> None:
        """Test getting recommendations for Word documents."""
        manager = ProcessingConfigManager()
        recommendations = manager.get_optimization_recommendations(IngestionMimeType.DOCX.value)

        assert recommendations["file_type"] == IngestionMimeType.DOCX.value
        assert recommendations["recommended_settings"]["enable_ocr"] is False
        assert len(recommendations["reasoning"]) > 0

    def test_recommendations_for_image(self) -> None:
        """Test getting recommendations for images."""
        manager = ProcessingConfigManager()
        recommendations = manager.get_optimization_recommendations(IngestionMimeType.PNG.value)

        assert recommendations["file_type"] == IngestionMimeType.PNG.value
        assert recommendations["recommended_settings"]["enable_ocr"] is True
        assert recommendations["recommended_settings"]["single_page_mode"] is True


class TestValidateConfig:
    """Test validate_config method."""

    def test_validate_config_valid(self) -> None:
        """Test validation of valid config."""
        manager = ProcessingConfigManager()
        config = DoclingConfig(
            provider=DoclingProviderEnum.LOCAL,
            output_format=DoclingOutputFormatEnum.MARKDOWN,
            chunk_size=1000,
            overlap_size=100,
            enable_ocr=True,
        )

        validation = manager.validate_config(config)

        assert validation["is_valid"] is True
        assert len(validation["errors"]) == 0

    def test_validate_config_small_chunk_warning(self) -> None:
        """Test validation warns about very small chunk size."""
        manager = ProcessingConfigManager()
        config = DoclingConfig(
            provider=DoclingProviderEnum.LOCAL,
            output_format=DoclingOutputFormatEnum.MARKDOWN,
            chunk_size=50,  # Very small
            overlap_size=10,
        )

        validation = manager.validate_config(config)

        assert validation["is_valid"] is True
        assert any("small chunk" in str(w).lower() for w in validation["warnings"])

    def test_validate_config_large_chunk_warning(self) -> None:
        """Test validation warns about very large chunk size."""
        manager = ProcessingConfigManager()
        config = DoclingConfig(
            provider=DoclingProviderEnum.LOCAL,
            output_format=DoclingOutputFormatEnum.MARKDOWN,
            chunk_size=6000,  # Very large
            overlap_size=100,
        )

        validation = manager.validate_config(config)

        assert validation["is_valid"] is True
        assert any("large chunk" in str(w).lower() for w in validation["warnings"])

    def test_validate_config_overlap_greater_than_chunk_error(self) -> None:
        """Test validation errors when overlap >= chunk size."""
        manager = ProcessingConfigManager()
        config = DoclingConfig(
            provider=DoclingProviderEnum.LOCAL,
            output_format=DoclingOutputFormatEnum.MARKDOWN,
            chunk_size=1000,
            overlap_size=1000,  # Equal to chunk size
        )

        validation = manager.validate_config(config)

        assert validation["is_valid"] is False
        assert any("overlap" in str(e).lower() for e in validation["errors"])

    def test_validate_config_all_features_disabled_warning(self) -> None:
        """Test validation warns when all extraction features disabled."""
        manager = ProcessingConfigManager()
        config = DoclingConfig(
            provider=DoclingProviderEnum.LOCAL,
            output_format=DoclingOutputFormatEnum.MARKDOWN,
            enable_ocr=False,
            enable_table_structure=False,
            enable_figure_extraction=False,
        )

        validation = manager.validate_config(config)

        assert validation["is_valid"] is True
        assert any("disabled" in str(w).lower() for w in validation["warnings"])

    def test_validate_config_ocr_with_single_page_suggestion(self) -> None:
        """Test validation suggests OCR with single-page mode."""
        manager = ProcessingConfigManager()
        config = DoclingConfig(
            provider=DoclingProviderEnum.LOCAL,
            output_format=DoclingOutputFormatEnum.MARKDOWN,
            enable_ocr=True,
            single_page_mode=True,
        )

        validation = manager.validate_config(config)

        assert validation["is_valid"] is True
        assert "suggestions" in validation


class TestGetConfigSummary:
    """Test get_config_summary method."""

    def test_get_config_summary_complete(self) -> None:
        """Test getting complete config summary."""
        manager = ProcessingConfigManager()
        config = DoclingConfig(
            provider=DoclingProviderEnum.LOCAL,
            output_format=DoclingOutputFormatEnum.MARKDOWN,
            enable_ocr=True,
            enable_table_structure=True,
            enable_figure_extraction=False,
            single_page_mode=False,
            extract_pages="all",
            chunk_size=1000,
            overlap_size=100,
        )

        summary = manager.get_config_summary(config)

        assert summary["provider"] == DoclingProviderEnum.LOCAL.value
        assert summary["output_format"] == DoclingOutputFormatEnum.MARKDOWN.value
        assert summary["processing_features"]["ocr_enabled"] is True
        assert summary["processing_features"]["table_detection"] is True
        assert summary["processing_features"]["figure_extraction"] is False
        assert summary["processing_mode"]["single_page_mode"] is False
        assert summary["processing_mode"]["pages_to_extract"] == "all"
        assert summary["chunking_strategy"]["chunk_size"] == 1000
        assert summary["chunking_strategy"]["overlap_size"] == 100
        assert summary["chunking_strategy"]["overlap_percentage"] == 10.0

    def test_get_config_summary_zero_chunk_size(self) -> None:
        """Test summary handles zero chunk size without division error."""
        manager = ProcessingConfigManager()
        config = DoclingConfig(
            provider=DoclingProviderEnum.LOCAL,
            output_format=DoclingOutputFormatEnum.MARKDOWN,
            chunk_size=0,  # Zero to test edge case
            overlap_size=0,
        )

        summary = manager.get_config_summary(config)

        assert summary["chunking_strategy"]["overlap_percentage"] == 0


class TestHealthCheck:
    """Test health_check method."""

    def test_health_check_returns_healthy(self) -> None:
        """Test that health check returns healthy status."""
        manager = ProcessingConfigManager()
        health = manager.health_check()

        assert health["processing_config_manager"] == "healthy"
        assert health["mime_type_support"] == "available"
