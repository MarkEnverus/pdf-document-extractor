"""
Comprehensive tests for AssetStorageService covering figure and table storage functionality.
"""

import io
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from PIL import Image

from src.services.asset_storage_service import AssetStorageService
from src.models.extraction_models import ExtractedFigure, ExtractedTable
from idp_file_management.models.mime_type import IngestionMimeType


@pytest.fixture
def mock_storage_service():
    """Mock storage service for testing."""
    mock = Mock()
    mock.upload_content = Mock(return_value="s3://test-bucket/test/path/file.ext")
    mock.health_check = Mock(return_value=True)
    return mock


@pytest.fixture
def mock_image_analysis_service():
    """Mock image analysis service for testing."""
    mock = Mock()
    mock.analyze_image.return_value = {
        "model_used": "anthropic.claude-3-sonnet-20240229-v1:0",
        "analysis_summary": "Test analysis result",
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "processing_time_ms": 500,
    }
    mock.health_check.return_value = {
        "bedrock_image_analysis_service": "healthy"
    }
    return mock


@pytest.fixture
def asset_storage_service(mock_storage_service, mock_image_analysis_service):
    """Create AssetStorageService instance for testing."""
    return AssetStorageService(
        storage_service=mock_storage_service,
        image_analysis_service=mock_image_analysis_service
    )


@pytest.fixture
def mock_docling_doc():
    """Mock Docling document for testing."""
    doc = Mock()

    # Mock picture with get_image method
    mock_picture = Mock()
    mock_picture_image = Image.new("RGB", (100, 100), color="red")
    mock_picture.get_image = Mock(return_value=mock_picture_image)
    mock_picture.text = "Figure caption"
    mock_picture.label = Mock(value="picture")
    mock_picture.prov = [Mock(page_no=1, bbox=Mock(l=10, t=20, r=110, b=120, coord_origin=Mock(value="TOP_LEFT")))]

    doc.pictures = [mock_picture]

    # Mock table with export_to_dataframe method
    import pandas as pd
    mock_table = Mock()
    mock_table.export_to_dataframe = Mock(return_value=pd.DataFrame({"A": [1, 2], "B": [3, 4]}))
    mock_table.text = "Table caption"
    mock_table.label = Mock(value="table")
    mock_table.prov = [Mock(page_no=1, bbox=Mock(l=10, t=20, r=110, b=120, coord_origin=Mock(value="TOP_LEFT")))]
    mock_table.data = Mock(table=Mock(
        num_rows=2,
        num_cols=2,
        table_cells=[]
    ))

    doc.tables = [mock_table]

    return doc


@pytest.fixture
def sample_extracted_figure():
    """Create sample ExtractedFigure for testing."""
    return ExtractedFigure(
        figure_id="test_figure_1",
        page_number=1,
        caption="Test figure caption",
        alt_text=None,
        label="picture",
        bbox={"left": 10, "top": 20, "right": 110, "bottom": 120, "coord_origin": "TOP_LEFT"},
        s3_image_path=None,
        s3_metadata_path="",
        image_format=None,
        image_size=None,
        extraction_method="docling_get_image",
        extracted_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_extracted_table():
    """Create sample ExtractedTable for testing."""
    return ExtractedTable(
        table_id="test_table_1",
        page_number=1,
        caption="Test table caption",
        label="table",
        num_rows=2,
        num_cols=2,
        headers=["A", "B"],
        bbox={"left": 10, "top": 20, "right": 110, "bottom": 120, "coord_origin": "TOP_LEFT"},
        s3_csv_path=None,
        s3_metadata_path="",
        has_merged_cells=False,
        cell_count=4,
        extraction_method="docling_export",
        extracted_at=datetime.now(timezone.utc)
    )


class TestAssetStorageServiceInit:
    """Test AssetStorageService initialization."""

    def test_init_success(self, asset_storage_service, mock_storage_service, mock_image_analysis_service):
        """Test successful initialization with all dependencies."""
        assert asset_storage_service.storage_service is mock_storage_service
        assert asset_storage_service.image_analysis_service is mock_image_analysis_service
        assert asset_storage_service.logger is not None


class TestStoreExtractedFigures:
    """Test store_extracted_figures method."""

    def test_store_extracted_figures_success(self, asset_storage_service, mock_docling_doc, sample_extracted_figure):
        """Test successful figure storage with all steps."""
        figures = [sample_extracted_figure]

        with patch.object(asset_storage_service, '_extract_figure_image_data', return_value=b"mock_image_data"), \
             patch.object(asset_storage_service, '_store_figure_image'), \
             patch.object(asset_storage_service, '_analyze_figure_with_bedrock', return_value={"analysis_summary": "Test"}), \
             patch.object(asset_storage_service, '_store_figure_analysis_text'), \
             patch.object(asset_storage_service, '_store_figure_metadata'):

            result = asset_storage_service.store_extracted_figures(
                doc=mock_docling_doc,
                extracted_figures=figures,
                project_id="proj-123",
                upload_id="upload-456",
                page_number=1
            )

        assert len(result) == 1
        assert result[0].figure_id == "test_figure_1"

    def test_store_extracted_figures_skip_already_stored(self, asset_storage_service, mock_docling_doc, sample_extracted_figure):
        """Test skipping figures that are already stored."""
        sample_extracted_figure.s3_image_path = "s3://bucket/existing/path.png"
        figures = [sample_extracted_figure]

        result = asset_storage_service.store_extracted_figures(
            doc=mock_docling_doc,
            extracted_figures=figures,
            project_id="proj-123",
            upload_id="upload-456",
            page_number=1
        )

        assert len(result) == 1
        assert result[0].s3_image_path == "s3://bucket/existing/path.png"

    def test_store_extracted_figures_no_image_data(self, asset_storage_service, mock_docling_doc, sample_extracted_figure):
        """Test handling figures with no extractable image data."""
        figures = [sample_extracted_figure]

        with patch.object(asset_storage_service, '_extract_figure_image_data', return_value=None), \
             patch.object(asset_storage_service, '_store_figure_metadata'):

            result = asset_storage_service.store_extracted_figures(
                doc=mock_docling_doc,
                extracted_figures=figures,
                project_id="proj-123",
                upload_id="upload-456",
                page_number=1
            )

        assert len(result) == 1
        assert result[0].s3_image_path is None

    def test_store_extracted_figures_handles_errors(self, asset_storage_service, mock_docling_doc, sample_extracted_figure):
        """Test error handling during figure processing."""
        figures = [sample_extracted_figure]

        with patch.object(asset_storage_service, '_extract_figure_image_data', side_effect=Exception("Test error")):
            result = asset_storage_service.store_extracted_figures(
                doc=mock_docling_doc,
                extracted_figures=figures,
                project_id="proj-123",
                upload_id="upload-456",
                page_number=1
            )

        # Should still return the figure even if processing failed
        assert len(result) == 1


class TestStoreExtractedTables:
    """Test store_extracted_tables method."""

    def test_store_extracted_tables_success(self, asset_storage_service, mock_docling_doc, sample_extracted_table):
        """Test successful table storage with all steps."""
        tables = [sample_extracted_table]

        with patch.object(asset_storage_service, '_extract_table_data', return_value="A,B\n1,3\n2,4"), \
             patch.object(asset_storage_service, '_store_table_csv'), \
             patch.object(asset_storage_service, '_store_table_metadata'):

            result = asset_storage_service.store_extracted_tables(
                doc=mock_docling_doc,
                extracted_tables=tables,
                project_id="proj-123",
                upload_id="upload-456",
                page_number=1
            )

        assert len(result) == 1
        assert result[0].table_id == "test_table_1"

    def test_store_extracted_tables_skip_already_stored(self, asset_storage_service, mock_docling_doc, sample_extracted_table):
        """Test skipping tables that are already stored."""
        sample_extracted_table.s3_csv_path = "s3://bucket/existing/path.csv"
        tables = [sample_extracted_table]

        result = asset_storage_service.store_extracted_tables(
            doc=mock_docling_doc,
            extracted_tables=tables,
            project_id="proj-123",
            upload_id="upload-456",
            page_number=1
        )

        assert len(result) == 1
        assert result[0].s3_csv_path == "s3://bucket/existing/path.csv"

    def test_store_extracted_tables_no_csv_data(self, asset_storage_service, mock_docling_doc, sample_extracted_table):
        """Test handling tables with no extractable CSV data."""
        tables = [sample_extracted_table]

        with patch.object(asset_storage_service, '_extract_table_data', return_value=None), \
             patch.object(asset_storage_service, '_store_table_metadata'):

            result = asset_storage_service.store_extracted_tables(
                doc=mock_docling_doc,
                extracted_tables=tables,
                project_id="proj-123",
                upload_id="upload-456",
                page_number=1
            )

        assert len(result) == 1
        assert result[0].s3_csv_path is None

    def test_store_extracted_tables_handles_errors(self, asset_storage_service, mock_docling_doc, sample_extracted_table):
        """Test error handling during table processing."""
        tables = [sample_extracted_table]

        with patch.object(asset_storage_service, '_extract_table_data', side_effect=Exception("Test error")):
            result = asset_storage_service.store_extracted_tables(
                doc=mock_docling_doc,
                extracted_tables=tables,
                project_id="proj-123",
                upload_id="upload-456",
                page_number=1
            )

        # Should still return the table even if processing failed
        assert len(result) == 1


class TestExtractFigureImageData:
    """Test _extract_figure_image_data and _get_figure_image_data methods."""

    def test_extract_figure_image_data_success(self, asset_storage_service, mock_docling_doc):
        """Test successful figure image extraction."""
        # Create figure with proper ID format that includes index 0
        figure = ExtractedFigure(
            figure_id="request_figure_0_abc123",  # Index 0 matches mock_docling_doc.pictures[0]
            page_number=1,
            caption="Test",
            alt_text=None,
            label="picture",
            bbox=None,
            s3_image_path=None,
            s3_metadata_path="",
            image_format=None,
            image_size=None,
            extraction_method="docling",
            extracted_at=datetime.now(timezone.utc)
        )

        result = asset_storage_service._extract_figure_image_data(mock_docling_doc, figure)

        assert result is not None
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_extract_figure_image_data_invalid_index(self, asset_storage_service, mock_docling_doc):
        """Test handling invalid figure index."""
        figure = ExtractedFigure(
            figure_id="invalid_figure_999",  # Index 999 doesn't exist
            page_number=1,
            caption=None,
            alt_text=None,
            label="picture",
            bbox=None,
            s3_image_path=None,
            s3_metadata_path="",
            image_format=None,
            image_size=None,
            extraction_method="docling",
            extracted_at=datetime.now(timezone.utc)
        )

        result = asset_storage_service._extract_figure_image_data(mock_docling_doc, figure)
        assert result is None

    def test_get_figure_image_data_no_get_image_method(self, asset_storage_service):
        """Test handling figure without get_image method."""
        doc = Mock()
        figure = Mock(spec=[])  # No get_image method

        result = asset_storage_service._get_figure_image_data(doc, figure, "test_id")
        assert result is None

    def test_get_figure_image_data_returns_none(self, asset_storage_service):
        """Test handling when get_image returns None."""
        doc = Mock()
        figure = Mock()
        figure.get_image = Mock(return_value=None)

        result = asset_storage_service._get_figure_image_data(doc, figure, "test_id")
        assert result is None

    def test_get_figure_image_data_unexpected_type(self, asset_storage_service):
        """Test handling unexpected image type."""
        doc = Mock()
        figure = Mock()
        figure.get_image = Mock(return_value="not_an_image")  # Wrong type

        result = asset_storage_service._get_figure_image_data(doc, figure, "test_id")
        assert result is None

    def test_get_figure_image_data_exception(self, asset_storage_service):
        """Test error handling in image extraction."""
        doc = Mock()
        figure = Mock()
        figure.get_image = Mock(side_effect=Exception("Test error"))

        result = asset_storage_service._get_figure_image_data(doc, figure, "test_id")
        assert result is None


class TestExtractTableData:
    """Test _extract_table_data and _get_table_export_data methods."""

    def test_extract_table_data_success(self, asset_storage_service, mock_docling_doc):
        """Test successful table data extraction."""
        # Create table with proper ID format that includes index 0
        table = ExtractedTable(
            table_id="request_table_0_abc123",  # Index 0 matches mock_docling_doc.tables[0]
            page_number=1,
            caption="Test",
            label="table",
            num_rows=2,
            num_cols=2,
            headers=["A", "B"],
            bbox=None,
            s3_csv_path=None,
            s3_metadata_path="",
            has_merged_cells=False,
            cell_count=4,
            extraction_method="docling",
            extracted_at=datetime.now(timezone.utc)
        )

        result = asset_storage_service._extract_table_data(mock_docling_doc, table)

        assert result is not None
        assert isinstance(result, str)
        assert "A,B" in result  # CSV header
        assert "1,3" in result  # First row

    def test_extract_table_data_invalid_index(self, asset_storage_service, mock_docling_doc):
        """Test handling invalid table index."""
        table = ExtractedTable(
            table_id="invalid_table_999",  # Index 999 doesn't exist
            page_number=1,
            caption=None,
            label="table",
            num_rows=0,
            num_cols=0,
            headers=[],
            bbox=None,
            s3_csv_path=None,
            s3_metadata_path="",
            has_merged_cells=False,
            cell_count=0,
            extraction_method="docling",
            extracted_at=datetime.now(timezone.utc)
        )

        result = asset_storage_service._extract_table_data(mock_docling_doc, table)
        assert result is None

    def test_get_table_export_data_no_export_method(self, asset_storage_service):
        """Test handling table without export_to_dataframe method."""
        doc = Mock()
        table = Mock(spec=[])  # No export_to_dataframe method

        result = asset_storage_service._get_table_export_data(doc, table, "test_id")
        assert result is None

    def test_get_table_export_data_returns_none(self, asset_storage_service):
        """Test handling when export_to_dataframe returns None."""
        doc = Mock()
        table = Mock()
        table.export_to_dataframe = Mock(return_value=None)

        result = asset_storage_service._get_table_export_data(doc, table, "test_id")
        assert result is None

    def test_get_table_export_data_empty_dataframe(self, asset_storage_service):
        """Test handling empty dataframe."""
        import pandas as pd
        doc = Mock()
        table = Mock()
        table.export_to_dataframe = Mock(return_value=pd.DataFrame())

        result = asset_storage_service._get_table_export_data(doc, table, "test_id")
        assert result is None

    def test_get_table_export_data_exception(self, asset_storage_service):
        """Test error handling in table extraction."""
        doc = Mock()
        table = Mock()
        table.export_to_dataframe = Mock(side_effect=Exception("Test error"))

        result = asset_storage_service._get_table_export_data(doc, table, "test_id")
        assert result is None


class TestStoreFigureImage:
    """Test _store_figure_image method."""

    def test_store_figure_image_success(self, asset_storage_service, sample_extracted_figure):
        """Test successful figure image storage."""
        # Create mock image data
        img = Image.new("RGB", (100, 100), color="red")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        image_data = img_bytes.getvalue()

        asset_storage_service._store_figure_image(
            figure=sample_extracted_figure,
            image_data=image_data,
            project_id="proj-123",
            upload_id="upload-456",
            page_number=1
        )

        assert sample_extracted_figure.s3_image_path == "s3://test-bucket/test/path/file.ext"
        assert sample_extracted_figure.image_format == IngestionMimeType.PNG
        assert sample_extracted_figure.image_size is not None
        assert sample_extracted_figure.image_size["width"] == 100
        assert sample_extracted_figure.image_size["height"] == 100

    def test_store_figure_image_handles_errors(self, asset_storage_service, sample_extracted_figure, mock_storage_service):
        """Test error handling during image storage."""
        mock_storage_service.upload_content = Mock(side_effect=Exception("S3 error"))

        # Should not raise exception, just log error
        asset_storage_service._store_figure_image(
            figure=sample_extracted_figure,
            image_data=b"test_data",
            project_id="proj-123",
            upload_id="upload-456",
            page_number=1
        )

        # Figure paths should remain None/empty on error
        assert sample_extracted_figure.s3_image_path is None


class TestStoreTableCsv:
    """Test _store_table_csv method."""

    def test_store_table_csv_success(self, asset_storage_service, sample_extracted_table):
        """Test successful table CSV storage."""
        csv_data = "A,B\n1,3\n2,4"

        asset_storage_service._store_table_csv(
            table=sample_extracted_table,
            csv_data=csv_data,
            project_id="proj-123",
            upload_id="upload-456",
            page_number=1
        )

        assert sample_extracted_table.s3_csv_path == "s3://test-bucket/test/path/file.ext"

    def test_store_table_csv_handles_errors(self, asset_storage_service, sample_extracted_table, mock_storage_service):
        """Test error handling during CSV storage."""
        mock_storage_service.upload_content = Mock(side_effect=Exception("S3 error"))

        # Should not raise exception, just log error
        asset_storage_service._store_table_csv(
            table=sample_extracted_table,
            csv_data="test,data",
            project_id="proj-123",
            upload_id="upload-456",
            page_number=1
        )

        # Table path should remain None on error
        assert sample_extracted_table.s3_csv_path is None


class TestStoreFigureAnalysisText:
    """Test _store_figure_analysis_text method."""

    def test_store_figure_analysis_text_success(self, asset_storage_service, sample_extracted_figure):
        """Test successful figure analysis text storage."""
        analysis_result = {"analysis_summary": "Test analysis summary"}

        asset_storage_service._store_figure_analysis_text(
            figure=sample_extracted_figure,
            analysis_result=analysis_result,
            project_id="proj-123",
            upload_id="upload-456",
            page_number=1
        )

        # Verify storage service was called
        asset_storage_service.storage_service.upload_content.assert_called()

    def test_store_figure_analysis_text_handles_errors(self, asset_storage_service, sample_extracted_figure, mock_storage_service):
        """Test error handling during analysis text storage."""
        mock_storage_service.upload_content = Mock(side_effect=Exception("S3 error"))
        analysis_result = {"analysis_summary": "Test"}

        # Should not raise exception, just log error
        asset_storage_service._store_figure_analysis_text(
            figure=sample_extracted_figure,
            analysis_result=analysis_result,
            project_id="proj-123",
            upload_id="upload-456",
            page_number=1
        )


class TestStoreFigureMetadata:
    """Test _store_figure_metadata method."""

    def test_store_figure_metadata_success(self, asset_storage_service, sample_extracted_figure):
        """Test successful figure metadata storage."""
        img = Image.new("RGB", (100, 100))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        image_data = img_bytes.getvalue()

        sample_extracted_figure.image_size = {"width": 100, "height": 100}
        analysis_result = {"analysis_summary": "Test analysis"}

        asset_storage_service._store_figure_metadata(
            figure=sample_extracted_figure,
            image_data=image_data,
            analysis_result=analysis_result,
            project_id="proj-123",
            upload_id="upload-456",
            page_number=1
        )

        assert sample_extracted_figure.s3_metadata_path == "s3://test-bucket/test/path/file.ext"

    def test_store_figure_metadata_no_image_data(self, asset_storage_service, sample_extracted_figure):
        """Test metadata storage with no image data."""
        asset_storage_service._store_figure_metadata(
            figure=sample_extracted_figure,
            image_data=None,
            analysis_result=None,
            project_id="proj-123",
            upload_id="upload-456",
            page_number=1
        )

        assert sample_extracted_figure.s3_metadata_path == "s3://test-bucket/test/path/file.ext"

    def test_store_figure_metadata_handles_errors(self, asset_storage_service, sample_extracted_figure, mock_storage_service):
        """Test error handling during metadata storage."""
        mock_storage_service.upload_content = Mock(side_effect=Exception("S3 error"))

        # Should not raise exception, just log error
        asset_storage_service._store_figure_metadata(
            figure=sample_extracted_figure,
            image_data=None,
            analysis_result=None,
            project_id="proj-123",
            upload_id="upload-456",
            page_number=1
        )


class TestAnalyzeFigureWithBedrock:
    """Test _analyze_figure_with_bedrock method."""

    def test_analyze_figure_success(self, asset_storage_service, sample_extracted_figure, mock_image_analysis_service):
        """Test successful Bedrock figure analysis."""
        result = asset_storage_service._analyze_figure_with_bedrock(
            figure=sample_extracted_figure,
            image_data=b"test_image_data",
            project_id="proj-123",
            upload_id="upload-456"
        )

        assert result is not None
        assert "model_used" in result
        assert "analysis_summary" in result
        assert result["analysis_summary"] == "Test analysis result"

    def test_analyze_figure_handles_errors(self, asset_storage_service, sample_extracted_figure, mock_image_analysis_service):
        """Test error handling during Bedrock analysis."""
        mock_image_analysis_service.analyze_image.side_effect = Exception("Bedrock error")

        result = asset_storage_service._analyze_figure_with_bedrock(
            figure=sample_extracted_figure,
            image_data=b"test_image_data",
            project_id="proj-123",
            upload_id="upload-456"
        )

        # Should return error result instead of raising exception
        assert result is not None
        assert "analysis_error" in result
        assert "Bedrock error" in result["analysis_error"]


class TestStoreTableMetadata:
    """Test _store_table_metadata method."""

    def test_store_table_metadata_success(self, asset_storage_service, sample_extracted_table):
        """Test successful table metadata storage."""
        csv_data = "A,B\n1,3\n2,4"

        asset_storage_service._store_table_metadata(
            table=sample_extracted_table,
            csv_data=csv_data,
            project_id="proj-123",
            upload_id="upload-456",
            page_number=1
        )

        assert sample_extracted_table.s3_metadata_path == "s3://test-bucket/test/path/file.ext"

    def test_store_table_metadata_no_csv_data(self, asset_storage_service, sample_extracted_table):
        """Test metadata storage with no CSV data."""
        asset_storage_service._store_table_metadata(
            table=sample_extracted_table,
            csv_data=None,
            project_id="proj-123",
            upload_id="upload-456",
            page_number=1
        )

        assert sample_extracted_table.s3_metadata_path == "s3://test-bucket/test/path/file.ext"

    def test_store_table_metadata_handles_errors(self, asset_storage_service, sample_extracted_table, mock_storage_service):
        """Test error handling during metadata storage."""
        mock_storage_service.upload_content = Mock(side_effect=Exception("S3 error"))

        # Should not raise exception, just log error
        asset_storage_service._store_table_metadata(
            table=sample_extracted_table,
            csv_data=None,
            project_id="proj-123",
            upload_id="upload-456",
            page_number=1
        )


class TestGenerateAssetS3Path:
    """Test _generate_asset_s3_path method."""

    @patch("src.services.asset_storage_service.settings")
    def test_generate_asset_s3_path_figures(self, mock_settings, asset_storage_service):
        """Test S3 path generation for figures."""
        mock_settings.S3_BUCKET_NAME = "test-bucket"

        bucket, key, base_path = asset_storage_service._generate_asset_s3_path(
            project_id="proj-123",
            upload_id="upload-456",
            page_number=1,
            filename="figure.png",
            asset_type="figures"
        )

        assert bucket == "test-bucket"
        assert key == "proj-123/upload-456/1/figures/figure.png"
        assert base_path == "s3://test-bucket/proj-123/upload-456"

    @patch("src.services.asset_storage_service.settings")
    def test_generate_asset_s3_path_tables(self, mock_settings, asset_storage_service):
        """Test S3 path generation for tables."""
        mock_settings.S3_BUCKET_NAME = "test-bucket"

        bucket, key, base_path = asset_storage_service._generate_asset_s3_path(
            project_id="proj-123",
            upload_id="upload-456",
            page_number=2,
            filename="table.csv",
            asset_type="tables"
        )

        assert bucket == "test-bucket"
        assert key == "proj-123/upload-456/2/tables/table.csv"
        assert base_path == "s3://test-bucket/proj-123/upload-456"


class TestGetImageSize:
    """Test _get_image_size method."""

    def test_get_image_size_success(self, asset_storage_service):
        """Test successful image size extraction."""
        img = Image.new("RGB", (150, 200), color="blue")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        image_data = img_bytes.getvalue()

        result = asset_storage_service._get_image_size(image_data)

        assert result is not None
        assert result["width"] == 150
        assert result["height"] == 200

    def test_get_image_size_invalid_data(self, asset_storage_service):
        """Test handling invalid image data."""
        result = asset_storage_service._get_image_size(b"not_an_image")
        assert result is None

    def test_get_image_size_empty_data(self, asset_storage_service):
        """Test handling empty data."""
        result = asset_storage_service._get_image_size(b"")
        assert result is None


class TestHealthCheck:
    """Test health_check method."""

    def test_health_check_all_healthy(self, asset_storage_service, mock_storage_service, mock_image_analysis_service):
        """Test health check when all services are healthy."""
        result = asset_storage_service.health_check()

        assert result["asset_storage_service"] == "healthy"
        assert result["storage_backend"] == "s3"
        assert result["s3_service"] == "healthy"
        assert result["image_analysis_service"] == "healthy"

    def test_health_check_s3_unhealthy(self, asset_storage_service, mock_storage_service):
        """Test health check when S3 service is unhealthy."""
        mock_storage_service.health_check.side_effect = Exception("S3 error")

        result = asset_storage_service.health_check()

        assert result["asset_storage_service"] == "healthy"
        assert result["s3_service"] == "unhealthy"

    def test_health_check_image_analysis_unhealthy(self, asset_storage_service, mock_image_analysis_service):
        """Test health check when image analysis service is unhealthy."""
        mock_image_analysis_service.health_check.side_effect = Exception("Bedrock error")

        result = asset_storage_service.health_check()

        assert result["asset_storage_service"] == "healthy"
        assert result["image_analysis_service"] == "unhealthy"

    def test_health_check_no_health_check_method(self, asset_storage_service, mock_storage_service):
        """Test health check when service doesn't have health_check method."""
        del mock_storage_service.health_check

        result = asset_storage_service.health_check()

        assert result["asset_storage_service"] == "healthy"
        assert result["s3_service"] == "available"
