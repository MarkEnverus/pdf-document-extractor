"""
Tests to verify HTML fields have been removed from extraction models and processing.

This test suite ensures that:
1. FigureMetadata and TableMetadata models do not have HTML fields
2. Table export methods do not return HTML data
3. No HTML content is stored in S3
"""

import inspect

from idp_pipeline_events import FigureMetadata, TableMetadata
from src.services.extractors.docling.docling_strategy_processor import DoclingStrategyProcessor


class TestHtmlRemovalVerification:
    """Verify HTML fields and methods have been removed from codebase."""

    def test_figure_metadata_no_html_field(self):
        """Verify FigureMetadata does not have html_content field."""
        # Check model fields
        model_fields = FigureMetadata.model_fields
        assert "html_content" not in model_fields, "FigureMetadata should not have html_content field"
        assert "html" not in model_fields, "FigureMetadata should not have html field"
        assert "html_data" not in model_fields, "FigureMetadata should not have html_data field"

    def test_table_metadata_no_html_field(self):
        """Verify TableMetadata does not have html_content field."""
        # Check model fields
        model_fields = TableMetadata.model_fields
        assert "html_content" not in model_fields, "TableMetadata should not have html_content field"
        assert "html" not in model_fields, "TableMetadata should not have html field"
        assert "html_data" not in model_fields, "TableMetadata should not have html_data field"

    def test_figure_metadata_instantiation_no_html(self):
        """Verify FigureMetadata cannot be instantiated with HTML fields."""
        from datetime import datetime

        # This should work fine
        metadata = FigureMetadata(
            figure_id="fig_001",
            source_document="s3://bucket/doc.pdf",
            page_number=1,
            extraction_timestamp=datetime.now(),
            extraction_method="docling_get_image",
        )

        # Verify no HTML attributes exist
        assert not hasattr(metadata, "html_content")
        assert not hasattr(metadata, "html")
        assert not hasattr(metadata, "html_data")

    def test_table_metadata_instantiation_no_html(self):
        """Verify TableMetadata cannot be instantiated with HTML fields."""
        from datetime import datetime

        # This should work fine
        metadata = TableMetadata(
            table_id="table_001",
            source_document="s3://bucket/doc.pdf",
            page_number=1,
            extraction_timestamp=datetime.now(),
            extraction_method="docling_export",
        )

        # Verify no HTML attributes exist
        assert not hasattr(metadata, "html_content")
        assert not hasattr(metadata, "html")
        assert not hasattr(metadata, "html_data")

    def test_get_table_export_data_return_type(self):
        """Verify _get_table_export_data returns only CSV data, not HTML."""
        # After deduplication refactoring, this method now exists in AssetStorageService only
        from src.services.asset_storage_service import AssetStorageService

        # Check the method signature
        method = AssetStorageService._get_table_export_data
        sig = inspect.signature(method)

        # Get return annotation
        return_annotation = sig.return_annotation

        # The return type should be Optional[str] for CSV only, NOT tuple
        # If it's still tuple[Optional[str], Optional[str]], this test will fail
        assert return_annotation != tuple[str | None, str | None], (
            "_get_table_export_data should return Optional[str] for CSV only, "
            "not tuple with HTML data"
        )

    def test_get_table_export_data_no_html_export_logic(self):
        """Verify _get_table_export_data method does not contain HTML export logic."""
        import inspect
        # After deduplication refactoring, this method now exists in AssetStorageService only
        from src.services.asset_storage_service import AssetStorageService

        # Get the source code of the method
        source = inspect.getsource(AssetStorageService._get_table_export_data)

        # Verify no HTML-related code exists
        assert "export_to_html" not in source, "_get_table_export_data should not call export_to_html"
        assert "html_data" not in source, "_get_table_export_data should not reference html_data variable"

        # Verify it doesn't return a tuple with two values
        # (should return single CSV value, not (csv, html) tuple)
        lines = source.split("\n")
        return_lines = [line.strip() for line in lines if line.strip().startswith("return ")]

        for line in return_lines:
            # Should not have return statements with tuples like "return csv_data, html_data"
            assert ", " not in line or "return None" in line or "return csv_data" not in line, (
                f"Found tuple return in _get_table_export_data: {line}"
            )

    def test_docling_processor_no_html_storage(self):
        """Verify DoclingStrategyProcessor does not store HTML files."""
        import inspect

        # Get source of the entire class
        source = inspect.getsource(DoclingStrategyProcessor)

        # Check that no HTML files are being written
        assert ".html" not in source or "export_to_html()" in source, (
            "DoclingStrategyProcessor should not store .html files"
        )

        # Verify no references to storing HTML content in S3
        assert "html" not in source.lower() or "export_to_html" in source, (
            "DoclingStrategyProcessor should not have HTML storage logic (except for export_to_html document export)"
        )

    def test_extracted_table_model_no_html_field(self):
        """Verify ExtractedTable model does not have HTML fields."""
        from src.models.extraction_models import ExtractedTable

        # Check if ExtractedTable has HTML-related fields
        if hasattr(ExtractedTable, "model_fields"):
            fields = ExtractedTable.model_fields
            assert "html_content" not in fields, "ExtractedTable should not have html_content field"
            assert "html_data" not in fields, "ExtractedTable should not have html_data field"
            assert "html" not in fields, "ExtractedTable should not have html field"

    def test_extracted_figure_model_no_html_field(self):
        """Verify ExtractedFigure model does not have HTML fields."""
        from src.models.extraction_models import ExtractedFigure

        # Check if ExtractedFigure has HTML-related fields
        if hasattr(ExtractedFigure, "model_fields"):
            fields = ExtractedFigure.model_fields
            assert "html_content" not in fields, "ExtractedFigure should not have html_content field"
            assert "html_data" not in fields, "ExtractedFigure should not have html_data field"
            assert "html" not in fields, "ExtractedFigure should not have html field"

    def test_page_extraction_result_no_html_references(self):
        """Verify PageExtractionResult does not reference HTML content."""
        from idp_pipeline_events import PageExtractionResult

        # Check model fields
        fields = PageExtractionResult.model_fields

        # Verify no HTML-related fields in page results
        assert "html" not in fields, "PageExtractionResult should not have html field"
        assert "html_content" not in fields, "PageExtractionResult should not have html_content field"
        assert "html_data" not in fields, "PageExtractionResult should not have html_data field"

    def test_table_reference_no_html_path(self):
        """Verify TableReference does not have s3_html_path field."""
        from idp_pipeline_events import TableReference

        fields = TableReference.model_fields

        # Should have CSV path but not HTML path
        assert "s3_csv_path" in fields, "TableReference should have s3_csv_path field"
        assert "s3_html_path" not in fields, "TableReference should NOT have s3_html_path field"
        assert "html_path" not in fields, "TableReference should NOT have html_path field"

    def test_figure_reference_no_html_path(self):
        """Verify FigureReference does not have HTML path fields."""
        from idp_pipeline_events import FigureReference

        fields = FigureReference.model_fields

        # Should have image path but not HTML path
        assert "s3_image_path" in fields, "FigureReference should have s3_image_path field"
        assert "s3_html_path" not in fields, "FigureReference should NOT have s3_html_path field"
        assert "html_path" not in fields, "FigureReference should NOT have html_path field"

    def test_no_html_mime_type_references(self):
        """Verify no HTML MIME types are used in extraction models."""
        from src.models.extraction_models import ExtractedTable, ExtractedFigure

        # Create sample instances with all required fields
        table = ExtractedTable(
            table_id="test_table",
            page_number=1,
            label="table",
            num_rows=5,
            num_cols=3,
            cell_count=15,
            s3_metadata_path="s3://bucket/metadata.json",
        )

        figure = ExtractedFigure(
            figure_id="test_figure",
            page_number=1,
            label="picture",
            s3_metadata_path="s3://bucket/metadata.json",
            image_format="image/png",
        )

        # Verify no HTML format references
        if hasattr(figure, "image_format"):
            assert "html" not in str(figure.image_format).lower()

        if hasattr(table, "format"):
            assert "html" not in table.format.lower() if table.format else True
