"""
Asset storage service for handling document figures and tables.

This service extracts figure/table storage logic from DocumentProcessingService
to follow single responsibility principle.
"""

from typing import Any, List, Optional, Tuple
from uuid import uuid4

from docling_core.types import DoclingDocument
from lib.models.mime_type import IngestionMimeType
from lib.logger import Logger
from lib.models.extraction_models import BoundingBox, FigureMetadata, ImageSize, TableMetadata

from src.configs.settings import settings
from src.interfaces.image_analysis import ImageAnalysisService
from src.interfaces.storage import StorageService
from src.models.extraction_models import ExtractedFigure, ExtractedTable
from src.services.storage_utils import store_text_file_to_s3

logger = Logger.get_logger(__name__)


class AssetStorageService:
    """
    Service responsible for storing document assets (figures and tables) to S3.

    This service handles:
    - Extracting figure images and table data from processed documents
    - Uploading assets to S3 with organized path structure
    - Creating and storing detailed metadata for each asset
    - Managing asset lifecycle and error handling
    """

    def __init__(self, storage_service: StorageService, image_analysis_service: ImageAnalysisService):
        """
        Initialize AssetStorageService.

        Args:
            storage_service: StorageService for uploading content to storage
            image_analysis_service: ImageAnalysisService for LLM-based image analysis (required)
        """
        self.storage_service = storage_service
        self.image_analysis_service = image_analysis_service
        self.logger = logger

    def store_extracted_figures(
        self,
        doc: Optional[DoclingDocument],
        extracted_figures: List[ExtractedFigure],
        project_id: str,
        upload_id: str,
        page_number: int,
    ) -> List[ExtractedFigure]:
        """
        Store extracted figures to S3 and update their paths.

        Args:
            doc: Processed Docling document (for actual image extraction)
            extracted_figures: List of ExtractedFigure objects with metadata
            project_id: Project ID for S3 path
            upload_id: Upload ID for S3 path
            page_number: Page number for S3 path

        Returns:
            Updated list of ExtractedFigure objects with S3 paths populated
        """
        updated_figures = []

        for figure in extracted_figures:
            try:
                # Skip if already stored (deduplication check)
                if hasattr(figure, "s3_image_path") and figure.s3_image_path:
                    self.logger.debug(
                        "Skipping figure - already stored",
                        figure_id=figure.figure_id,
                        s3_path=figure.s3_image_path,
                    )
                    updated_figures.append(figure)
                    continue

                # Get the actual image data using DoclingService method
                figure_image_data = self._extract_figure_image_data(doc, figure)

                # Track analysis results separately (not stored on DTO)
                analysis_result = None

                # Store image and perform analysis if we have data
                if figure_image_data:
                    self._store_figure_image(figure, figure_image_data, project_id, upload_id, page_number)
                    # Always perform Bedrock analysis (required)
                    analysis_result = self._analyze_figure_with_bedrock(
                        figure, figure_image_data, project_id=project_id, upload_id=upload_id
                    )
                    # Store LLM analysis as separate text file alongside PNG and metadata
                    analysis_summary = analysis_result.get("analysis_summary") if analysis_result else None
                    if analysis_summary and analysis_summary.strip():
                        self._store_figure_analysis_text(
                            figure, analysis_result, project_id, upload_id, page_number
                        )

                # Create and store metadata (includes analysis results)
                self._store_figure_metadata(
                    figure, figure_image_data, analysis_result, project_id, upload_id, page_number
                )

                updated_figures.append(figure)

            except Exception as e:
                self.logger.error(
                    "Failed to process figure",
                    figure_id=figure.figure_id,
                    project_id=project_id,
                    upload_id=upload_id,
                    error=str(e),
                )
                # Add figure even if processing failed, so we don't lose metadata
                updated_figures.append(figure)

        return updated_figures

    def store_extracted_tables(
        self,
        doc: Any,
        extracted_tables: List[ExtractedTable],
        project_id: str,
        upload_id: str,
        page_number: int,
    ) -> List[ExtractedTable]:
        """
        Store extracted tables to S3 and update their paths.

        Args:
            doc: Processed Docling document (for actual table data extraction)
            extracted_tables: List of ExtractedTable objects with metadata
            project_id: Project ID for S3 path
            upload_id: Upload ID for S3 path
            page_number: Page number for S3 path

        Returns:
            Updated list of ExtractedTable objects with S3 paths populated
        """
        updated_tables = []

        for table in extracted_tables:
            try:
                # Skip if already stored (deduplication check)
                if hasattr(table, "s3_csv_path") and table.s3_csv_path:
                    self.logger.debug(
                        "Skipping table - already stored",
                        table_id=table.table_id,
                        s3_csv_path=table.s3_csv_path,
                    )
                    updated_tables.append(table)
                    continue

                # Get the actual table data using DoclingService method
                csv_data = self._extract_table_data(doc, table)

                # Store CSV data
                if csv_data:
                    self._store_table_csv(table, csv_data, project_id, upload_id, page_number)

                # Create and store metadata
                self._store_table_metadata(table, csv_data, project_id, upload_id, page_number)

                updated_tables.append(table)

            except Exception as e:
                self.logger.error(
                    "Failed to process table",
                    table_id=table.table_id,
                    project_id=project_id,
                    upload_id=upload_id,
                    error=str(e),
                )
                # Add table even if processing failed, so we don't lose metadata
                updated_tables.append(table)

        return updated_tables

    def _extract_figure_image_data(self, doc: Any, figure: ExtractedFigure) -> Optional[bytes]:
        """Extract image data for a figure from the document."""
        try:
            if hasattr(doc, "pictures") and doc.pictures:
                # Find the picture by index (extracted from figure_id)
                figure_idx = int(figure.figure_id.split("_figure_")[1].split("_")[0])
                if 0 <= figure_idx < len(doc.pictures):
                    picture = doc.pictures[figure_idx]
                    return self._get_figure_image_data(doc, picture, figure.figure_id)
        except (ValueError, IndexError) as e:
            self.logger.warning("Could not extract image for figure", figure_id=figure.figure_id, error=str(e))
        return None

    def _get_figure_image_data(self, doc: Any, figure: Any, figure_id: str) -> Optional[bytes]:
        """
        Extract actual image data for a figure using Docling's get_image API.

        Args:
            doc: Processed Docling document
            figure: Docling picture object
            figure_id: Unique figure identifier

        Returns:
            Image bytes in PNG format, or None if extraction fails
        """
        try:
            # Use Docling's PictureItem.get_image API to extract the actual image
            # Note: get_image is called ON the PictureItem (figure), passing the document
            if hasattr(figure, "get_image"):
                image = figure.get_image(doc)
                if image is not None:
                    # Convert PIL Image to PNG bytes
                    import io

                    from PIL import Image

                    # image is a PIL Image object from Docling
                    if hasattr(image, "save"):
                        # PIL Image object
                        img_bytes = io.BytesIO()
                        image.save(img_bytes, format="PNG")
                        return img_bytes.getvalue()
                    else:
                        self.logger.warning(
                            "Unexpected image type for figure", figure_id=figure_id, image_type=type(image).__name__
                        )
                        return None
                else:
                    self.logger.warning("get_image returned None for figure", figure_id=figure_id)
                    return None
            else:
                self.logger.warning("Figure does not have get_image method", figure_id=figure_id)
                return None

        except Exception as e:
            self.logger.error("Failed to extract image data for figure", figure_id=figure_id, error=str(e))
            return None

    def _extract_table_data(self, doc: Any, table: ExtractedTable) -> Optional[str]:
        """Extract CSV data for a table from the document."""
        try:
            if hasattr(doc, "tables") and doc.tables:
                # Find the table by index (extracted from table_id)
                table_idx = int(table.table_id.split("_table_")[1].split("_")[0])
                if 0 <= table_idx < len(doc.tables):
                    doc_table = doc.tables[table_idx]
                    return self._get_table_export_data(doc, doc_table, table.table_id)
        except (ValueError, IndexError) as e:
            self.logger.warning("Could not extract data for table", table_id=table.table_id, error=str(e))
        return None

    def _get_table_export_data(self, doc: Any, table: Any, table_id: str) -> Optional[str]:
        """
        Extract table data in CSV format using Docling's export APIs.

        Args:
            doc: Processed Docling document
            table: Docling table object
            table_id: Unique table identifier

        Returns:
            CSV data as string, or None if extraction fails
        """
        csv_data = None

        try:
            # Try to export table to dataframe and then to CSV
            try:
                if hasattr(table, "export_to_dataframe"):
                    df = table.export_to_dataframe(doc=doc)  # Pass document for better context
                    if df is not None and not df.empty:
                        csv_data = df.to_csv(index=False)
                        self.logger.debug(
                            "Exported table to CSV", table_id=table_id, table_dimensions=f"{df.shape[0]}x{df.shape[1]}"
                        )
            except Exception as e:
                self.logger.warning("Failed to export table to CSV", table_id=table_id, error=str(e))

            return csv_data

        except Exception as e:
            self.logger.error("Failed to export table data", table_id=table_id, error=str(e))
            return None

    def _store_figure_image(
        self, figure: ExtractedFigure, image_data: bytes, project_id: str, upload_id: str, page_number: int
    ) -> None:
        """Store figure image to S3 and update figure object."""
        try:
            # Generate S3 path for image
            img_bucket, img_key, _ = self._generate_asset_s3_path(
                project_id=project_id,
                upload_id=upload_id,
                page_number=page_number,
                filename=f"{figure.figure_id}.png",
                asset_type="figures",
            )

            # Upload image to S3
            s3_image_path = self.storage_service.upload_content(
                image_data, img_bucket, img_key, project_id=project_id, upload_id=upload_id
            )

            # Get image metadata
            image_size = self._get_image_size(image_data)

            # Update figure with image info
            figure.s3_image_path = s3_image_path
            figure.image_format = IngestionMimeType.PNG
            figure.image_size = image_size

            self.logger.info(
                "Stored figure image",
                project_id=project_id,
                upload_id=upload_id,
                s3_image_path=s3_image_path,
                figure_id=figure.figure_id,
            )

        except Exception as e:
            self.logger.error(
                "Failed to store image for figure",
                project_id=project_id,
                upload_id=upload_id,
                figure_id=figure.figure_id,
                error=str(e),
            )

    def _store_table_csv(
        self, table: ExtractedTable, csv_data: str, project_id: str, upload_id: str, page_number: int
    ) -> None:
        """Store table CSV data to S3."""
        try:
            csv_bucket, csv_key, _ = self._generate_asset_s3_path(
                project_id=project_id,
                upload_id=upload_id,
                page_number=page_number,
                filename=f"{table.table_id}.csv",
                asset_type="tables",
            )

            s3_csv_path = self.storage_service.upload_content(
                csv_data.encode("utf-8"), csv_bucket, csv_key, project_id=project_id, upload_id=upload_id
            )

            table.s3_csv_path = s3_csv_path
            self.logger.info(
                "Stored table CSV",
                project_id=project_id,
                upload_id=upload_id,
                s3_csv_path=s3_csv_path,
                table_id=table.table_id,
            )

        except Exception as e:
            self.logger.error(
                "Failed to store CSV for table",
                project_id=project_id,
                upload_id=upload_id,
                table_id=table.table_id,
                error=str(e),
            )


    def _store_figure_analysis_text(
        self,
        figure: ExtractedFigure,
        analysis_result: dict[str, Any],
        project_id: str,
        upload_id: str,
        page_number: int,
    ) -> None:
        """Store figure LLM analysis as a separate text file.

        This creates a {figure_id}.txt file alongside the PNG and metadata JSON,
        containing the LLM analysis summary for easy consumption by downstream services.
        """
        try:
            # Generate S3 path for analysis text file
            text_bucket, text_key, _ = self._generate_asset_s3_path(
                project_id=project_id,
                upload_id=upload_id,
                page_number=page_number,
                filename=f"{figure.figure_id}.txt",
                asset_type="figures",
            )

            # Extract analysis summary
            analysis_summary = analysis_result.get("analysis_summary", "")

            # Upload text file to S3 using shared utility
            store_text_file_to_s3(
                storage_service=self.storage_service,
                logger=self.logger,
                text_content=analysis_summary,
                bucket=text_bucket,
                s3_key=text_key,
                project_id=project_id,
                upload_id=upload_id,
                log_context={"figure_id": figure.figure_id},
            )

        except Exception as e:
            self.logger.error(
                "Failed to store analysis text for figure",
                project_id=project_id,
                upload_id=upload_id,
                figure_id=figure.figure_id,
                error=str(e),
            )

    def _store_figure_metadata(
        self,
        figure: ExtractedFigure,
        image_data: Optional[bytes],
        analysis_result: Optional[dict[str, Any]],
        project_id: str,
        upload_id: str,
        page_number: int,
    ) -> None:
        """Create and store figure metadata to S3."""
        try:
            # Convert dict fields to Pydantic models
            image_size_model = None
            if figure.image_size:
                image_size_model = ImageSize(
                    width=figure.image_size["width"],
                    height=figure.image_size["height"],
                )

            bbox_model = None
            if figure.bbox:
                bbox_model = BoundingBox(
                    left=figure.bbox["left"],
                    top=figure.bbox["top"],
                    right=figure.bbox["right"],
                    bottom=figure.bbox["bottom"],
                    coord_origin=figure.bbox.get("coord_origin", "TOP_LEFT"),
                )

            # Create detailed metadata (includes LLM analysis if available)
            figure_metadata = FigureMetadata(
                figure_id=figure.figure_id,
                source_document=f"s3://{settings.S3_BUCKET_NAME}/{project_id}/{upload_id}",
                page_number=figure.page_number,
                extraction_timestamp=figure.extracted_at,
                caption=figure.caption,
                alt_text=figure.alt_text,
                image_format=figure.image_format,
                image_size=image_size_model,
                file_size_bytes=len(image_data) if image_data else None,
                bbox=bbox_model,
                docling_label=figure.label,
                extraction_method=figure.extraction_method,
                llm_analysis=analysis_result,  # LLM analysis results passed separately
            )

            # Store metadata JSON
            metadata_bucket, metadata_key, _ = self._generate_asset_s3_path(
                project_id=project_id,
                upload_id=upload_id,
                page_number=page_number,
                filename=f"{figure.figure_id}_metadata.json",
                asset_type="figures",
            )

            metadata_content = figure_metadata.model_dump_json(indent=2).encode("utf-8")

            s3_metadata_path = self.storage_service.upload_content(
                metadata_content, metadata_bucket, metadata_key, project_id=project_id, upload_id=upload_id
            )

            figure.s3_metadata_path = s3_metadata_path
            self.logger.debug(
                "Stored figure metadata",
                figure_id=figure.figure_id,
                s3_path=s3_metadata_path,
                project_id=project_id,
                upload_id=upload_id,
            )

        except Exception as e:
            self.logger.error(
                "Failed to store metadata for figure",
                project_id=project_id,
                upload_id=upload_id,
                figure_id=figure.figure_id,
                error=str(e),
            )

    def _analyze_figure_with_bedrock(
        self, figure: ExtractedFigure, image_data: bytes, project_id: str, upload_id: str
    ) -> dict[str, Any]:
        """
        Analyze figure using Bedrock LLM and return analysis results.

        This is required for all images - analysis failures are logged but don't block processing.

        Returns:
            Dictionary containing analysis results or error information
        """
        try:
            self.logger.info(
                "Starting Bedrock analysis for figure",
                figure_id=figure.figure_id,
                project_id=project_id,
                upload_id=upload_id,
                image_size_bytes=len(image_data),
            )

            # Prepare context for analysis
            context = {"figure_id": figure.figure_id, "caption": figure.caption, "page_number": figure.page_number}

            # Determine image format (enum values are strings, so they work directly)
            image_format = figure.image_format or IngestionMimeType.PNG

            # Perform Bedrock analysis
            analysis_result = self.image_analysis_service.analyze_image(
                image_data=image_data, image_format=image_format, context=context
            )

            return analysis_result

        except Exception as e:
            self.logger.error(
                "Bedrock analysis failed for figure",
                figure_id=figure.figure_id,
                project_id=project_id,
                upload_id=upload_id,
                error=str(e),
            )

            # Return error result
            return {
                "model_used": "anthropic.claude-3-sonnet-20240229-v1:0",
                "analysis_summary": "Analysis failed",
                "analysis_error": str(e),
                "analysis_timestamp": figure.extracted_at.isoformat(),
            }

    def _store_table_metadata(
        self,
        table: ExtractedTable,
        csv_data: Optional[str],
        project_id: str,
        upload_id: str,
        page_number: int,
    ) -> None:
        """Create and store table metadata to S3."""
        try:
            # Convert dict bbox to Pydantic model
            bbox_model = None
            if table.bbox:
                bbox_model = BoundingBox(
                    left=table.bbox["left"],
                    top=table.bbox["top"],
                    right=table.bbox["right"],
                    bottom=table.bbox["bottom"],
                    coord_origin=table.bbox.get("coord_origin", "TOP_LEFT"),
                )

            # Create detailed metadata
            table_metadata = TableMetadata(
                table_id=table.table_id,
                source_document=f"s3://{settings.S3_BUCKET_NAME}/{project_id}/{upload_id}",
                page_number=table.page_number,
                extraction_timestamp=table.extracted_at,
                caption=table.caption,
                num_rows=table.num_rows,
                num_cols=table.num_cols,
                headers=table.headers,
                cell_count=table.cell_count,
                has_merged_cells=table.has_merged_cells,
                csv_available=bool(csv_data),
                bbox=bbox_model,
                docling_label=table.label,
                extraction_method=table.extraction_method,
            )

            # Store metadata JSON
            metadata_bucket, metadata_key, _ = self._generate_asset_s3_path(
                project_id=project_id,
                upload_id=upload_id,
                page_number=page_number,
                filename=f"{table.table_id}_metadata.json",
                asset_type="tables",
            )

            metadata_content = table_metadata.model_dump_json(indent=2).encode("utf-8")

            s3_metadata_path = self.storage_service.upload_content(metadata_content, metadata_bucket, metadata_key)

            table.s3_metadata_path = s3_metadata_path
            self.logger.debug(
                "Stored table metadata",
                table_id=table.table_id,
                s3_path=s3_metadata_path,
            )

        except Exception as e:
            self.logger.error(
                "Failed to store metadata for table",
                table_id=table.table_id,
                error=str(e),
            )

    def _generate_asset_s3_path(
        self,
        project_id: str,
        upload_id: str,
        page_number: int,
        filename: str,
        asset_type: str,
    ) -> Tuple[str, str, str]:
        """
        Generate S3 bucket and key for asset storage with organized path structure.

        Args:
            project_id: Project identifier
            upload_id: Upload ID (primary key across all services)
            page_number: Page number for multi-page documents
            filename: Name of the file to store
            asset_type: Asset type ("figures" or "tables")

        Returns:
            Tuple of (bucket, key, base_path)
        """
        bucket = settings.S3_BUCKET_NAME
        # For assets: project_id/upload_id/page_number/asset_type/filename
        key = f"{project_id}/{upload_id}/{page_number}/{asset_type}/{filename}"
        base_path = f"s3://{bucket}/{project_id}/{upload_id}"
        return bucket, key, base_path

    def _get_image_size(self, image_data: bytes) -> Optional[dict[str, Any]]:
        """Get image dimensions from image data."""
        try:
            import io

            from PIL import Image

            img = Image.open(io.BytesIO(image_data))
            return {"width": img.width, "height": img.height}
        except Exception as e:
            self.logger.warning(
                "Could not get image size",
                error=str(e),
                error_type=type(e).__name__,
            )
            return None

    def health_check(self) -> dict[str, Any]:
        """
        Check health of asset storage service.

        Returns:
            Dictionary with health status
        """
        health = {"asset_storage_service": "healthy", "storage_backend": "s3"}

        # Check S3 service health
        try:
            if hasattr(self.storage_service, "health_check"):
                s3_health = self.storage_service.health_check()
                health["s3_service"] = "healthy" if s3_health else "unhealthy"
            else:
                health["s3_service"] = "available"
        except Exception:
            health["s3_service"] = "unhealthy"

        # Document processor health check removed - asset extraction is now self-contained

        # Check image analysis service health
        try:
            analysis_health = self.image_analysis_service.health_check()
            health["image_analysis_service"] = (
                "healthy" if analysis_health.get("bedrock_image_analysis_service") == "healthy" else "unhealthy"
            )
        except Exception:
            health["image_analysis_service"] = "unhealthy"

        return health
