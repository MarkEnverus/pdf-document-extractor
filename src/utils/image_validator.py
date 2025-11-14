"""Image validation utilities for pre-processing checks."""

from pathlib import Path
from typing import Optional, Tuple
from PIL import Image

from lib.logger import Logger

logger = Logger.get_logger(__name__)

# Conservative limits based on PIL's decompression bomb protection
# PIL default limit is ~178MP, we'll be more conservative
MAX_IMAGE_PIXELS = 100_000_000  # 100 megapixels
MAX_DIMENSION = 8192  # 8192x8192 maximum (common GPU texture limit)


class ImageValidationError(Exception):
    """Raised when image validation fails."""

    def __init__(self, message: str, should_commit: bool = True):
        """
        Initialize validation error.

        Args:
            message: Error message
            should_commit: Whether this is a hard failure (should commit offset)
        """
        super().__init__(message)
        self.should_commit = should_commit


class ImageValidator:
    """Validator for image files before processing."""

    @staticmethod
    def validate_image_size(
        file_path: str,
        max_pixels: int = MAX_IMAGE_PIXELS,
        max_dimension: int = MAX_DIMENSION,
        upload_id: str | None = None,
        project_id: str | None = None,
        request_id: str | None = None,
    ) -> Tuple[int, int]:
        """
        Validate image dimensions before processing.

        Args:
            file_path: Path to image file
            max_pixels: Maximum total pixels allowed
            max_dimension: Maximum width or height allowed
            upload_id: Upload ID for tracking (optional)
            project_id: Project ID for tracking (optional)
            request_id: Request ID for tracking (optional)

        Returns:
            Tuple of (width, height) if validation passes

        Raises:
            ImageValidationError: If image exceeds size limits (hard failure - should commit)
        """
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                total_pixels = width * height

                logger.info(
                    "Image dimensions detected",
                    file_path=Path(file_path).name,
                    width=width,
                    height=height,
                    total_pixels=total_pixels,
                    format=img.format,
                    upload_id=upload_id,
                    project_id=project_id,
                    request_id=request_id,
                )

                # Check individual dimensions
                if width > max_dimension or height > max_dimension:
                    error_msg = (
                        f"Image dimensions ({width}x{height}) exceed maximum allowed dimension "
                        f"({max_dimension}px). This image is too large to process safely. "
                        f"Please resize the image to a maximum of {max_dimension}x{max_dimension} pixels."
                    )
                    logger.error(
                        "Image dimension validation failed",
                        file_path=Path(file_path).name,
                        width=width,
                        height=height,
                        max_dimension=max_dimension,
                        validation_error=error_msg,
                        upload_id=upload_id,
                        project_id=project_id,
                        request_id=request_id,
                    )
                    raise ImageValidationError(error_msg, should_commit=True)

                # Check total pixels
                if total_pixels > max_pixels:
                    megapixels = total_pixels / 1_000_000
                    max_megapixels = max_pixels / 1_000_000
                    error_msg = (
                        f"Image size ({width}x{height} = {megapixels:.1f}MP) exceeds maximum "
                        f"allowed size ({max_megapixels:.1f}MP). This image is too large to process safely. "
                        f"Please reduce the image resolution."
                    )
                    logger.error(
                        "Image pixel count validation failed",
                        file_path=Path(file_path).name,
                        width=width,
                        height=height,
                        total_pixels=total_pixels,
                        max_pixels=max_pixels,
                        validation_error=error_msg,
                        upload_id=upload_id,
                        project_id=project_id,
                        request_id=request_id,
                    )
                    raise ImageValidationError(error_msg, should_commit=True)

                logger.info(
                    "Image validation passed",
                    file_path=Path(file_path).name,
                    width=width,
                    height=height,
                    total_pixels=total_pixels,
                    upload_id=upload_id,
                    project_id=project_id,
                    request_id=request_id,
                )

                return width, height

        except ImageValidationError:
            # Re-raise our validation errors
            raise
        except Exception as e:
            # Any other error opening the image is a hard failure
            error_msg = f"Failed to read image file: {str(e)}. File may be corrupted or not a valid image format."
            logger.error(
                "Image file reading failed",
                file_path=Path(file_path).name,
                error=str(e),
                error_type=type(e).__name__,
                validation_error=error_msg,
                upload_id=upload_id,
                project_id=project_id,
                request_id=request_id,
            )
            raise ImageValidationError(error_msg, should_commit=True) from e

    @staticmethod
    def should_validate_as_image(file_path: str) -> bool:
        """
        Check if file should be validated as an image.

        Args:
            file_path: Path to file

        Returns:
            True if file is an image type that needs validation
        """
        image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp"}
        suffix = Path(file_path).suffix.lower()
        return suffix in image_extensions
