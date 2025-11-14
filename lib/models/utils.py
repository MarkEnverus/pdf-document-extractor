"""
Utility functions for pipeline events.

Provides helper functions for working with pipeline events, including UUID conversion.
"""

from uuid import UUID, uuid4


def safe_uuid_conversion(id_string: str) -> UUID:
    """
    Safely convert a string to UUID, handling various formats.

    This function attempts to parse a string as a UUID. If the string is empty
    or not a valid UUID format, it generates a new random UUID as a fallback.

    Args:
        id_string: String that may or may not be a valid UUID

    Returns:
        UUID object, either parsed from string or a newly generated UUID

    Examples:
        >>> safe_uuid_conversion("a2cc5ad8-a583-47f8-be1e-1e6273b06346")
        UUID('a2cc5ad8-a583-47f8-be1e-1e6273b06346')

        >>> safe_uuid_conversion("invalid")  # Returns new UUID
        UUID('...')  # New random UUID

        >>> safe_uuid_conversion("")  # Returns new UUID
        UUID('...')  # New random UUID
    """
    if not id_string:
        # Empty string - generate new UUID
        return uuid4()

    try:
        # Try direct UUID conversion
        return UUID(id_string)
    except ValueError:
        # If not a valid UUID, generate a new one
        return uuid4()
