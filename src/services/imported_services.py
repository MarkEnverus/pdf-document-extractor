from idp_file_management import get_file_upload_tracking_service
from idp_ingestion_status import get_ingestion_status_service

from src.core.postgres_connection_manager import (
    get_async_connection,
    get_sync_connection,
)

file_upload_tracking_service = get_file_upload_tracking_service(get_sync_connection(), get_async_connection())

ingestion_status_service = get_ingestion_status_service(
    get_sync_connection(), get_async_connection(), file_upload_tracking_service
)
