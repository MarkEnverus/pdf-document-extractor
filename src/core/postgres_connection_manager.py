from functools import cache

from py_postgres.aio import AsyncPostgresClient
from py_postgres.sync import PostgresClient

from src.configs.settings import settings


@cache
def get_sync_connection() -> PostgresClient:
    return PostgresClient(settings.get_postgres_config())


@cache
def get_async_connection() -> AsyncPostgresClient:
    return AsyncPostgresClient(settings.get_postgres_config())
