from fastapi import APIRouter

from src.api.routes import info, swagger, document_processing

api_router = APIRouter()
api_router.include_router(info.info_router)
api_router.include_router(document_processing.document_router)

private_router = APIRouter()
private_router.include_router(swagger.router)
