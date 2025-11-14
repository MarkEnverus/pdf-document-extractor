import os
import uuid
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException, BackgroundTasks

from src.dependencies.providers import (
    ProcessingOrchestratorDep,
    LoggerDep,
)
from src.exceptions.domain_exceptions import ProcessingError
from lib.logger import Logger
from src.models.docling_models import DoclingConfig, DoclingProcessRequest
from src.models.processing_models import DocumentRequest
from src.services.processing_orchestrator import ProcessingOrchestrator

logger = Logger.get_logger(os.path.basename(__file__))

document_router = APIRouter(prefix="/documents", tags=["Document Processing"])


async def process_docling_pipeline(
    request_id: str,
    project_id: str,
    upload_id: str,
    request: DoclingProcessRequest,
    orchestrator: ProcessingOrchestrator,
) -> None:
    """Background task to process documents through Docling pipeline"""
    try:
        if not request.document_urls:
            raise ValueError("No document URLs provided")

        # Process documents with Docling through ProcessingOrchestrator
        result = await orchestrator.process_documents(
            strategy="docling",
            request_id=request_id,
            project_id=project_id,
            upload_id=upload_id,
            document_urls=request.document_urls,
            source="api",
        )

        # Service layer handles all status updates via centralized StatusTracker
        logger.info(
            f"Docling pipeline completed for request {request_id} with status: {'completed' if result.get('success', False) else 'failed'}"
        )

    except Exception as e:
        logger.error(f"Docling processing pipeline failed for request {request_id}: {str(e)}")
        # Exception handling in service layer already updates status


async def process_document_pipeline(
    request_id: str, document_urls: List[str], metadata: Optional[Dict[str, Any]], orchestrator: ProcessingOrchestrator
) -> None:
    """Background task to process the documents through the entire pipeline"""
    try:
        # Extract required parameters from metadata or use defaults
        strategy = metadata.get("processing_strategy", "docling") if metadata else "docling"
        project_id = metadata.get("project_id", "default") if metadata else "default"
        upload_id = metadata.get("upload_id", request_id) if metadata else request_id

        result = await orchestrator.process_documents(
            strategy=strategy,
            request_id=request_id,
            project_id=project_id,
            upload_id=upload_id,
            document_urls=document_urls,
            source="api",
        )
        # Service layer handles all status updates via centralized StatusTracker
        logger.info(f"Pipeline completed for request {request_id} with success: {result.get('success', False)}")

    except ProcessingError as e:
        logger.error(f"Processing error in pipeline {request_id}: {e.message}")
        # Exception handling in service layer already updates status
    except Exception as e:
        logger.error(f"Unexpected error in pipeline {request_id}: {str(e)}")
        # Exception handling in service layer already updates status


@document_router.post("/process", response_model=Dict[str, str])
async def process_document(
    request: DocumentRequest,
    background_tasks: BackgroundTasks,
    orchestrator: ProcessingOrchestratorDep,
    logger_instance: LoggerDep,
) -> Dict[str, str]:
    """
    Start document processing pipeline for multiple documents.

    This endpoint accepts a JSON payload with document URLs and metadata,
    then processes them through the entire pipeline:
    1. Call genai-mcp-document-extraction API with all URLs
    2. Store extraction result in S3 /EXTRACTION bucket
    3. Publish events to Kafka
    """
    request_id = str(uuid.uuid4())

    try:
        # Start background processing with dependency injection
        background_tasks.add_task(
            process_document_pipeline,
            request_id,
            request.document_urls,
            request.metadata,
            orchestrator,  # Inject orchestrator into background task
        )

        logger_instance.info(f"Document processing queued for request {request_id}")

        return {"request_id": request_id, "status": "queued", "message": "Document processing started"}

    except Exception as e:
        logger_instance.error(f"Failed to queue document processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start document processing: {str(e)}")


# Status endpoints removed - status tracking now handled by external API
# Individual status checks should be made directly to the status API endpoint


# Health endpoint removed - health checks now handled by ProcessingOrchestrator.health_check()
# This eliminates FastAPI Depends injection issues and centralizes health checking logic
# To get health status, use the ProcessingOrchestrator's health_check method instead


@document_router.post("/process-docling", response_model=Dict[str, str])
async def process_documents_docling(
    request: DoclingProcessRequest,
    background_tasks: BackgroundTasks,
    orchestrator: ProcessingOrchestratorDep,
    logger_instance: LoggerDep,
) -> Dict[str, str]:
    """
    Start Docling document processing pipeline.

    This endpoint provides an alternative processing path using Docling for:
    - Local document processing with OCR, table detection, and figure extraction
    - AWS Bedrock/Claude-based document processing for enhanced extraction

    Returns immediately with a request ID that can be used to check status via /status endpoint.
    """
    request_id = str(uuid.uuid4())

    try:
        # Generate project_id and upload_id for API requests
        project_id = request.metadata.get("project_id", "default") if request.metadata else "default"
        upload_id = str(uuid.uuid4())  # Generate new upload_id for API requests

        # Start background processing with dependency injection
        background_tasks.add_task(
            process_docling_pipeline,
            request_id,
            project_id,
            upload_id,
            request,
            orchestrator,  # Inject orchestrator into background task
        )

        logger_instance.info(f"Docling processing queued for request {request_id}")

        return {"request_id": request_id, "status": "queued", "message": "Docling processing started"}

    except Exception as e:
        logger_instance.error(f"Failed to queue Docling processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start Docling processing: {str(e)}")
