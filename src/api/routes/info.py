import os
from typing import Dict, Any

from fastapi import APIRouter, status, Response

from lib.logger import Logger
from src.dependencies.providers import ProcessingOrchestratorDep

# Create our own info router for health check endpoints
info_router = APIRouter()

# Re-export for type checking
__all__ = ["info_router"]

logger = Logger.get_logger(os.path.basename(__file__))


@info_router.get("/liveness", tags=["Healthcheck"], status_code=status.HTTP_200_OK)
async def liveness_check() -> Dict[str, str]:
    """Simple liveness check - returns OK if service is running."""
    return {"status": "ok"}


@info_router.get("/readiness", tags=["Healthcheck"], status_code=status.HTTP_200_OK)
async def readiness_check(response: Response, processing_orchestrator: ProcessingOrchestratorDep) -> Dict[str, Any]:
    """Deep health check that verifies all dependent services"""
    try:
        # Perform comprehensive health check through processing orchestrator
        health_result = processing_orchestrator.health_check()

        # Determine HTTP status code based on overall health
        overall_health = health_result.get("overall_health", "unknown")
        if overall_health == "healthy":
            status_code = status.HTTP_200_OK
            service_status = "ready"
        else:
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            service_status = "not ready"

        response.status_code = status_code

        # Extract services and unhealthy services information
        services: Dict[str, Any] = {}
        unhealthy_services: list[str] = []

        # Flatten nested health information for API response
        for service_name, service_health in health_result.items():
            if isinstance(service_health, dict):
                services[service_name] = service_health
                # Check for unhealthy services
                health_status = service_health.get("overall_health")
                if health_status in ["degraded", "unhealthy"]:
                    unhealthy_services.append(service_name)
            elif isinstance(service_health, str) and service_health in ["degraded", "unhealthy"]:
                services[service_name] = service_health
                unhealthy_services.append(service_name)

        # Log the health check results
        logger.info(f"Health check completed: {service_status} (HTTP {status_code})")
        if overall_health != "healthy":
            logger.warning(f"Unhealthy services detected: {unhealthy_services}")

        return {
            "status": service_status,
            "overall_health": overall_health,
            "timestamp": health_result.get("timestamp", "not_provided"),
            "services": services,
            "unhealthy_services": unhealthy_services,
        }

    except Exception as e:
        # If health check itself fails, return service unavailable
        logger.error(f"Health check failed with exception: {str(e)}")
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

        return {
            "status": "not ready",
            "overall_health": "unhealthy",
            "error": f"Health check failed: {str(e)}",
            "services": {},
            "unhealthy_services": ["health_check_system"],
        }
