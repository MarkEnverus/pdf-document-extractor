import os

from fastapi import APIRouter, status
from fastapi.responses import RedirectResponse

from lib.logger import Logger

logger = Logger.get_logger(os.path.basename(__file__))
router = APIRouter()


@router.get("/", status_code=status.HTTP_301_MOVED_PERMANENTLY)
async def redirect_root_to_docs() -> RedirectResponse:
    return RedirectResponse("/docs")
