"""Health check route."""
from fastapi import APIRouter

router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("")
async def health():
    """Basic health check."""
    return {
        "status": "ok",
        "service": "learner",
        "version": "0.1.0",
    }