"""Model checkpoint management routes."""
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/model", tags=["model"])


@router.get("/list")
async def list_models():
    from learner.core.checkpoint import list_checkpoints
    return {"checkpoints": list_checkpoints()}


@router.delete("/{name}")
async def delete_model(name: str):
    from learner.core.checkpoint import models_dir
    path = models_dir() / f"{name}.pt"
    if not path.exists():
        raise HTTPException(404, f"No checkpoint: {name}")
    path.unlink()
    return {"deleted": name}