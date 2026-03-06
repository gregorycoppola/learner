"""Training routes."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/api/train", tags=["train"])


class TrainRequest(BaseModel):
    machine: str = "incrementer"
    n_samples: int = 2000
    n_epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    d_model: int = 32
    n_layers: int = 2
    n_heads: int = 2
    min_tape_len: int = 16
    seed: int = 42


@router.post("/run")
async def run_training(req: TrainRequest):
    """Train a transformer to predict one TM step. Runs synchronously."""
    from learner.core.trainer import train
    from learner.core.machines import MACHINES

    if req.machine not in MACHINES:
        raise HTTPException(400, f"Unknown machine '{req.machine}'. Available: {list(MACHINES.keys())}")

    results = train(
        machine_name=req.machine,
        n_samples=req.n_samples,
        n_epochs=req.n_epochs,
        batch_size=req.batch_size,
        lr=req.lr,
        d_model=req.d_model,
        n_layers=req.n_layers,
        n_heads=req.n_heads,
        min_tape_len=req.min_tape_len,
        seed=req.seed,
    )

    # Don't serialize the model itself
    return {
        "machine": results["machine"],
        "n_train": results["n_train"],
        "n_val": results["n_val"],
        "d_input": results["d_input"],
        "final_val_acc": results["final_val_acc"],
        "final_val_loss": results["final_val_loss"],
        "history": results["history"],
    }