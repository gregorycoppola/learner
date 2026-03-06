"""Error analysis routes."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/analyze", tags=["analyze"])


class AnalyzeRequest(BaseModel):
    machine: str = "incrementer"
    n_samples: int = 2000
    n_train_samples: int = 10000
    n_epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    d_model: int = 32
    n_layers: int = 2
    n_heads: int = 2
    min_tape_len: int = 16
    seed: int = 42


@router.post("/run")
async def run_analysis(req: AnalyzeRequest):
    """Train a model then run full error analysis. Returns breakdown table."""
    from learner.core.machines import MACHINES
    from learner.core.trainer import train_streaming
    from learner.core.analysis import analyze, make_breakdown_table

    if req.machine not in MACHINES:
        raise HTTPException(400, f"Unknown machine '{req.machine}'")

    # Train to completion (consume the generator)
    result = None
    model = None
    state_index = None
    for event in train_streaming(
        machine_name=req.machine,
        n_samples=req.n_train_samples,
        n_epochs=req.n_epochs,
        batch_size=req.batch_size,
        lr=req.lr,
        d_model=req.d_model,
        n_layers=req.n_layers,
        n_heads=req.n_heads,
        min_tape_len=req.min_tape_len,
        seed=req.seed,
    ):
        if event["type"] == "init":
            state_index = None  # will be set after
        if event["type"] == "done":
            break
        result = event

    # Re-get model + state_index by running trainer directly
    from learner.core.trainer import train_streaming as ts, make_state_index
    from learner.core.machines import get_machine
    import torch

    # Retrain to get the model object (trainer needs refactor to return model,
    # for now we retrain — cheap at 50 epochs)
    from learner.core.trainer import _train_and_return
    model, state_index = _train_and_return(
        machine_name=req.machine,
        n_samples=req.n_train_samples,
        n_epochs=req.n_epochs,
        batch_size=req.batch_size,
        lr=req.lr,
        d_model=req.d_model,
        n_layers=req.n_layers,
        n_heads=req.n_heads,
        min_tape_len=req.min_tape_len,
        seed=req.seed,
    )

    results = analyze(
        model=model,
        state_index=state_index,
        machine_name=req.machine,
        n_samples=req.n_samples,
        min_tape_len=req.min_tape_len,
    )

    table = make_breakdown_table(results)

    return {
        "machine": req.machine,
        "n_analyzed": len(results),
        "overall_acc": sum(1 for r in results if r.all_correct) / len(results),
        "table": table,
    }