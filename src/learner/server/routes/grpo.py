"""GRPO training routes."""
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api/grpo", tags=["grpo"])


class GRPORequest(BaseModel):
    machine: str = "incrementer"
    n_samples: int = 2000
    n_epochs: int = 200
    batch_size: int = 16
    lr: float = 1e-4
    d_model: int = 32
    n_layers: int = 2
    n_heads: int = 2
    min_tape_len: int = 16
    seed: int = 42
    K: int = 8
    kl_coef: float = 0.01
    analyze_every: int = 5
    analyze_samples: int = 500


@router.post("/run")
async def run_grpo(req: GRPORequest):
    """GRPO training with TM verifier. Streams epoch updates as SSE."""
    from learner.core.machines import MACHINES

    if req.machine not in MACHINES:
        raise HTTPException(400, f"Unknown machine '{req.machine}'")

    def generate():
        from learner.core.trainer_grpo import train_grpo_streaming
        for event in train_grpo_streaming(
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
            K=req.K,
            kl_coef=req.kl_coef,
            analyze_every=req.analyze_every,
            analyze_samples=req.analyze_samples,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")