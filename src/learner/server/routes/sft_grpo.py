"""SFT then GRPO training route."""
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/api/sft-grpo", tags=["sft-grpo"])


class SFTGRPORequest(BaseModel):
    machine: str = "incrementer"
    n_samples: int = 2000
    sft_max_epochs: int = 50
    sft_threshold: float = 0.90
    sft_lr: float = 1e-3
    sft_batch_size: int = 32
    grpo_epochs: int = 100000
    grpo_lr: float = 1e-4
    grpo_batch_size: int = 16
    K: int = 8
    kl_coef: float = 0.01
    d_model: int = 32
    n_layers: int = 2
    n_heads: int = 2
    min_tape_len: int = 16
    seed: int = 42
    analyze_every: int = 5
    analyze_samples: int = 500
    checkpoint_name: Optional[str] = None


@router.post("/run")
async def run_sft_grpo(req: SFTGRPORequest):
    """SFT warmup then GRPO. Streams events as SSE."""
    from learner.core.machines import MACHINES
    if req.machine not in MACHINES:
        raise HTTPException(400, f"Unknown machine '{req.machine}'")

    def generate():
        from learner.core.trainer_sft_grpo import train_sft_then_grpo_streaming
        for event in train_sft_then_grpo_streaming(
            machine_name=req.machine,
            n_samples=req.n_samples,
            sft_max_epochs=req.sft_max_epochs,
            sft_threshold=req.sft_threshold,
            sft_lr=req.sft_lr,
            sft_batch_size=req.sft_batch_size,
            grpo_epochs=req.grpo_epochs,
            grpo_lr=req.grpo_lr,
            grpo_batch_size=req.grpo_batch_size,
            K=req.K,
            kl_coef=req.kl_coef,
            d_model=req.d_model,
            n_layers=req.n_layers,
            n_heads=req.n_heads,
            min_tape_len=req.min_tape_len,
            seed=req.seed,
            analyze_every=req.analyze_every,
            analyze_samples=req.analyze_samples,
            checkpoint_name=req.checkpoint_name,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")