"""Server routes for hybrid loss training."""
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

from learner.core.trainer_hybrid import train_hybrid_streaming

router = APIRouter(prefix="/api/hybrid")


class HybridRunRequest(BaseModel):
    machine:         str   = "incrementer"
    n_samples:       int   = 2000
    n_epochs:        int   = 100000
    batch_size:      int   = 32
    lr:              float = 1e-3
    d_model:         int   = 32
    n_layers:        int   = 2
    n_heads:         int   = 2
    min_tape_len:    int   = 16
    seed:            int   = 42
    carry_weight:    float = 10.0
    tape_weight:     float = 1.0
    head_weight:     float = 1.0
    state_weight:    float = 1.0
    analyze_every:   int   = 5
    analyze_samples: int   = 1000


@router.post("/run")
def run(req: HybridRunRequest):
    def event_stream():
        for event in train_hybrid_streaming(
            machine_name=req.machine,
            n_samples=req.n_samples,
            n_epochs=req.n_epochs,
            batch_size=req.batch_size,
            lr=req.lr,
            d_model=req.d_model,
            n_layers=req.n_layers,
            min_tape_len=req.min_tape_len,
            seed=req.seed,
            carry_weight=req.carry_weight,
            tape_weight=req.tape_weight,
            head_weight=req.head_weight,
            state_weight=req.state_weight,
            analyze_every=req.analyze_every,
            analyze_samples=req.analyze_samples,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")