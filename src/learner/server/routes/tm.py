"""Turing machine routes."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from learner.core.machines import get_machine, MACHINES
from learner.core.data import generate_pairs, int_to_tape, tape_to_int

router = APIRouter(prefix="/api/tm", tags=["tm"])


class RunRequest(BaseModel):
    machine: str = "incrementer"
    tape: Optional[list[str]] = None
    input_int: Optional[int] = None
    max_steps: int = 1000


class GenerateRequest(BaseModel):
    machine: str = "incrementer"
    n_samples: int = 100
    min_val: int = 0
    max_val: int = 255
    seed: int = 42


@router.get("/machines")
async def list_machines():
    return {"machines": list(MACHINES.keys())}


@router.post("/run")
async def run_tm(req: RunRequest):
    """Run a TM on a tape and return the full execution trace."""
    try:
        tm = get_machine(req.machine)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if req.tape is not None:
        tape = req.tape
    elif req.input_int is not None:
        tape = int_to_tape(req.input_int)
    else:
        raise HTTPException(400, "Provide either 'tape' or 'input_int'")

    trace = tm.run(tape, max_steps=req.max_steps)
    final = trace[-1]

    return {
        "machine": req.machine,
        "input_tape": tape,
        "input_int": tape_to_int(tape),
        "output_tape": final["tape"],
        "output_int": tape_to_int(final["tape"]),
        "steps": len(trace) - 1,
        "halted": final["halted"],
        "trace": trace,
    }


@router.post("/generate")
async def generate_data(req: GenerateRequest):
    """Generate synthetic step pairs for training."""
    try:
        get_machine(req.machine)
    except ValueError as e:
        raise HTTPException(400, str(e))

    pairs = generate_pairs(
        machine_name=req.machine,
        n_samples=req.n_samples,
        min_val=req.min_val,
        max_val=req.max_val,
        seed=req.seed,
    )

    return {
        "machine": req.machine,
        "n_samples": len(pairs),
        "pairs": pairs,
    }