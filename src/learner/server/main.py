"""Learner API server."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Learner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from learner.server.routes import (
    health, tm, train, analyze, grpo, sft_grpo, model, hybrid
)

app.include_router(health.router)
app.include_router(tm.router)
app.include_router(train.router)
app.include_router(analyze.router)
app.include_router(grpo.router)
app.include_router(sft_grpo.router)
app.include_router(model.router)
app.include_router(hybrid.router)


@app.on_event("startup")
async def startup():
    print("\n📡 Learner API Routes:")
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            methods = ", ".join(route.methods - {"HEAD", "OPTIONS"})
            if methods:
                print(f"  {methods:8} {route.path}")
    print()


@app.get("/")
async def root():
    return {"status": "ok", "service": "learner"}