"""
Lead-Ops · FastAPI Application
==============================
Entrypoint for the OpenEnv environment server.

Provides the OpenEnv REST API:
    - GET  /health               — container health check (200 OK)
    - GET  /state/{session_id}   — database state snapshot
    - POST /reset                — initialize a new episode
    - POST /step                 — execute an action

Port: 7860 (Hugging Face Spaces requirement)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

from session_manager import SessionManager
from environment import LeadOpsEnv
from models import TaskID, Action, LeadObservation, StepResult

app = FastAPI(
    title="Lead-Ops OpenEnv",
    description=(
        "RL environment for autonomous sales lead enrichment, "
        "MEDDIC qualification, and strategic routing."
    ),
    version="1.0.0",
)

# Global instances
sm = SessionManager()
env = LeadOpsEnv(sm)


# ── API Models ──────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: TaskID


class ResetResponse(BaseModel):
    session_id: str
    task_id: str
    observation: LeadObservation


class StepRequest(BaseModel):
    session_id: str
    action: Action


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Basic health check endpoint for HF Spaces monitoring."""
    from config import get_settings
    settings = get_settings()
    return {
        "status": "healthy",
        "version": "1.0.0",
        "active_sessions": sm.active_count,
        "configured": settings.is_configured,
    }


@app.post("/reset", response_model=ResetResponse)
async def reset(req: ResetRequest = Body(...)):
    """Initialize a new RL episode."""
    try:
        session_id, obs = env.reset(req.task_id)
        return {
            "session_id": session_id,
            "task_id": req.task_id.value,
            "observation": obs,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResult)
async def step(req: StepRequest = Body(...)):
    """Execute an action and advance the environment."""
    try:
        result = env.step(req.session_id, req.action)
        return result
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state/{session_id}")
async def state(session_id: str):
    """Retrieve full database state for debugging."""
    try:
        return env.state(session_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Direct run (for local testing) ──────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
