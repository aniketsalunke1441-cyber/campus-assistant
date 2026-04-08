import os
import sys
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import uvicorn

# Ensure the root directory is on the path to import env.env_logic correctly
# From /app/server/app.py, we need sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env.env_logic import CampusAssistantEnv
from env.models import CampusAction, ResetResponse, StepResponse, CampusState

# ── App Init ──────────────────────────────────────────────────────────────

app = FastAPI(title="Campus AI OpenEnv Space", version="1.0.0")

# Singleton environment per session
env = CampusAssistantEnv(seed=42)

# ── API Models ─────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    difficulty: str = "easy"

class StepRequest(BaseModel):
    action: str
    parameters: Dict[str, Any] = {}

# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "environment": "CampusAssistantEnv"}

@app.get("/state")
def get_state():
    current_state = env.state()
    if current_state is None:
        raise HTTPException(status_code=400, detail="Environment not reset. Call /reset first.")
    return current_state.dict()

@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest = Body(default=ResetRequest())):
    new_state = env.reset(difficulty=request.difficulty)
    return ResetResponse(state=new_state, done=False)

@app.post("/step", response_model=StepResponse)
def step(request: StepRequest = Body(...)):
    try:
        current_state = env.state()
        if current_state is None:
            raise HTTPException(status_code=400, detail="Environment not reset. Call /reset first.")
        result_state, reward, done, info = env.step(request.dict())
        return StepResponse(state=result_state, reward=reward, done=done, info=info)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ── Mandatory Entry Point ───────────────────────────────────────────────────

def main():
    """Entry point for the OpenEnv multi-mode validator."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()
