from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# ── Actions ──────────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    SEARCH_NOTES = "search_notes"
    SUMMARIZE_NOTES = "summarize_notes"
    CREATE_STUDY_PLAN = "create_study_plan"
    CREATE_QUIZ = "create_quiz"
    GENERATE_REMINDER = "generate_reminder"
    CREATE_CHECKLIST = "create_checklist"
    FINISH_TASK = "finish_task"

class CampusAction(BaseModel):
    action: ActionType
    parameters: Dict[str, Any] = Field(default_factory=dict)

# ── State ────────────────────────────────────────────────────────────────────

class CampusState(BaseModel):
    goal: str
    tasks_remaining: int
    completed_steps: int
    difficulty: str
    
    # Extra fields for logic tracking (can be hidden in final response if needed)
    last_message: str = ""
    current_reward: float = 0.0
    done: bool = False
    
    # Internal tracking
    _required_steps: List[ActionType] = []
    _completed_list: List[ActionType] = []

# ── API Responses ────────────────────────────────────────────────────────────

class ResetResponse(BaseModel):
    state: CampusState
    done: bool = False

class StepResponse(BaseModel):
    state: CampusState
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
