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
    user_goal: str
    task_name: str
    tasks_remaining: List[str]
    completed_steps: List[str]
    tools_used: List[str]
    difficulty_level: str
    time_remaining: int
    current_reward: float = 0.0
    done: bool = False
    last_message: str = ""
    generated_content: Dict[str, str] = Field(default_factory=dict)
    
    # Internal tracking
    _required_steps: List[ActionType] = []
    _completed_list: List[ActionType] = []

# ── Tasks ───────────────────────────────────────────────────────────────────

class TaskInfo(BaseModel):
    id: str
    name: str
    goal: str
    description: str
    difficulty: str
    required_steps: List[ActionType]

# ── API Responses ────────────────────────────────────────────────────────────

class ResetResponse(BaseModel):
    state: CampusState
    done: bool = False

class StepResponse(BaseModel):
    state: CampusState
    reward: float
    done: bool

    def to_tuple(self):
        """Compatibility with Phase 1 code expecting a tuple."""
        return (self.state, self.reward, self.done)
