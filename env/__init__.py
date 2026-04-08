"""
CampusAssistantEnv - OpenEnv Compatible Environment
Autonomous AI Campus Assistant for college task automation.
"""

from .env_logic import CampusAssistantEnv
from .models import (
    CampusState,
    CampusAction,
    ActionType,
    DifficultyLevel,
    TaskDefinition,
    StepResult,
)

__all__ = [
    "CampusAssistantEnv",
    "CampusState",
    "CampusAction",
    "ActionType",
    "DifficultyLevel",
    "TaskDefinition",
    "StepResult",
]

__version__ = "1.0.0"
