"""
tasks.py — Fractional grader functions for CampusAssistantEnv

Each grader returns a score STRICTLY between 0 and 1:
  - Never 0.0
  - Never 1.0
  - Always in range (0.01, 0.99)

Required by OpenEnv Task Validation.
"""

from typing import Any, Dict


def grade_summarize_notes(state: Dict[str, Any]) -> float:
    """
    Task: summarize_notes (easy)
    Fractional rewards:
      +0.33 if search_notes used
      +0.33 if summarize_notes used
      +0.33 if finished
    Base: 0.01, Max: 0.99
    """
    reward = 0.01
    completed = state.get("completed_steps", [])
    finished = state.get("done", False)

    if "search_notes" in completed:
        reward += 0.33
    if "summarize_notes" in completed:
        reward += 0.33
    if finished:
        reward += 0.33

    return min(reward, 0.99)


def grade_prepare_viva(state: Dict[str, Any]) -> float:
    """
    Task: prepare_viva (medium)
    Fractional rewards:
      +0.25 if search_notes used
      +0.25 if create_study_plan used
      +0.25 if create_quiz used
      +0.24 if finished
    Base: 0.01, Max: 0.99
    """
    reward = 0.01
    completed = state.get("completed_steps", [])
    finished = state.get("done", False)

    if "search_notes" in completed:
        reward += 0.25
    if "create_study_plan" in completed:
        reward += 0.25
    if "create_quiz" in completed:
        reward += 0.25
    if finished:
        reward += 0.24

    return min(reward, 0.99)


def grade_complete_study_workflow(state: Dict[str, Any]) -> float:
    """
    Task: complete_study_workflow (hard)
    Fractional rewards:
      +0.20 if create_study_plan used
      +0.20 if create_quiz used
      +0.20 if create_checklist used
      +0.20 if generate_reminder used
      +0.19 if finished
    Base: 0.01, Max: 0.99
    """
    reward = 0.01
    completed = state.get("completed_steps", [])
    finished = state.get("done", False)

    if "create_study_plan" in completed:
        reward += 0.20
    if "create_quiz" in completed:
        reward += 0.20
    if "create_checklist" in completed:
        reward += 0.20
    if "generate_reminder" in completed:
        reward += 0.20
    if finished:
        reward += 0.19

    return min(reward, 0.99)
