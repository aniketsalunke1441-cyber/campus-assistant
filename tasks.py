"""
tasks.py — Binary grader functions for CampusAssistantEnv

Each grader receives the final episode state dict and returns ONLY:
  1.0  — task fully completed
  0.0  — task not completed

Required by the OpenEnv Task Validation spec (no partial rewards).
"""

from typing import Any, Dict


# ── Task 1: Easy ──────────────────────────────────────────────────────────────

def grade_summarize_notes(state: Dict[str, Any]) -> float:
    """
    Task: summarize_notes (easy)
    Goal: Summarize campus notes.
    Success requires:
      - search_notes used
      - summarize_notes used
      - finished is True
    Returns 1.0 if all conditions met, else 0.0.
    """
    completed = state.get("completed_steps", [])
    finished = state.get("done", False)

    if (
        "search_notes" in completed
        and "summarize_notes" in completed
        and finished
    ):
        return 1.0
    return 0.0


# ── Task 2: Medium ────────────────────────────────────────────────────────────

def grade_prepare_viva(state: Dict[str, Any]) -> float:
    """
    Task: prepare_viva (medium)
    Goal: Prepare for viva exam.
    Success requires:
      - search_notes used
      - create_study_plan used
      - create_quiz used
      - finished is True
    Returns 1.0 if all conditions met, else 0.0.
    """
    completed = state.get("completed_steps", [])
    finished = state.get("done", False)

    if (
        "search_notes" in completed
        and "create_study_plan" in completed
        and "create_quiz" in completed
        and finished
    ):
        return 1.0
    return 0.0


# ── Task 3: Hard ──────────────────────────────────────────────────────────────

def grade_complete_study_workflow(state: Dict[str, Any]) -> float:
    """
    Task: complete_study_workflow (hard)
    Goal: Complete full study workflow.
    Success requires:
      - create_study_plan used
      - create_quiz used
      - create_checklist used
      - generate_reminder used
      - finished is True
    Returns 1.0 if all conditions met, else 0.0.
    """
    completed = state.get("completed_steps", [])
    finished = state.get("done", False)

    if (
        "create_study_plan" in completed
        and "create_quiz" in completed
        and "create_checklist" in completed
        and "generate_reminder" in completed
        and finished
    ):
        return 1.0
    return 0.0
