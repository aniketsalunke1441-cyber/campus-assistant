"""
grader.py — Task grader functions for CampusAssistantEnv

Each function receives the episode's final state dict and returns a
normalized score in (0.0, 1.0) — exclusive on both ends, as required
by the OpenEnv Task Validation spec.
"""

from typing import Any, Dict

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _score_from_state(state: Dict[str, Any], required_steps: list) -> float:
    """
    Compute a normalized score in (0.05, 0.95) from the final episode state.
    
    - Partial credit per completed step
    - Bonus for full completion (all steps done + finish_task called)
    """
    completed = state.get("completed_steps", [])
    done = state.get("done", False)
    current_reward = state.get("current_reward", 0.0)
    
    # Fraction of required steps completed
    if required_steps:
        fraction = len([s for s in completed if s in required_steps]) / len(required_steps)
    else:
        fraction = 0.0
    
    # Combine: 0.7 weight on internal reward, 0.3 on step fraction
    score = 0.7 * current_reward + 0.3 * fraction
    
    # Add completion bonus if all done
    if done and all(s in completed for s in required_steps):
        score = min(score + 0.1, 0.95)
    
    # Clamp strictly between 0.05 and 0.95 (never 0.0 or 1.0)
    return round(max(0.05, min(0.95, score)), 4)


# ---------------------------------------------------------------------------
# Task Graders
# ---------------------------------------------------------------------------

EASY_STEPS = ["search_notes", "summarize_notes", "finish_task"]
MEDIUM_STEPS = ["search_notes", "summarize_notes", "create_study_plan", "generate_reminder", "finish_task"]
HARD_STEPS = ["search_notes", "summarize_notes", "create_study_plan", "create_quiz", "generate_reminder", "create_checklist", "finish_task"]


def grade_easy(state: Dict[str, Any]) -> float:
    """
    Grader for the Easy task: 'Quick Review — Summarize DBMS Notes'.
    Required steps: search_notes → summarize_notes → finish_task
    Returns a score in (0.05, 0.95).
    """
    return _score_from_state(state, EASY_STEPS)


def grade_medium(state: Dict[str, Any]) -> float:
    """
    Grader for the Medium task: 'Exam Prep — Prepare for SQL Viva'.
    Required steps: search_notes → summarize_notes → create_study_plan → generate_reminder → finish_task
    Returns a score in (0.05, 0.95).
    """
    return _score_from_state(state, MEDIUM_STEPS)


def grade_hard(state: Dict[str, Any]) -> float:
    """
    Grader for the Hard task: 'Full Sprint — Complete 2-Hour Study Package'.
    Required steps: search_notes → summarize_notes → create_study_plan → create_quiz → generate_reminder → create_checklist → finish_task
    Returns a score in (0.05, 0.95).
    """
    return _score_from_state(state, HARD_STEPS)
