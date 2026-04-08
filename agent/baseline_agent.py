"""
baseline_agent.py - LLM-Based Baseline Agent for CampusAssistantEnv

This agent:
  1. Reads the current environment state via state()
  2. Uses an LLM (OpenAI GPT or open-source via litellm/fallback) to decide
     the next action based on the current observation
  3. Calls env.step(action)
  4. Loops until done=True
  5. Returns final score and episode transcript

The agent automatically falls back to a rule-based policy if no OPENAI_API_KEY
is found, making it fully runnable without any API key.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional LLM Imports
# ---------------------------------------------------------------------------

try:
    import openai

    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------

from env.env_logic import CampusAssistantEnv
from env.models import ActionType, CampusAction, CampusState


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert AI campus assistant agent operating inside the CampusAssistantEnv.

Your goal is to help a student complete their task by choosing the best action at each step.

Available actions (use EXACTLY these strings):
- search_notes         → Search the notes knowledge base
- summarize_notes      → Summarize the retrieved notes
- create_study_plan    → Generate a structured study plan
- generate_reminder    → Set an exam or task reminder
- create_quiz          → Generate a practice quiz
- create_checklist     → Create a preparation checklist
- finish_task          → Signal that the task is complete

Rules:
1. Always search_notes FIRST before summarizing.
2. Follow a logical sequence: search → summarize → plan/quiz/checklist → reminder → finish.
3. Call finish_task ONLY after all required steps are completed.
4. Return ONLY a JSON object like: {"action": "search_notes", "parameters": {"topic": "DBMS"}}
5. Do NOT add any explanation — ONLY the raw JSON.
"""

USER_PROMPT_TEMPLATE = """Current Environment State:
{state_json}

Task remaining steps: {remaining}
Steps completed: {completed}

What is the single best action to take RIGHT NOW? Return JSON only."""


# ---------------------------------------------------------------------------
# Fallback Rule-Based Policy
# ---------------------------------------------------------------------------

def _rule_based_policy(state_dict: Dict[str, Any]) -> CampusAction:
    """
    Deterministic rule-based policy used when no LLM is available.
    Follows the optimal sequence for each task.
    """
    remaining: List[str] = state_dict.get("tasks_remaining", [])
    completed: List[str] = state_dict.get("completed_steps", [])

    # Ordered preference
    preference = [
        ActionType.SEARCH_NOTES,
        ActionType.SUMMARIZE_NOTES,
        ActionType.CREATE_STUDY_PLAN,
        ActionType.CREATE_QUIZ,
        ActionType.GENERATE_REMINDER,
        ActionType.CREATE_CHECKLIST,
        ActionType.FINISH_TASK,
    ]

    topic_map = {
        "DBMS": "dbms",
        "SQL": "sql",
        "SQL Viva": "sql",
        "Full 2-Hour": "dbms",
    }
    task_name = state_dict.get("task_name", "")
    topic = next(
        (v for k, v in topic_map.items() if k.lower() in task_name.lower()), "dbms"
    )

    for action in preference:
        if action.value in remaining:
            params = {}
            if action == ActionType.SEARCH_NOTES:
                params = {"topic": topic}
            elif action == ActionType.SUMMARIZE_NOTES:
                params = {"topic": topic.upper()}
            elif action == ActionType.CREATE_STUDY_PLAN:
                params = {"subject": "DBMS & SQL", "hours": 2}
            elif action == ActionType.GENERATE_REMINDER:
                params = {"event": "SQL Viva", "time": "8:00 AM"}
            elif action == ActionType.CREATE_QUIZ:
                params = {"topic": "DBMS & SQL", "num_questions": 5}
            elif action == ActionType.CREATE_CHECKLIST:
                params = {"topic": "DBMS"}
            return CampusAction(action_type=action, parameters=params)

    return CampusAction(action_type=ActionType.FINISH_TASK)


# ---------------------------------------------------------------------------
# LLM Policy
# ---------------------------------------------------------------------------

def _llm_policy(
    state_dict: Dict[str, Any],
    client: "openai.OpenAI",
    model: str = "gpt-3.5-turbo",
) -> CampusAction:
    """
    Use OpenAI GPT to decide the next action given the current state.
    Falls back to rule-based on parse error.
    """
    remaining = state_dict.get("tasks_remaining", [])
    completed = state_dict.get("completed_steps", [])

    user_msg = USER_PROMPT_TEMPLATE.format(
        state_json=json.dumps(state_dict, indent=2),
        remaining=", ".join(remaining),
        completed=", ".join(completed),
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=150,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        action_str = data.get("action", "finish_task")
        params = data.get("parameters", {})
        return CampusAction(
            action_type=ActionType(action_str),
            parameters=params,
        )
    except Exception as exc:
        print(f"  [LLM] Parse error: {exc}. Falling back to rule-based.")
        return _rule_based_policy(state_dict)


# ---------------------------------------------------------------------------
# Baseline Agent
# ---------------------------------------------------------------------------

class BaselineAgent:
    """
    LLM-based baseline agent for CampusAssistantEnv.

    If OPENAI_API_KEY is set, uses GPT to select actions.
    Otherwise uses a deterministic rule-based fallback policy.

    Usage:
        agent = BaselineAgent()
        result = agent.run("hard")
        print(result)
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        use_llm: Optional[bool] = None,
        verbose: bool = True,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.verbose = verbose
        self.seed = seed

        api_key = os.getenv("OPENAI_API_KEY", "")
        self._use_llm = (
            use_llm if use_llm is not None else bool(_OPENAI_AVAILABLE and api_key)
        )

        self._client = None
        if self._use_llm and _OPENAI_AVAILABLE and api_key:
            self._client = openai.OpenAI(api_key=api_key)

        if self.verbose:
            mode = "LLM (GPT)" if self._use_llm else "Rule-Based (no API key)"
            print(f"[BaselineAgent] Policy mode: {mode}")

    # ------------------------------------------------------------------

    def select_action(self, state_dict: Dict[str, Any]) -> CampusAction:
        """
        Select the next action given the current state observation.
        Delegates to LLM or rule-based policy.
        """
        if self._use_llm and self._client:
            return _llm_policy(state_dict, self._client, self.model)
        return _rule_based_policy(state_dict)

    # ------------------------------------------------------------------

    def run(
        self,
        task_difficulty: str = "easy",
        max_steps: int = 20,
    ) -> Dict[str, Any]:
        """
        Run a complete episode.

        Parameters
        ----------
        task_difficulty : "easy" | "medium" | "hard"
        max_steps       : safety cap on number of steps

        Returns
        -------
        dict with keys: task, difficulty, steps, final_reward, transcript
        """
        env = CampusAssistantEnv(seed=self.seed)
        init_state = env.reset(task_difficulty)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[Episode] Task : {init_state.task_name}")
            print(f"[Episode] Goal : {init_state.user_goal[:80]}...")
            print(f"[Episode] Diff : {init_state.difficulty_level.value.upper()}")
            print(f"{'='*60}\n")

        transcript: List[Dict[str, Any]] = []
        step_count = 0

        while step_count < max_steps:
            state_dict = env.state()
            action = self.select_action(state_dict)

            if self.verbose:
                print(
                    f"  Step {step_count + 1:02d} | "
                    f"Action: {action.action_type.value:<22} | "
                    f"Params: {action.parameters}"
                )

            result_obs, reward, done, info = env.step(action).to_tuple()

            transcript.append(
                {
                    "step": step_count + 1,
                    "action": action.action_type.value,
                    "parameters": action.parameters,
                    "step_reward": round(reward, 4),
                    "total_reward": round(result_obs.current_reward, 4),
                    "message": result_obs.last_message,
                    "done": done,
                }
            )

            if self.verbose:
                print(
                    f"           | Reward: {reward:.4f} (total: {result_obs.current_reward:.4f})"
                    f" | {result_obs.last_message}"
                )

            step_count += 1
            if done:
                break

        final_reward = env.state()["current_reward"]

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[Result] Steps taken : {step_count}")
            print(f"[Result] Final Reward: {final_reward:.4f}")
            print(f"{'='*60}\n")

        return {
            "task": init_state.task_name,
            "difficulty": task_difficulty,
            "steps_taken": step_count,
            "final_reward": final_reward,
            "transcript": transcript,
        }

    # ------------------------------------------------------------------

    def benchmark(self) -> Dict[str, Any]:
        """
        Run all three tasks and return aggregated benchmark results.
        Produces reproducible scores (same seed on each run).
        """
        difficulties = ["easy", "medium", "hard"]
        results = {}

        print("\n" + "=" * 60)
        print("  CAMPUS ASSISTANT ENV  —  BASELINE BENCHMARK")
        print("=" * 60)

        for diff in difficulties:
            result = self.run(task_difficulty=diff)
            results[diff] = result

        # Aggregate
        scores = [r["final_reward"] for r in results.values()]
        avg = round(sum(scores) / len(scores), 4)
        results["_summary"] = {
            "easy_score": results["easy"]["final_reward"],
            "medium_score": results["medium"]["final_reward"],
            "hard_score": results["hard"]["final_reward"],
            "average_score": avg,
        }

        print("\n" + "=" * 60)
        print("  BENCHMARK SUMMARY")
        print("=" * 60)
        for diff in difficulties:
            score = results[diff]["final_reward"]
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"  {diff.upper():8s} [{bar}] {score:.4f}")
        print(f"  {'AVERAGE':8s} [{('█' * int(avg * 20) + '░' * (20 - int(avg * 20)))}] {avg:.4f}")
        print("=" * 60 + "\n")

        return results
