"""
env_logic.py - Core CampusAssistantEnv logic

Implements the OpenEnv API:
  - reset()  → CampusState
  - step(action) → StepResult
  - state()  → Dict

Also contains:
  - TASK_REGISTRY : all three task definitions
  - Action handlers that simulate realistic tool outputs
  - Reward computation with partial progress signals
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Union

from .models import (
    ActionType,
    CampusAction,
    CampusState,
    DifficultyLevel,
    StepResult,
    TaskDefinition,
)

# ---------------------------------------------------------------------------
# Simulated Knowledge Base
# ---------------------------------------------------------------------------

NOTES_DB: Dict[str, str] = {
    "dbms": (
        "DBMS Notes:\n"
        "• DBMS (Database Management System) is software for storing, retrieving, and managing data.\n"
        "• Key concepts: Data Independence, ACID properties, ER Model, Normalization (1NF–BCNF).\n"
        "• Transactions: Atomicity, Consistency, Isolation, Durability.\n"
        "• SQL commands: DDL (CREATE, DROP, ALTER), DML (SELECT, INSERT, UPDATE, DELETE).\n"
        "• Indexing: B-Tree, Hash Index, Clustered vs Non-Clustered.\n"
        "• Relational Algebra: Selection (σ), Projection (π), Join (⋈), Union (∪).\n"
        "• Normal Forms: 1NF (no repeating groups), 2NF (full functional dependency),\n"
        "  3NF (no transitive dependency), BCNF (every determinant is a candidate key).\n"
        "• Concurrency Control: Two-Phase Locking (2PL), Timestamp Ordering.\n"
        "• Recovery: Log-based recovery, Checkpointing, Shadow Paging."
    ),
    "sql": (
        "SQL Notes:\n"
        "• SELECT col FROM table WHERE cond ORDER BY col LIMIT n;\n"
        "• JOINs: INNER JOIN (matching rows), LEFT JOIN (all left rows), RIGHT JOIN, FULL OUTER JOIN.\n"
        "• GROUP BY with aggregate functions: COUNT, SUM, AVG, MAX, MIN.\n"
        "• HAVING: filter after GROUP BY (unlike WHERE which filters before).\n"
        "• Subqueries: correlated vs non-correlated.\n"
        "• Views: virtual tables created from SELECT statements.\n"
        "• Stored Procedures & Triggers.\n"
        "• Indexes speed up queries; can be unique or composite.\n"
        "• Transactions: BEGIN, COMMIT, ROLLBACK, SAVEPOINT.\n"
        "• Common SQL interview topics: joins, subqueries, window functions (ROW_NUMBER, RANK, DENSE_RANK)."
    ),
    "os": (
        "Operating Systems Notes:\n"
        "• Process management: Process states, Context switching, PCB.\n"
        "• CPU Scheduling: FCFS, SJF, Round Robin, Priority Scheduling.\n"
        "• Memory Management: Paging, Segmentation, Virtual Memory, TLB.\n"
        "• Deadlocks: Conditions, Prevention, Avoidance (Banker's Algorithm), Detection.\n"
        "• File Systems: FAT, NTFS, Inodes, Directory structures.\n"
        "• Synchronization: Semaphores, Mutex, Monitors."
    ),
    "networks": (
        "Computer Networks Notes:\n"
        "• OSI Model: 7 layers — Physical, Data Link, Network, Transport, Session, Presentation, Application.\n"
        "• TCP/IP: 4-layer model. Protocols: HTTP, FTP, SMTP, DNS, DHCP.\n"
        "• IP Addressing: IPv4, IPv6, Subnetting, CIDR.\n"
        "• Routing: RIP, OSPF, BGP.\n"
        "• Error Detection: CRC, Checksum, Parity.\n"
        "• Security: SSL/TLS, Firewalls, NAT."
    ),
}


def _get_notes(topic: str) -> str:
    """Return notes for a topic, with fuzzy fallback."""
    topic_lower = topic.lower()
    for key, val in NOTES_DB.items():
        if key in topic_lower or topic_lower in key:
            return val
    # Fallback: combine DBMS + SQL
    return NOTES_DB["dbms"] + "\n\n" + NOTES_DB["sql"]


# ---------------------------------------------------------------------------
# Task Registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: Dict[str, TaskDefinition] = {
    "easy": TaskDefinition(
        task_id="task_easy_summarize_dbms",
        name="Summarize DBMS Notes",
        user_goal=(
            "The student needs a concise summary of their DBMS (Database Management Systems) "
            "notes to quickly review key concepts before class."
        ),
        difficulty=DifficultyLevel.EASY,
        required_steps=[
            ActionType.SEARCH_NOTES,
            ActionType.SUMMARIZE_NOTES,
            ActionType.FINISH_TASK,
        ],
        optional_steps=[ActionType.CREATE_CHECKLIST],
        time_limit=120,
        reward_weights={
            "step_completion": 0.50,
            "correct_sequence": 0.20,
            "tool_diversity": 0.10,
            "final_completion": 0.15,
            "time_efficiency": 0.05,
        },
        description=(
            "Single-step summarization task. Search DBMS notes and produce a summary, "
            "then signal completion."
        ),
    ),
    "medium": TaskDefinition(
        task_id="task_medium_sql_viva",
        name="Prepare for SQL Viva Tomorrow",
        user_goal=(
            "The student has a SQL viva tomorrow and needs to search DBMS/SQL notes, "
            "summarize them, create a study plan, and generate a reminder."
        ),
        difficulty=DifficultyLevel.MEDIUM,
        required_steps=[
            ActionType.SEARCH_NOTES,
            ActionType.SUMMARIZE_NOTES,
            ActionType.CREATE_STUDY_PLAN,
            ActionType.GENERATE_REMINDER,
            ActionType.FINISH_TASK,
        ],
        optional_steps=[ActionType.CREATE_QUIZ],
        time_limit=240,
        reward_weights={
            "step_completion": 0.40,
            "correct_sequence": 0.20,
            "tool_diversity": 0.15,
            "final_completion": 0.20,
            "time_efficiency": 0.05,
        },
        description=(
            "Multi-step preparation task. Retrieve notes, summarize, build study plan, "
            "set reminder, then finish."
        ),
    ),
    "hard": TaskDefinition(
        task_id="task_hard_full_prep",
        name="Full 2-Hour Study Sprint",
        user_goal=(
            "The student needs a complete 2-hour study sprint package: a structured study "
            "plan, a quiz to test their knowledge, a reminder for the exam, and a "
            "preparation checklist — all built from their notes."
        ),
        difficulty=DifficultyLevel.HARD,
        required_steps=[
            ActionType.SEARCH_NOTES,
            ActionType.SUMMARIZE_NOTES,
            ActionType.CREATE_STUDY_PLAN,
            ActionType.CREATE_QUIZ,
            ActionType.GENERATE_REMINDER,
            ActionType.CREATE_CHECKLIST,
            ActionType.FINISH_TASK,
        ],
        optional_steps=[],
        time_limit=360,
        reward_weights={
            "step_completion": 0.40,
            "correct_sequence": 0.20,
            "tool_diversity": 0.15,
            "final_completion": 0.20,
            "time_efficiency": 0.05,
        },
        description=(
            "Complex multi-tool workflow. Orchestrate all available tools in a meaningful "
            "sequence to produce a full exam-preparation package."
        ),
    ),
}

# Optimal action sequences for sequence reward scoring
OPTIMAL_SEQUENCES: Dict[str, List[ActionType]] = {
    "easy": [
        ActionType.SEARCH_NOTES,
        ActionType.SUMMARIZE_NOTES,
        ActionType.FINISH_TASK,
    ],
    "medium": [
        ActionType.SEARCH_NOTES,
        ActionType.SUMMARIZE_NOTES,
        ActionType.CREATE_STUDY_PLAN,
        ActionType.GENERATE_REMINDER,
        ActionType.FINISH_TASK,
    ],
    "hard": [
        ActionType.SEARCH_NOTES,
        ActionType.SUMMARIZE_NOTES,
        ActionType.CREATE_STUDY_PLAN,
        ActionType.CREATE_QUIZ,
        ActionType.GENERATE_REMINDER,
        ActionType.CREATE_CHECKLIST,
        ActionType.FINISH_TASK,
    ],
}


# ---------------------------------------------------------------------------
# Action Content Generators
# ---------------------------------------------------------------------------

def _gen_summary(topic: str, notes: str) -> str:
    lines = [l for l in notes.splitlines() if l.startswith("•")][:6]
    bullet_block = "\n".join(lines)
    return (
        f"📖 Summary of {topic.upper()} Notes:\n"
        f"{bullet_block}\n\n"
        f"Key takeaway: Master ACID properties, SQL JOINs, Normalization, and Indexing."
    )


def _gen_study_plan(subject: str, hours: int = 2) -> str:
    slot = max(1, hours // 2)
    return (
        f"📅 {hours}-Hour Study Plan for {subject}:\n"
        f"  Hour 1 (0–{slot}h): Review core concepts — ER Model, Normalization, ACID.\n"
        f"  Hour 2 ({slot}–{hours}h): Practice SQL queries, JOINs, and past exam questions.\n"
        f"  Last 15 min: Quick revision of important formulas and definitions.\n"
        f"  Tip: Take a 5-min break every 45 minutes for better retention."
    )


def _gen_quiz(topic: str, num_q: int = 5) -> str:
    questions = [
        ("What does ACID stand for?",
         "Atomicity, Consistency, Isolation, Durability"),
        ("Which normal form removes transitive dependencies?",
         "Third Normal Form (3NF)"),
        ("What is the difference between WHERE and HAVING?",
         "WHERE filters before aggregation; HAVING filters after GROUP BY"),
        ("Name two types of JOINs in SQL.",
         "INNER JOIN and LEFT JOIN (also RIGHT JOIN, FULL OUTER JOIN)"),
        ("What is a primary key?",
         "A column (or set of columns) that uniquely identifies each row in a table"),
        ("What is the purpose of an index?",
         "To speed up data retrieval without scanning the entire table"),
        ("Explain the difference between DELETE and TRUNCATE.",
         "DELETE removes specific rows and can be rolled back; TRUNCATE removes all rows faster and cannot be rolled back"),
    ]
    selected = questions[: min(num_q, len(questions))]
    quiz_lines = [f"📝 Quiz: {topic} ({len(selected)} Questions)\n"]
    for i, (q, a) in enumerate(selected, 1):
        quiz_lines.append(f"Q{i}. {q}")
        quiz_lines.append(f"   ✅ Answer: {a}\n")
    return "\n".join(quiz_lines)


def _gen_reminder(event: str, time_str: str = "8:00 AM") -> str:
    return (
        f"⏰ Reminder Set!\n"
        f"  📌 Event : {event}\n"
        f"  🕗 Time  : {time_str} tomorrow\n"
        f"  📱 Alert : 30 minutes before the event\n"
        f"  Note: Ensure your notes and study materials are ready the night before."
    )


def _gen_checklist(topic: str) -> str:
    return (
        f"✅ Exam Preparation Checklist for {topic}:\n"
        f"  [ ] Review all lecture notes and slides\n"
        f"  [ ] Summarize key concepts on flashcards\n"
        f"  [ ] Complete at least 2 past exam papers\n"
        f"  [ ] Practice SQL queries on a local database\n"
        f"  [ ] Understand all Normal Forms with examples\n"
        f"  [ ] Review ACID properties and transaction isolation levels\n"
        f"  [ ] Sleep at least 7 hours before the exam\n"
        f"  [ ] Pack materials (pens, student ID) the night before"
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CampusAssistantEnv:
    """
    OpenEnv-compatible environment simulating a college AI assistant.

    Public API
    ----------
    reset(task_difficulty)  → CampusState
    step(action)            → StepResult
    state()                 → Dict[str, Any]
    """

    VALID_DIFFICULTIES = list(TASK_REGISTRY.keys())

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._state: CampusState = CampusState()
        self._task: Optional[TaskDefinition] = None
        self._episode_start: float = 0.0
        self._action_sequence: List[ActionType] = []  # track call order

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, task_difficulty: str = "easy") -> CampusState:
        """
        Reset the environment to the start of a new episode.

        Parameters
        ----------
        task_difficulty : one of "easy", "medium", "hard"

        Returns
        -------
        CampusState – the initial observation
        """
        if task_difficulty not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown difficulty '{task_difficulty}'. "
                f"Choose from {self.VALID_DIFFICULTIES}"
            )

        self._task = TASK_REGISTRY[task_difficulty]
        self._episode_start = time.time()
        self._action_sequence = []

        self._state = CampusState(
            user_goal=self._task.user_goal,
            task_name=self._task.name,
            tasks_remaining=[s.value for s in self._task.required_steps],
            completed_steps=[],
            tools_used=[],
            time_remaining=self._task.time_limit,
            difficulty_level=self._task.difficulty,
            current_reward=0.0,
            done=False,
            last_message=f"New episode started. Goal: {self._task.name}",
            generated_content={},
        )
        return self._state

    def step(self, action: Union[CampusAction, str, Dict]) -> StepResult:
        """
        Execute one action in the environment.

        Parameters
        ----------
        action : CampusAction | str | dict
            The action to execute. Accepts typed objects, raw strings, or dicts.

        Returns
        -------
        StepResult(observation, reward, done, info)
        """
        if self._task is None:
            raise RuntimeError("Call reset() before step().")

        # Normalise input
        if isinstance(action, str):
            action = CampusAction.from_string(action)
        elif isinstance(action, dict):
            action = CampusAction.from_dict(action)

        if self._state.done:
            return StepResult(
                observation=self._state,
                reward=0.0,
                done=True,
                info={"message": "Episode already done. Call reset() to start again."},
            )

        # Update time
        elapsed = time.time() - self._episode_start
        self._state.time_remaining = max(0, self._task.time_limit - int(elapsed))

        if self._state.time_remaining == 0:
            self._state.done = True
            self._state.last_message = "⏰ Time's up! Episode ended."
            return StepResult(
                observation=self._state,
                reward=0.0,
                done=True,
                info={"reason": "timeout"},
            )

        # Execute action & get step reward
        step_reward, message, content_key, content_value = self._execute_action(action)

        # Update state
        atype = action.action_type
        if atype not in self._action_sequence:
            self._action_sequence.append(atype)

        if atype.value in self._state.tasks_remaining:
            self._state.tasks_remaining.remove(atype.value)
            self._state.completed_steps.append(atype.value)

        if atype.value not in self._state.tools_used:
            self._state.tools_used.append(atype.value)

        if content_key:
            self._state.generated_content[content_key] = content_value

        self._state.last_message = message

        # Compute full reward
        total_reward = self._compute_reward(action, step_reward)
        self._state.current_reward = min(1.0, total_reward)

        done = atype == ActionType.FINISH_TASK or self._state.time_remaining == 0
        self._state.done = done

        return StepResult(
            observation=self._state,
            reward=step_reward,
            done=done,
            info={
                "action": atype.value,
                "tasks_remaining": list(self._state.tasks_remaining),
                "completed_steps": list(self._state.completed_steps),
                "time_remaining": self._state.time_remaining,
                "total_reward": self._state.current_reward,
            },
        )

    def state(self) -> Dict[str, Any]:
        """Return the current observation as a plain JSON-serialisable dict."""
        return self._state.to_dict()

    # ------------------------------------------------------------------
    # Action Handlers
    # ------------------------------------------------------------------

    def _execute_action(
        self, action: CampusAction
    ) -> tuple[float, str, Optional[str], Optional[str]]:
        """
        Dispatch action to its handler.

        Returns (step_reward, message, content_key, content_value)
        """
        params = action.parameters
        atype = action.action_type

        if atype == ActionType.SEARCH_NOTES:
            topic = params.get("topic", "dbms")
            notes = _get_notes(topic)
            self._state.generated_content["raw_notes"] = notes
            return (
                0.10,
                f"🔍 Searched notes for topic: '{topic}'. Found {len(notes.splitlines())} lines.",
                "raw_notes",
                notes,
            )

        elif atype == ActionType.SUMMARIZE_NOTES:
            topic = params.get("topic", "DBMS")
            raw = self._state.generated_content.get("raw_notes", _get_notes(topic))
            summary = _gen_summary(topic, raw)
            return (
                0.15,
                f"📖 Summary generated for '{topic}'.",
                "summary",
                summary,
            )

        elif atype == ActionType.CREATE_STUDY_PLAN:
            subject = params.get("subject", "DBMS & SQL")
            hours = int(params.get("hours", 2))
            plan = _gen_study_plan(subject, hours)
            return (
                0.15,
                f"📅 Study plan created for '{subject}' ({hours} hours).",
                "study_plan",
                plan,
            )

        elif atype == ActionType.GENERATE_REMINDER:
            event = params.get("event", "SQL Viva")
            t = params.get("time", "8:00 AM")
            reminder = _gen_reminder(event, t)
            return (
                0.10,
                f"⏰ Reminder set for '{event}' at {t}.",
                "reminder",
                reminder,
            )

        elif atype == ActionType.CREATE_QUIZ:
            topic = params.get("topic", "DBMS & SQL")
            num_q = int(params.get("num_questions", 5))
            quiz = _gen_quiz(topic, num_q)
            return (
                0.15,
                f"📝 Quiz created for '{topic}' with {num_q} questions.",
                "quiz",
                quiz,
            )

        elif atype == ActionType.CREATE_CHECKLIST:
            topic = params.get("topic", "DBMS")
            checklist = _gen_checklist(topic)
            return (
                0.10,
                f"✅ Preparation checklist created for '{topic}'.",
                "checklist",
                checklist,
            )

        elif atype == ActionType.FINISH_TASK:
            all_done = not bool(self._state.tasks_remaining) or (
                set(self._state.tasks_remaining) == {ActionType.FINISH_TASK.value}
            )
            if all_done:
                return (
                    0.20,
                    "🎉 Task completed successfully! Great work!",
                    None,
                    None,
                )
            else:
                remaining = ", ".join(self._state.tasks_remaining)
                return (
                    0.05,
                    f"⚠️ Finished early — steps still pending: {remaining}",
                    None,
                    None,
                )

        return (0.0, f"Unknown action: {atype}", None, None)

    # ------------------------------------------------------------------
    # Reward Function
    # ------------------------------------------------------------------

    def _compute_reward(self, action: CampusAction, step_reward: float) -> float:
        """
        Multi-component reward with partial progress signals.

        Components:
          1. step_completion  – fraction of required steps completed
          2. correct_sequence – how well the agent follows the optimal sequence
          3. tool_diversity   – unique relevant tools used
          4. final_completion – bonus when finish_task is called after all steps
          5. time_efficiency  – small bonus for faster completion
        """
        if self._task is None:
            return 0.0

        weights = self._task.reward_weights
        required = self._task.required_steps
        total = len(required)

        # 1. Step completion (partial credit)
        n_done = len(self._state.completed_steps)
        r_step = (n_done / total) if total > 0 else 0.0

        # 2. Correct sequence reward
        optimal = OPTIMAL_SEQUENCES.get(self._task.difficulty.value, required)
        matches = sum(
            1
            for i, act in enumerate(self._action_sequence)
            if i < len(optimal) and act == optimal[i]
        )
        r_seq = (matches / len(optimal)) if optimal else 0.0

        # 3. Tool diversity
        used_set = set(self._state.tools_used)
        useful = {a.value for a in required}
        r_div = len(used_set & useful) / len(useful) if useful else 0.0

        # 4. Final completion
        remaining_non_finish = [
            s for s in self._state.tasks_remaining
            if s != ActionType.FINISH_TASK.value
        ]
        is_complete_finish = (
            action.action_type == ActionType.FINISH_TASK and not remaining_non_finish
        )
        r_final = 1.0 if is_complete_finish else 0.0

        # 5. Time efficiency
        elapsed = time.time() - self._episode_start
        time_fraction_used = min(1.0, elapsed / self._task.time_limit)
        r_time = max(0.0, 1.0 - time_fraction_used)

        total_reward = (
            weights.get("step_completion", 0.40) * r_step
            + weights.get("correct_sequence", 0.20) * r_seq
            + weights.get("tool_diversity", 0.15) * r_div
            + weights.get("final_completion", 0.20) * r_final
            + weights.get("time_efficiency", 0.05) * r_time
        )

        return round(min(1.0, total_reward), 4)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def available_actions(self) -> List[str]:
        """Return the list of valid action type strings."""
        return ActionType.values()

    def task_info(self) -> Optional[Dict[str, Any]]:
        """Return the current task definition as a dict."""
        if self._task:
            return self._task.to_dict()
        return None
