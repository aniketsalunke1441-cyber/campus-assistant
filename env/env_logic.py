import json
import os
import random
from typing import List, Dict, Any, Union
from .models import ActionType, CampusAction, CampusState, TaskInfo, StepResponse
import tasks

# ── Knowledge Base ────────────────────────────────────────────────────────────

NOTES_DB = {
    "dbms": [
        "DBMS: Database Management System.",
        "SQL: Structured Query Language used for database operations.",
        "ACID Properties: Atomicity, Consistency, Isolation, Durability.",
        "Normalization ensures data integrity by reducing redundancy (1NF, 2NF, 3NF).",
        "Indexing: Improves the speed of data retrieval operations on a table.",
        "Primary Key: A unique identifier for every record in a table.",
        "ER Model: Entity-Relationship diagram representing the data structure.",
        "Transactions: Logical unit of database operations.",
        "Concurrency Control: Handling simultaneous operations securely.",
        "Integrity Constraints: Rules to maintain data consistency.",
        "NoSQL: Scalable unstructured databases like MongoDB."
    ],
    "sql": [
        "SELECT: Used to retrieve data from a database.",
        "JOIN: Combines rows from two or more tables.",
        "WHERE: Filters records based on a condition.",
        "GROUP BY: Aggregates records into summary rows.",
        "INSERT / UPDATE / DELETE: Modify database content.",
        "CREATE TABLE: Define a new database schema.",
        "TRIGGERS: Automated procedures that execute on events.",
        "VIEW: Virtual table based on a SQL query result.",
        "INDEX: Optimizer tool to speed up searches.",
        "GRANT / REVOKE: Access control permissions.",
        "DDL vs DML: Definition vs Manipulation language."
    ],
    "general": [
        "Keep your study space clean.", "Break tasks into manageable 25-minute chunks.",
        "Review your summary every 2 hours.", "Quiz yourself on definitions frequently."
    ]
}

# ── Task Configuration ────────────────────────────────────────────────────────

TASK_REGISTRY: Dict[str, TaskInfo] = {}
TASK_GRADERS: Dict[str, Any] = {}

def _load_tasks():
    global TASK_REGISTRY, TASK_GRADERS
    # Load from tasks/tasks.json relative to project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, "tasks", "tasks.json")
    
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
            for item in data:
                diff = item["difficulty"]
                TASK_REGISTRY[diff] = TaskInfo(
                    id=item["id"],
                    name=item["name"],
                    goal=item["goal"],
                    description=item["description"],
                    difficulty=diff,
                    required_steps=[ActionType(s) for s in item["required_steps"]]
                )
                # Map grader string to actual function in tasks module
                g_str = item["grader"]
                func_name = g_str.split(".")[-1]
                TASK_GRADERS[diff] = getattr(tasks, func_name)

_load_tasks()

# ── Environment Logic ─────────────────────────────────────────────────────────

class CampusAssistantEnv:
    def __init__(self, seed: int = 42):
        self._seed = seed
        self._state: CampusState = None
        self._action_history: List[ActionType] = []
        self._generated_content: Dict[str, str] = {}
        self._grader = None

    def reset(self, task_difficulty: str = "easy") -> CampusState:
        if task_difficulty not in TASK_REGISTRY:
            task_difficulty = "easy"

        conf = TASK_REGISTRY[task_difficulty]
        self._action_history = []
        self._generated_content = {}
        self._grader = TASK_GRADERS[task_difficulty]

        self._state = CampusState(
            user_goal=conf.goal,
            task_name=conf.name,
            tasks_remaining=[s.value for s in conf.required_steps],
            completed_steps=[],
            tools_used=[],
            difficulty_level=task_difficulty,
            time_remaining=360 if task_difficulty == "hard" else 240 if task_difficulty == "medium" else 120,
            last_message=f"Environment reset. Task: {conf.name} ({task_difficulty}).",
            current_reward=0.01,  # Never 0.0
            done=False,
            generated_content={}
        )
        self._state._required_steps = conf.required_steps
        self._state._completed_list = []

        return self._state

    def step(self, action_request: Union[str, Dict, CampusAction]) -> tuple:
        if self._state is None or self._state.done:
            raise Exception("Episode is finished or not initialized (call reset first).")

        # Parse action
        if isinstance(action_request, str):
            act_type = ActionType(action_request)
            params = {}
        elif isinstance(action_request, dict):
            act_val = action_request.get("action") or action_request.get("action_type")
            act_type = ActionType(act_val)
            params = action_request.get("parameters", {})
        else:
            act_type = action_request.action
            params = action_request.parameters

        # Record action
        self._action_history.append(act_type)
        if act_type.value not in self._state.tools_used:
            self._state.tools_used.append(act_type.value)

        msg = f"Action {act_type.value} executed."
        required = self._state._required_steps

        # Track completed steps
        if act_type in required and act_type not in self._state._completed_list:
            self._state._completed_list.append(act_type)
            msg = f"✅ Step completed: {act_type.value}."

            # Generate content
            if act_type == ActionType.SEARCH_NOTES:
                topic = params.get("topic", "general").lower()
                self._state.generated_content["raw_notes"] = "\n".join(
                    NOTES_DB.get(topic, NOTES_DB["general"])
                )
            elif act_type == ActionType.SUMMARIZE_NOTES:
                self._state.generated_content["summary"] = "Concise DBMS/SQL summary for exam prep."
            elif act_type == ActionType.CREATE_STUDY_PLAN:
                self._state.generated_content["study_plan"] = "2-hour DBMS study plan created."
            elif act_type == ActionType.CREATE_QUIZ:
                self._state.generated_content["quiz"] = "5 DBMS/SQL practice questions generated."
            elif act_type == ActionType.GENERATE_REMINDER:
                self._state.generated_content["reminder"] = "Reminder set for exam preparation."
            elif act_type == ActionType.CREATE_CHECKLIST:
                self._state.generated_content["checklist"] = "Exam preparation checklist created."

        # Handle finish
        if act_type == ActionType.FINISH_TASK:
            self._state.done = True
            msg = "🎉 Task finished."

        # Update state lists
        self._state.completed_steps = [s.value for s in self._state._completed_list]
        self._state.tasks_remaining = [
            s.value for s in required if s not in self._state._completed_list
        ]
        self._state.last_message = msg

        # Run grader every step
        reward = self._grader(self._state.dict())
        self._state.current_reward = reward

        # Requirement: step() returns (state, reward, done)
        return self._state, reward, self._state.done

    def state(self) -> CampusState:
        if self._state is None:
            raise Exception("Environment not initialized. Call reset() first.")
        return self._state
