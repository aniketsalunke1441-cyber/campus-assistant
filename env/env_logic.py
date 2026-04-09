import random
from typing import List, Dict, Any, Union
from .models import ActionType, CampusAction, CampusState
from tasks import grade_summarize_notes, grade_prepare_viva, grade_complete_study_workflow

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

TASK_CONFIGS = {
    "easy": {
        "id": "summarize_notes",
        "name": "Quick Review",
        "goal": "Summarize notes: search DBMS notes and generate a quick summary.",
        "steps": [ActionType.SEARCH_NOTES, ActionType.SUMMARIZE_NOTES, ActionType.FINISH_TASK],
        "grader": grade_summarize_notes,
    },
    "medium": {
        "id": "prepare_viva",
        "name": "Prepare Viva",
        "goal": "Prepare for viva: search notes, create a study plan, and generate a quiz.",
        "steps": [ActionType.SEARCH_NOTES, ActionType.CREATE_STUDY_PLAN, ActionType.CREATE_QUIZ, ActionType.FINISH_TASK],
        "grader": grade_prepare_viva,
    },
    "hard": {
        "id": "complete_study_workflow",
        "name": "Full Sprint",
        "goal": "Complete the full study workflow: plan, quiz, checklist, and reminder.",
        "steps": [ActionType.SEARCH_NOTES, ActionType.CREATE_STUDY_PLAN, ActionType.CREATE_QUIZ,
                  ActionType.GENERATE_REMINDER, ActionType.CREATE_CHECKLIST, ActionType.FINISH_TASK],
        "grader": grade_complete_study_workflow,
    }
}

# ── Environment Logic ─────────────────────────────────────────────────────────

class CampusAssistantEnv:
    def __init__(self, seed: int = 42):
        self._seed = seed
        self._state: CampusState = None
        self._action_history: List[ActionType] = []
        self._generated_content: Dict[str, str] = {}
        self._grader = None

    def reset(self, difficulty: str = "easy") -> CampusState:
        if difficulty not in TASK_CONFIGS:
            difficulty = "easy"

        conf = TASK_CONFIGS[difficulty]
        self._action_history = []
        self._generated_content = {}
        self._grader = conf["grader"]

        self._state = CampusState(
            user_goal=conf["goal"],
            task_name=conf["name"],
            tasks_remaining=[s.value for s in conf["steps"]],
            completed_steps=[],
            tools_used=[],
            difficulty_level=difficulty,
            time_remaining=360 if difficulty == "hard" else 240 if difficulty == "medium" else 120,
            last_message=f"Environment reset. Task: {conf['name']} ({difficulty}).",
            current_reward=0.0,
            done=False,
            generated_content={}
        )
        self._state._required_steps = conf["steps"]
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

        # Binary reward: run grader only when done, else 0.0
        if self._state.done and self._grader:
            state_dict = self._state.dict()
            reward = self._grader(state_dict)  # returns 1.0 or 0.0
        else:
            reward = 0.0

        self._state.current_reward = reward

        return (self._state, reward, self._state.done, {})

    def state(self) -> CampusState:
        if self._state is None:
            raise Exception("Environment not initialized. Call reset() first.")
        return self._state
