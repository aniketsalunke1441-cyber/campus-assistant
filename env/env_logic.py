import random
from typing import List, Dict, Any, Union
from .models import ActionType, CampusAction, CampusState

# ── Knowledge Base (Predefined notes) ────────────────────────────────────────

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

# ── Env Configuration ────────────────────────────────────────────────────────

TASK_CONFIGS = {
    "easy": {
        "name": "Quick Review",
        "description": "Just search notes and summarize them.",
        "goal": "Search DBMS notes and generate a quick summary for class.",
        "steps": [ActionType.SEARCH_NOTES, ActionType.SUMMARIZE_NOTES, ActionType.FINISH_TASK]
    },
    "medium": {
        "name": "Exam Prep",
        "description": "Study plan and reminders for SQL exam.",
        "goal": "Prepare for SQL viva: search, summarize, plan tasks, and set a reminder.",
        "steps": [ActionType.SEARCH_NOTES, ActionType.SUMMARIZE_NOTES, ActionType.CREATE_STUDY_PLAN, ActionType.GENERATE_REMINDER, ActionType.FINISH_TASK]
    },
    "hard": {
        "name": "Full Sprint",
        "description": "Comprehensive study package.",
        "goal": "Complete 2-hour sprint: full notes package with quiz, plan, reminder and checklist.",
        "steps": [ActionType.SEARCH_NOTES, ActionType.SUMMARIZE_NOTES, ActionType.CREATE_STUDY_PLAN, ActionType.CREATE_QUIZ, ActionType.GENERATE_REMINDER, ActionType.CREATE_CHECKLIST, ActionType.FINISH_TASK]
    }
}

# ── Environment Logic ──────────────────────────────────────────────────────────

class CampusAssistantEnv:
    def __init__(self, seed: int = 42):
        self._seed = seed
        self._state: CampusState = None
        self._action_history: List[ActionType] = []
        self._generated_content: Dict[str, str] = {}

    def reset(self, difficulty: str = "easy") -> CampusState:
        if difficulty not in TASK_CONFIGS:
            difficulty = "easy"
            
        conf = TASK_CONFIGS[difficulty]
        self._action_history = []
        self._generated_content = {}
        
        self._state = CampusState(
            goal=conf["goal"],
            tasks_remaining=len(conf["steps"]),
            completed_steps=0,
            difficulty=difficulty,
            last_message=f"Environment reset. Task: {conf['name']} ({difficulty}).",
            current_reward=0.0,
            done=False
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
            act_type = ActionType(action_request["action"])
            params = action_request.get("parameters", {})
        else:
            act_type = action_request.action
            params = action_request.parameters

        # Record action
        self._action_history.append(act_type)
        
        # Reward logic
        reward = 0.0
        msg = f"Action {act_type.value} executed."
        
        required = self._state._required_steps
        
        # Check if action was required and not yet completed
        if act_type in required and act_type not in self._state._completed_list:
            self._state._completed_list.append(act_type)
            reward += 0.15 # Small reward for a correct step
            msg = f"✅ Progress! {act_type.value} completed."
        
        # Finish check
        if act_type == ActionType.FINISH_TASK:
            self._state.done = True
            # Bonus reward if everything is done
            if len(self._state._completed_list) == len(required):
                reward += 0.5
                msg = "🎉 ALL STEPS COMPLETED! Excellent job."
            else:
                msg = "⚠️ Task finished early. Missing steps."

        # Update counters
        self._state.completed_steps = len(self._state._completed_list)
        self._state.tasks_remaining = max(0, len(required) - self._state.completed_steps)
        self._state.last_message = msg
        self._state.current_reward = min(1.0, self._state.current_reward + reward)

        return (self._state, reward, self._state.done, {})

    def state(self) -> CampusState:
        return self._state
