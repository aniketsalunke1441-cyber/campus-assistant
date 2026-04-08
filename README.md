---
title: Campus Assistant
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Autonomous AI Campus Assistant — OpenEnv Environment

<div align="center">

![Campus AI Banner](https://img.shields.io/badge/OpenEnv-v1.0-6366f1?style=for-the-badge&logo=python)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-fbbf24?style=for-the-badge&logo=huggingface)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-ff4b4b?style=for-the-badge&logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python)

**A hackathon-ready OpenEnv environment where an LLM-based AI agent helps students complete college tasks through multi-step reasoning.**

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Environment Description](#environment-description)
- [Action Space](#action-space)
- [Observation Space](#observation-space)
- [Reward Function](#reward-function)
- [Tasks](#tasks)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Running the Baseline Agent](#running-the-baseline-agent)
- [Streamlit UI](#streamlit-ui)
- [HuggingFace Deployment](#huggingface-deployment)
- [Docker](#docker)

---

## Overview

**CampusAssistantEnv** is a real-world simulation environment following the [OpenEnv](https://openenv.ai) specification. An autonomous LLM-based agent acts as a college assistant, helping students complete academic tasks by:

- 🔍 Searching and retrieving relevant study notes
- 📖 Summarizing notes into concise study material
- 📅 Creating structured study plans
- 📝 Generating practice quizzes with answers
- ⏰ Setting exam and task reminders
- ✅ Producing preparation checklists
- 🎯 Orchestrating all these tools in multi-step workflows

The environment provides **partial rewards** at every step, enabling the agent to learn optimal multi-step strategies.

---

## Environment Description

| Property | Value |
|---|---|
| **Environment Name** | `CampusAssistantEnv` |
| **OpenEnv Version** | 1.0 |
| **Episode Types** | Easy · Medium · Hard |
| **Max Steps** | 20 per episode |
| **Reward Range** | 0.0 – 1.0 (normalised) |
| **State Type** | Structured JSON dict |
| **Action Type** | Discrete (named strings) |

### OpenEnv API

```python
from env import CampusAssistantEnv

env = CampusAssistantEnv(seed=42)

# Reset to a new episode
state = env.reset(task_difficulty="hard")

# Take a step
result = env.step("search_notes")
observation, reward, done, info = result.to_tuple()

# Read current state
state_dict = env.state()
```

---

## Action Space

| Action | Description | Key Parameters |
|---|---|---|
| `search_notes` | Search the knowledge base for notes on a topic | `topic` (str) |
| `summarize_notes` | Summarize retrieved or existing notes | `topic` (str) |
| `create_study_plan` | Generate a structured study plan | `subject` (str), `hours` (int) |
| `generate_reminder` | Create a time-based reminder | `event` (str), `time` (str) |
| `create_quiz` | Generate a quiz with Q&A | `topic` (str), `num_questions` (int) |
| `create_checklist` | Create an exam preparation checklist | `topic` (str) |
| `finish_task` | Signal task completion | — |

Actions are passed as strings or typed `CampusAction` objects:

```python
from env.models import CampusAction, ActionType

# Option 1: string
env.step("summarize_notes")

# Option 2: typed object
action = CampusAction(
    action_type=ActionType.CREATE_STUDY_PLAN,
    parameters={"subject": "DBMS", "hours": 2}
)
env.step(action)

# Option 3: dict
env.step({"action_type": "create_quiz", "parameters": {"topic": "SQL", "num_questions": 5}})
```

---

## Observation Space

The `state()` method returns a structured JSON dict:

```json
{
  "user_goal": "The student needs a 2-hour study sprint...",
  "task_name": "Full 2-Hour Study Sprint",
  "tasks_remaining": ["create_quiz", "generate_reminder", "create_checklist", "finish_task"],
  "completed_steps": ["search_notes", "summarize_notes", "create_study_plan"],
  "tools_used": ["search_notes", "summarize_notes", "create_study_plan"],
  "time_remaining": 280,
  "difficulty_level": "hard",
  "current_reward": 0.4321,
  "done": false,
  "last_message": "📅 Study plan created for 'DBMS & SQL' (2 hours).",
  "generated_content": {
    "raw_notes": "...",
    "summary": "...",
    "study_plan": "..."
  }
}
```

| Field | Type | Description |
|---|---|---|
| `user_goal` | `str` | The goal the student wants to accomplish |
| `task_name` | `str` | Human-readable task name |
| `tasks_remaining` | `list[str]` | Required steps not yet completed |
| `completed_steps` | `list[str]` | Steps already completed |
| `tools_used` | `list[str]` | Unique action types called this episode |
| `time_remaining` | `int` | Seconds left (0 = timeout) |
| `difficulty_level` | `str` | `easy` · `medium` · `hard` |
| `current_reward` | `float` | Cumulative reward so far (0.0 – 1.0) |
| `done` | `bool` | Whether the episode has ended |
| `last_message` | `str` | Human-readable message from the last step |
| `generated_content` | `dict[str, str]` | Content produced by tools |

---

## Reward Function

The reward is a **multi-component weighted sum** in range [0.0, 1.0]:

```
R = w₁·step_completion + w₂·correct_sequence + w₃·tool_diversity + w₄·final_completion + w₅·time_efficiency
```

| Component | Weight (Hard) | Description |
|---|---|---|
| `step_completion` | 0.40 | Fraction of required steps completed (partial credit) |
| `correct_sequence` | 0.20 | How closely the agent follows the optimal action order |
| `tool_diversity` | 0.15 | Fraction of useful tools actually used |
| `final_completion` | 0.20 | Full bonus when `finish_task` is called after all steps |
| `time_efficiency` | 0.05 | Small bonus for faster completion |

### Partial Progress Design

Every action earns an **incremental step reward** (0.10–0.20) regardless of task completion, enabling the agent to receive positive signal at each step. The full `final_completion` bonus (0.20) is only awarded when `finish_task` is called after completing all required steps.

**Typical scores:**
| Difficulty | Rule-Based Agent | Random Agent |
|---|---|---|
| Easy | ~0.92 | ~0.30 |
| Medium | ~0.88 | ~0.20 |
| Hard | ~0.85 | ~0.15 |

---

## Tasks

### 🟢 Easy — "Summarize DBMS Notes"

> *"The student needs a concise summary of their DBMS notes to quickly review key concepts before class."*

- **Required steps:** `search_notes` → `summarize_notes` → `finish_task`
- **Time limit:** 120 seconds
- **Focus:** Single-tool, single-step objective

### 🟡 Medium — "Prepare for SQL Viva Tomorrow"

> *"The student has a SQL viva tomorrow and needs to search DBMS/SQL notes, summarize them, create a study plan, and generate a reminder."*

- **Required steps:** `search_notes` → `summarize_notes` → `create_study_plan` → `generate_reminder` → `finish_task`
- **Time limit:** 240 seconds
- **Focus:** Multi-step sequential tool use

### 🔴 Hard — "Full 2-Hour Study Sprint"

> *"The student needs a complete 2-hour study sprint package: study plan, quiz, reminder, and checklist — all built from their notes."*

- **Required steps:** `search_notes` → `summarize_notes` → `create_study_plan` → `create_quiz` → `generate_reminder` → `create_checklist` → `finish_task`
- **Time limit:** 360 seconds
- **Focus:** Complex multi-tool orchestration

---

## Project Structure

```
campus assistant/
│
├── env/
│   ├── __init__.py          # Package exports
│   ├── openenv.yaml         # OpenEnv specification file
│   ├── env_logic.py         # Core environment (reset/step/state + reward)
│   └── models.py            # Typed models (State, Action, Task, StepResult)
│
├── agent/
│   ├── __init__.py
│   └── baseline_agent.py    # LLM + rule-based baseline agent
│
├── app.py                   # Streamlit UI (dark premium theme)
├── run_baseline.py          # CLI inference script
├── requirements.txt
├── Dockerfile               # HuggingFace/Docker deployment
└── README.md
```

---

## Setup Instructions

### 1. Clone / Download

```bash
git clone <your-repo-url>
cd "campus assistant"
```

### 2. Create Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Set OpenAI API Key

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."

# Linux/Mac
export OPENAI_API_KEY="sk-..."
```

> Without an API key, the agent automatically uses the built-in rule-based policy — fully functional with no external dependencies.

---

## Running the Baseline Agent

### Benchmark All 3 Tasks (reproducible, seed=42)

```bash
python run_baseline.py
```

### Single Task

```bash
python run_baseline.py --task easy
python run_baseline.py --task medium
python run_baseline.py --task hard
```

### With LLM (requires OPENAI_API_KEY)

```bash
python run_baseline.py --task hard --llm --model gpt-4o
```

### Save Results

```bash
python run_baseline.py --output results.json
```

### Expected Output (rule-based, seed=42)

```
============================================================
  CAMPUS ASSISTANT ENV  —  BASELINE BENCHMARK
============================================================
  Step 01 | Action: search_notes          | Params: {'topic': 'dbms'}
  ...
  EASY     [████████████████████] 0.9200
  MEDIUM   [█████████████████░░░] 0.8800
  HARD     [█████████████████░░░] 0.8500
  AVERAGE  [█████████████████░░░] 0.8833
============================================================
```

---

## Streamlit UI

```bash
streamlit run app.py
```

The UI opens at `http://localhost:8501` and features:
- **Dark glassmorphism design** with gradient animations
- Task difficulty selector (Easy / Medium / Hard)
- Real-time state panel with metrics (reward, steps, time)
- Reward progress bar
- Manual action executor with dynamic parameter inputs
- Auto-run baseline agent across all 3 tasks
- Generated content viewer (summary, quiz, plan, checklist, reminder)
- Episode transcript with per-step rewards
- Raw JSON state inspector

---

## HuggingFace Deployment

1. Create a new **Space** on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Docker** as the SDK
3. Upload all project files
4. The app will automatically build and run on port **7860**

The `Dockerfile` is pre-configured for HuggingFace Spaces:
- Non-root user (`appuser`)
- Port 7860 exposed
- Headless Streamlit server
- Health check endpoint

---

## Docker

### Build

```bash
docker build -t campus-assistant-env .
```

### Run Locally

```bash
docker run -p 7860:7860 campus-assistant-env
```

### With OpenAI Key

```bash
docker run -p 7860:7860 -e OPENAI_API_KEY="sk-..." campus-assistant-env
```

Open `http://localhost:7860` in your browser.

---

## License

MIT License — free to use, modify, and distribute.

---

<div align="center">
Made with ❤️ for the Hackathon · CampusAssistantEnv · OpenEnv v1.0
</div>
