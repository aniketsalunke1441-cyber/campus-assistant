import os
import requests
import time
import sys
from typing import List, Dict, Any

# Environment logic for task order (Rule-Based Agent)
TASK_SOLUTIONS = {
    "easy": ["search_notes", "summarize_notes", "finish_task"],
    "medium": ["search_notes", "summarize_notes", "create_study_plan", "generate_reminder", "finish_task"],
    "hard": ["search_notes", "summarize_notes", "create_study_plan", "create_quiz", "generate_reminder", "create_checklist", "finish_task"]
}

def run_agent(server_url: str = "http://localhost:8000", difficulty: str = "easy"):
    # Clear wait
    time.sleep(1)
    
    # 1. [START] block
    print(f"[START] task={difficulty}", flush=True)
    sys.stdout.flush()

    # Reset environment
    try:
        resp = requests.post(f"{server_url}/reset", json={"difficulty": difficulty}, timeout=10)
        resp.raise_for_status()
    except Exception:
        print(f"[END] task={difficulty} score=0.0 steps=0", flush=True)
        sys.stdout.flush()
        return
        
    data = resp.json()
    state = data["state"]
    
    actions = TASK_SOLUTIONS.get(difficulty, [])
    completed_steps = 0

    for idx, act_name in enumerate(actions, 1):
        try:
            resp = requests.post(f"{server_url}/step", json={
                "action": act_name,
                "parameters": {"topic": "dbms"} if "notes" in act_name else {}
            }, timeout=10)
            resp.raise_for_status()
        except Exception:
            break
            
        step_data = resp.json()
        reward = step_data["reward"]
        state = step_data["state"]
        done = step_data["done"]
        completed_steps = state['completed_steps']
        
        # 2. [STEP] block
        print(f"[STEP] step={idx} reward={reward:.2f}", flush=True)
        sys.stdout.flush()
        
        if done:
            break
            
    # 3. [END] block
    print(f"[END] task={difficulty} score={state['current_reward']:.2f} steps={completed_steps}", flush=True)
    sys.stdout.flush()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", "--URL", default="http://localhost:8000")
    parser.add_argument("--task", default="hard")
    args = parser.parse_args()
    run_agent(args.url, args.task)
