import os
import requests
import time
from typing import List, Dict, Any

# Environment logic for task order (Rule-Based Agent)
# This mimics the agent side of the environment.

TASK_SOLUTIONS = {
    "easy": ["search_notes", "summarize_notes", "finish_task"],
    "medium": ["search_notes", "summarize_notes", "create_study_plan", "generate_reminder", "finish_task"],
    "hard": ["search_notes", "summarize_notes", "create_study_plan", "create_quiz", "generate_reminder", "create_checklist", "finish_task"]
}

def run_agent(server_url: str = "http://localhost:8000", difficulty: str = "easy"):
    """
    OpenEnv Baseline Inference Script.
    Communicates via REST API with the Campus Assistant environment.
    """
    print(f"\n--- OpenEnv Baseline Agent | Running {difficulty} task ---", flush=True)
    
    # 1. [START] block
    print(f"[START] task={difficulty}", flush=True)

    # Reset environment
    print(f"POST {server_url}/reset...", flush=True)
    try:
        resp = requests.post(f"{server_url}/reset", json={"difficulty": difficulty})
        resp.raise_for_status()
    except Exception as e:
        print(f"Error reset: {e}", flush=True)
        return
        
    data = resp.json()
    state = data["state"]
    print(f"Initial State | Goal: {state['goal']}", flush=True)
    
    # Solution path (Perfect Rule-Based Agent)
    actions = TASK_SOLUTIONS.get(difficulty, [])
    
    total_reward = 0.0
    completed_steps = 0

    for idx, act_name in enumerate(actions, 1):
        # Step action
        print(f"Step {idx:02} | Executing: {act_name}", flush=True)
        try:
            resp = requests.post(f"{server_url}/step", json={
                "action": act_name,
                "parameters": {"topic": "dbms"} if "notes" in act_name else {}
            })
            resp.raise_for_status()
        except Exception as e:
            print(f"Error step: {e}", flush=True)
            break
            
        step_data = resp.json()
        reward = step_data["reward"]
        total_reward += reward
        state = step_data["state"]
        done = step_data["done"]
        completed_steps = state['completed_steps']
        
        # 2. [STEP] block
        print(f"[STEP] step={idx} reward={reward:.4f}", flush=True)
        print(f"        | Reward: {reward:.2f} | Cumulative: {state['current_reward']:.4f}", flush=True)
        
        if done:
            break
            
    # 3. [END] block
    print(f"[END] task={difficulty} score={state['current_reward']:.4f} steps={completed_steps}", flush=True)

    print(f"\nTask Finished | Final Reward Sum: {total_reward:.4f} | State Reward: {state['current_reward']:.4f}", flush=True)
    print(f"Steps: {completed_steps} / {completed_steps + state['tasks_remaining']}", flush=True)
    
    if state["current_reward"] >= 0.85:
        print("✅ SUCCESS: PERFECT SUBMISSION SCORE", flush=True)
    else:
        print("⚠️ PARTIAL SCORE: TRY AGAIN", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OpenEnv Baseline Inference Script")
    parser.add_argument("--url", "--URL", default="http://localhost:8000", help="API URL (e.g. http://localhost:8000)")
    parser.add_argument("--task", default="hard", help="Task difficulty (easy, medium, hard)")
    args = parser.parse_args()
    
    run_agent(args.url, args.task)
