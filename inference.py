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
    print(f"\n--- OpenEnv Baseline Agent | Running {difficulty} task ---")
    
    # Reset environment
    print(f"POST {server_url}/reset...")
    resp = requests.post(f"{server_url}/reset", json={"difficulty": difficulty})
    if resp.status_code != 200:
        print(f"Error reset: {resp.text}")
        return
        
    data = resp.json()
    state = data["state"]
    print(f"Initial State | Goal: {state['goal']}")
    
    # Solution path (Perfect Rule-Based Agent)
    actions = TASK_SOLUTIONS.get(difficulty, [])
    
    total_reward = 0.0
    for idx, act_name in enumerate(actions, 1):
        # Step action
        print(f"Step {idx:02} | Executing: {act_name}")
        resp = requests.post(f"{server_url}/step", json={
            "action": act_name,
            "parameters": {"topic": "dbms"} if "notes" in act_name else {}
        })
        
        if resp.status_code != 200:
            print(f"Error step: {resp.text}")
            break
            
        step_data = resp.json()
        reward = step_data["reward"]
        total_reward += reward
        state = step_data["state"]
        done = step_data["done"]
        
        print(f"        | Reward: {reward:.2f} | Cumulative: {state['current_reward']:.4f}")
        
        if done:
            break
            
    print(f"\nTask Finished | Final Reward Sum: {total_reward:.4f} | State Reward: {state['current_reward']:.4f}")
    print(f"Steps: {state['completed_steps']} / {state['completed_steps'] + state['tasks_remaining']}")
    
    if state["current_reward"] >= 0.85:
        print("✅ SUCCESS: PERFECT SUBMISSION SCORE")
    else:
        print("⚠️ PARTIAL SCORE: TRY AGAIN")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_stdio("URL", default="http://localhost:8000", help="API URL (e.g. http://localhost:8000)")
    parser.add_argument("--url", default="http://localhost:8000", help="API URL")
    parser.add_argument("--task", default="hard", help="Task difficulty (easy, medium, hard)")
    args = parser.parse_args()
    
    # Change current working directory to the API URL if needed
    run_agent(args.url, args.task)
