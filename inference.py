import os
import requests
import time
import sys
from typing import List, Dict, Any

from agent.baseline_agent import BaselineAgent

def run_agent(server_url: str = "http://localhost:8000", difficulty: str = "easy"):
    # Clear wait
    time.sleep(1)
    
    # 1. [START] block
    print(f"[START] task={difficulty}", flush=True)
    sys.stdout.flush()

    # Connect to server
    try:
        resp = requests.post(f"{server_url}/reset", json={"difficulty": difficulty}, timeout=15)
        resp.raise_for_status()
    except Exception:
        # NEVER print 0.0 — use 0.01 as minimum score
        print(f"[END] task={difficulty} score=0.01 steps=0", flush=True)
        return
        
    data = resp.json()
    state_dict = data["state"]
    
    # Initialize Agent (it will pick up API_KEY and API_BASE_URL from env)
    agent = BaselineAgent(use_llm=True, verbose=False)
    
    completed_steps = 0
    max_steps = 20
    final_reward = 0.01  # Never 0.0

    for idx in range(1, max_steps + 1):
        # Let agent decide based on state_dict
        try:
            action = agent.select_action(state_dict)
        except Exception as e:
            print(f"ERROR: Agent select_action failed: {e}", flush=True)
            break
        
        try:
            resp = requests.post(f"{server_url}/step", json={
                "action": action.action.value,
                "parameters": action.parameters
            }, timeout=15)
            resp.raise_for_status()
        except Exception:
            break
            
        step_data = resp.json()
        reward = step_data["reward"]
        state_dict = step_data["state"]
        done = step_data["done"]
        completed_steps = len(state_dict.get('completed_steps', []))
        
        # Track reward — clamp to (0.01, 0.99)
        final_reward = max(0.01, min(0.99, reward))
        
        # 2. [STEP] block — reward always strictly between 0 and 1
        print(f"[STEP] step={idx} reward={final_reward:.4f}", flush=True)
        sys.stdout.flush()
        
        if done:
            break
    
    # Ensure final score from state is also clamped
    state_reward = state_dict.get('current_reward', 0.01)
    final_score = max(0.01, min(0.99, state_reward))
            
    # 3. [END] block — score is NEVER 0.0 or 1.0
    print(f"[END] task={difficulty} score={final_score:.4f} steps={completed_steps}", flush=True)
    sys.stdout.flush()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", "--URL", default="http://localhost:8000")
    parser.add_argument("--task", default="hard")
    args = parser.parse_args()
    run_agent(args.url, args.task)
