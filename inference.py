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

    # Connect to server
    try:
        resp = requests.post(f"{server_url}/reset", json={"difficulty": difficulty}, timeout=15)
        resp.raise_for_status()
    except Exception:
        # 3. [END] block — fallback
        print(f"[END] task={difficulty} score=0.01 steps=0", flush=True)
        return
        
    try:
        data = resp.json()
        state_dict = data["state"]
        
        # Initialize Agent
        agent = BaselineAgent(use_llm=True, verbose=False)
        
        completed_steps = 0
        max_steps = 20
        final_reward = 0.01
        is_done = False

        for idx in range(1, max_steps + 1):
            try:
                action = agent.select_action(state_dict)
                resp = requests.post(f"{server_url}/step", json={
                    "action": action.action.value,
                    "parameters": action.parameters
                }, timeout=15)
                resp.raise_for_status()
                
                step_data = resp.json()
                reward = step_data["reward"]
                state_dict = step_data["state"]
                is_done = step_data["done"]
                completed_steps = len(state_dict.get('completed_steps', []))
                
                # Track reward — clamp to (0.01, 0.99)
                final_reward = max(0.01, min(0.99, reward))
                
                # 2. [STEP] block — reward always strictly between 0 and 1
                print(f"[STEP] step={idx} reward={final_reward:.4f} done={is_done}", flush=True)
                
                if is_done:
                    break
            except Exception:
                break
        
        # Ensure final score from state is also clamped
        state_reward = state_dict.get('current_reward', 0.01)
        final_score = max(0.01, min(0.99, state_reward))
                
        # 3. [END] block — score is NEVER 0.0 or 1.0
        print(f"[END] task={difficulty} score={final_score:.4f} steps={completed_steps}", flush=True)
    except Exception:
        print(f"[END] task={difficulty} score=0.01 steps=0", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", "--URL", default="http://localhost:8000")
    parser.add_argument("--task", default="all")
    args = parser.parse_args()
    
    if args.task == "all":
        for diff in ["easy", "medium", "hard"]:
            run_agent(args.url, diff)
    else:
        run_agent(args.url, args.task)
