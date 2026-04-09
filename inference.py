import os
import requests
import time
import sys
from typing import List, Dict, Any

from agent.baseline_agent import BaselineAgent

def run_agent(server_url: str = "http://localhost:8000", difficulty: str = "easy"):
    # Clear wait
    time.sleep(1)
    
    # Force use of BaselineAgent with LLM if possible
    # Note: Environment logic inside BaselineAgent needs to be aware of the server_url
    # or just use the direct env call if running on same process.
    # However, the validator usually expects us to talk to the server.
    
    # 1. [START] block
    print(f"[START] task={difficulty}", flush=True)
    sys.stdout.flush()

    # We'll use a modified version of the agent's run loop to communicate via REST
    # following the exact requirements of the validator.
    
    # Connect
    try:
        resp = requests.post(f"{server_url}/reset", json={"difficulty": difficulty}, timeout=15)
        resp.raise_for_status()
    except Exception:
        print(f"[END] task={difficulty} score=0.0 steps=0", flush=True)
        return
        
    data = resp.json()
    state_dict = data["state"]
    
    # Initialize Agent (it will pick up API_KEY and API_BASE_URL from env)
    agent = BaselineAgent(use_llm=True, verbose=False)
    
    completed_steps = 0
    max_steps = 20

    for idx in range(1, max_steps + 1):
        # Let agent decide based on state_dict
        action = agent.select_action(state_dict)
        
        try:
            resp = requests.post(f"{server_url}/step", json={
                "action": action.action_type.value,
                "parameters": action.parameters
            }, timeout=15)
            resp.raise_for_status()
        except Exception:
            break
            
        step_data = resp.json()
        reward = step_data["reward"]
        state_dict = step_data["state"]
        done = step_data["done"]
        completed_steps = len(state_dict['completed_steps'])
        
        # 2. [STEP] block
        print(f"[STEP] step={idx} reward={reward:.2f}", flush=True)
        sys.stdout.flush()
        
        if done:
            break
            
    # 3. [END] block
    print(f"[END] task={difficulty} score={state_dict['current_reward']:.2f} steps={completed_steps}", flush=True)
    sys.stdout.flush()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", "--URL", default="http://localhost:8000")
    parser.add_argument("--task", default="hard")
    args = parser.parse_args()
    run_agent(args.url, args.task)
