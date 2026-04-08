"""
Mandatory inference script for the Tour Planner environment.
Strictly follows the [START], [STEP], and [END] stdout format.
"""

import os
import sys
import argparse
from typing import List, Dict, Any
from openai import OpenAI

# Robust imports for both Local testing and Grading server
try:
    # 1. Grader: Will work if they `pip install -e .` on your repo
    from tour_planner_env import TourPlannerEnv, TourAction
except ImportError:
    import sys
    import os
    
    # 2. Local fallback: If you run it from C:\Scaler on your laptop
    envs_path = os.path.join(os.getcwd(), "OpenEnv", "envs")
    openenv_src = os.path.join(os.getcwd(), "OpenEnv", "src")
    if os.path.exists(envs_path):
        sys.path.append(envs_path)
        sys.path.append(openenv_src)
        from tour_planner_env import TourPlannerEnv, TourAction
        
    # 3. Direct execution fallback: If inference.py is run inside the repo itself
    else:
        sys.path.append(os.path.join(os.getcwd(), "..", "..", "src"))
        from client import TourPlannerEnv
        from models import TourAction

def run_episode(task_id: str, city_name: str):
    # Environment Variables for LLM
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct:fastest")
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    # Debug info to stderr
    print(f"DEBUG: Using API_BASE_URL={API_BASE_URL}", file=sys.stderr)
    print(f"DEBUG: Using MODEL_NAME={MODEL_NAME}", file=sys.stderr)
    
    # Optional - if you use from_docker_image():
    LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

    # Use API_KEY (injected by hackathon grader), or fallback to HF_TOKEN / OPENAI_API_KEY
    api_key = os.getenv("API_KEY") or HF_TOKEN or os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY or HF_TOKEN must be set.", file=sys.stderr)
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=api_key)

    # Initialize environment
    try:
        if LOCAL_IMAGE_NAME:
            # Use local docker image if specified
            _client = TourPlannerEnv.from_docker_image(LOCAL_IMAGE_NAME)
        else:
            # Fallback to remote or local server
            ENV_BASE_URL = os.getenv("SPACE_URL", "https://thefated-tour-planner-env.hf.space")
            _client = TourPlannerEnv(base_url=ENV_BASE_URL)
            
        # Use sync wrapper for easier scripting
        env = _client.sync()
    except Exception as e:
        print(f"Error connecting to environment: {e}", file=sys.stderr)
        return

    # [START] task=<task_name> env=<benchmark> model=<model_name>
    print(f"[START] task={task_id} env=tour_planner_benchmark model={MODEL_NAME}", flush=True)

    # Reset
    obs_result = env.reset(task_id=task_id, city_name=city_name)
    obs = obs_result.observation
    
    done = False
    step_count = 0
    max_steps = 20
    rewards: List[float] = []

    while not done and step_count < max_steps:
        # Construct prompt
        prompt = f"""
        You are a tour planning agent. Your task is {task_id} in {city_name}.
        Current Observation:
        - Budget: {obs.remaining_budget}
        - Current Day: {obs.current_day}/{obs.current_day + obs.days_left}
        - Location: {obs.current_location}
        - Itinerary: {obs.current_itinerary}
        - Available Places: {obs.available_place_ids}
        
        Choose one action from:
        1. add_place (provide place_id)
        2. rest (advance to next day)
        3. finalize_plan (complete the episode)
        
        Respond ONLY with a JSON object like: {{"type": "add_place", "place_id": "..."}} or {{"type": "rest"}} or {{"type": "finalize_plan"}}
        """

        try:
            # Get action from LLM
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}]
            )
            
            import json
            action_dict = json.loads(response.choices[0].message.content)
            action = TourAction(**action_dict)

            # Step
            step_result = env.step(action)
            obs = step_result.observation
            done = step_result.done
            reward = step_result.reward
            
            # [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
            action_str = action.model_dump_json().replace('\n', '')
            rewards.append(reward)
            print(f"[STEP] step={step_count + 1} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
            
        except Exception as e:
            print(f"Error during step {step_count}: {e}", file=sys.stderr)
            break
            
        step_count += 1

    # End
    try:
        # Grade the episode
        # env.state is a method in the OpenEnv client
        final_state = env.state() if callable(env.state) else env.state
        from tour_planner_env.server.grader import TourGrader
        grader = TourGrader()
        
        # Robust dictionary conversion for Pydantic V1/V2 compatibility
        state_dict = final_state.model_dump() if hasattr(final_state, "model_dump") else final_state.dict()
        report = grader.grade(state_dict)
        
        # [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
        success_threshold = 0.70  # General threshold, actual depends on task
        success = report.final_score >= success_threshold
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={str(success).lower()} steps={step_count} score={report.final_score:.3f} rewards={rewards_str}", flush=True)
        
        # Human-readable summary to stderr (doesn't interfere with scoring)
        print(report, file=sys.stderr)
        
    except Exception as e:
        print(f"Error finalizing episode: {e}", file=sys.stderr)
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all")
    parser.add_argument("--city", default="paris")
    args = parser.parse_args()

    # Support TASK_NAME env var injected by hackathon grader
    task_override = os.getenv("TASK_NAME")
    if task_override:
        args.task = task_override

    # Run all 3 tasks in sequence — required for "3 tasks with graders" check
    ALL_TASKS = [
        ("task_1_easy",   "paris"),
        ("task_2_medium", "tokyo"),
        ("task_3_hard",   "mumbai"),
    ]

    if args.task == "all":
        for task_id, city in ALL_TASKS:
            run_episode(task_id, city)
    else:
        run_episode(args.task, args.city)
