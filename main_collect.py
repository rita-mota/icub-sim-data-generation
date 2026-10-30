from http import client
import os
import argparse
import subprocess
from icub_pybullet.pycub import pyCub
from core.data_collector import DataCollector
from core.robot_skills import RobotSkills
from datetime import datetime
import pybullet as p
import signal
import re
from pathlib import Path

# Import tasks
from tasks.task_1 import T1Stacking
from tasks.task_1_smooth import T1SmoothStacking
from tasks.task_2 import T2Moving
from tasks.task_2_smooth import T2SmoothMoving
from tasks.t2_new_smooth import T2NewSmoothMoving
from tasks.task_3 import T3Pushing
from tasks.task_3_simple import T3SimplePushing
from tasks.task_3_smooth import T3SmoothPushing
from tasks.task_4 import T4Opening
from tasks.task_4_smooth import T4SmoothOpening
from tasks.task_5 import T5Cleaning
from tasks.task_5_smooth import T5SmoothCleaning

#Import image decoder
#from recordings.decoder import extract_npz_to_images


class EpisodeTimeoutException(Exception):
    """Raised when an episode exceeds the maximum allowed simulation steps."""
    pass

def generate_llm_variations(base_prompt, task_name):
    """Load pre-generated prompts; BaseTask chooses one per episode.

    Variants such as t1_smooth and t2_new_smooth share their base task's
    recordings/tN/tN_prompts.txt. No LLM is called.
    """
    match = re.fullmatch(r"(t\d+)(?:_.*)?", task_name)
    if match is None:
        raise ValueError(f"Invalid task name: {task_name!r}")
    base_task = match.group(1)
    prompts_file = (
        Path(__file__).resolve().parent
        / "recordings" / base_task / f"{base_task}_prompts.txt"
    )
    try:
        lines = prompts_file.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError:
        print(f"[!] Prompt file missing: {prompts_file}. Using BASE_PROMPT.")
        return [base_prompt]

    # Remove list numbering and duplicate/blank lines without changing order.
    prompts = []
    seen = set()
    for line in lines:
        prompt = re.sub(r"^\d+[.)]\s*", "", line.strip()).strip()
        if prompt and prompt not in seen:
            prompts.append(prompt)
            seen.add(prompt)
    if not prompts:
        print(f"[!] Prompt file empty: {prompts_file}. Using BASE_PROMPT.")
        return [base_prompt]
    print(f"[*] Loaded {len(prompts)} prompt variations from {prompts_file}")
    return prompts

def get_task_class(task_name):
    registry = {
        "t1": T1Stacking,
        "t1_smooth": T1SmoothStacking,
        "t2": T2Moving,
        "t2_smooth": T2SmoothMoving,
        "t2_new_smooth": T2NewSmoothMoving,
        "t3": T3Pushing,
        "t3_simple": T3SimplePushing,
        "t3_smooth": T3SmoothPushing,
        "t4": T4Opening,
        "t4_smooth": T4SmoothOpening,
        "t5": T5Cleaning,
        "t5_smooth": T5SmoothCleaning
    }
    return registry[task_name]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["t1", "t1_smooth", "t2", "t2_smooth", "t2_new_smooth", "t3", "t3_simple", "t3_smooth", "t4", "t4_smooth", "t5", "t5_smooth"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--randomizer", type=bool, default=False)
    parser.add_argument("--debug", action="store_true", help="Keep window open, disable saving/rendering")
    args = parser.parse_args()

    TaskClass = get_task_class(args.task)

    # Load the task-specific prompt pool once; BaseTask samples each episode.
    variations = generate_llm_variations(TaskClass.BASE_PROMPT, args.task)

    # 2. Boot PyBullet EXACTLY ONCE outside the loop
    print(f"Booting environment using config: {TaskClass.CONFIG_PATH}")
    client = pyCub(config=TaskClass.CONFIG_PATH)
    skills = RobotSkills(client)
    original_update = client.update_simulation
    
    
    # --- THE FIX: Snapshot the pristine environment ---
    pristine_state_id = p.saveState(physicsClientId=client._client)
    
    # Record every default body ID (Robot, Table, etc.) so we know what belongs here
    num_bodies = p.getNumBodies(physicsClientId=client._client)
    pristine_bodies = set(p.getBodyUniqueId(i, physicsClientId=client._client) for i in range(num_bodies))

    # 3. Setup Session Folder
    task_dir = os.path.join("./recordings", args.task)
    session_name = f"{args.task}_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir = os.path.join(task_dir, session_name)
    
    os.makedirs(session_dir, exist_ok=True)
    print(f"\n[Session Setup] Created new session directory: {session_dir}")
    
    success_count = 0
    attempt_count = 0
    
    print(f"Goal: Collect {args.episodes} SUCCESSFUL episodes.")
    
    while success_count < args.episodes:
        attempt_count += 1
        print(f"\n--- Starting Attempt {attempt_count} (Successes: {success_count}/{args.episodes}) ---")
        
        # Save inside the session folder, numbered by attempt
        save_dir = os.path.join(session_dir, f"episode_{attempt_count:03d}")
        
        # Dynamically grab the correct object IDs depending on the task
        track_objs = [client.get_object_id(name) for name in TaskClass.OBJECT_NAMES]
            
        collector = DataCollector(
            client=client,
            camera_keys=[],
            save_dir=save_dir,
            hz=5.0,
            object_ids=track_objs
        )
        
        if args.debug:
            collector.save = lambda: print("[DEBUG] Data saving disabled.")
            
        # Define maximum allowed simulation steps (240 Hz * 45 seconds = 10,800 steps)
        MAX_EPISODE_STEPS = 240 * 60 * 5 
        
        # Capture the EXACT physics step count at the start of this specific episode
        start_steps = client.steps_done

        # def patched_update(sleep_duration=-1, _collector=collector):
        #     original_update(sleep_duration)
        #     _collector.step_collection()
            
        #     # Calculate the true number of physics steps elapsed in this episode
        #     steps_elapsed = client.steps_done - start_steps
            
        #     # Trigger the safeguard if we take too long in simulation time
        #     if steps_elapsed > MAX_EPISODE_STEPS:
        #         raise EpisodeTimeoutException(f"Safeguard Triggered: Episode exceeded {steps_elapsed} physics steps.")
        
        def patched_update(sleep_duration=-1, _collector=collector):
            steps_before = client.steps_done
            original_update(sleep_duration)
            
            if client.steps_done > steps_before:
                _collector.step_collection()
            
            steps_elapsed = client.steps_done - start_steps
            if steps_elapsed > MAX_EPISODE_STEPS:
                raise EpisodeTimeoutException(f"Safeguard Triggered: Episode exceeded {steps_elapsed} physics steps.")
                
        client.update_simulation = patched_update

       # Instantiate and run the task
        task = TaskClass(client, collector, skills, args.randomizer, prompt_variations=variations)
        
        if args.debug:
            task.save_metadata = lambda: None
            
        try:
            is_success = task.run()
        except EpisodeTimeoutException as e:
            # If the robot gets stuck, this block executes instantly
            print(f"\n[!] {e}")
            is_success = False
            # Ensure we mark the collector as failed so it doesn't save garbage data
            collector.success = False
            
            
        if is_success: 
            success_count += 1
            print(f"Attempt {attempt_count} SUCCESS! (Total Successes: {success_count}/{args.episodes})")
        else:
            print(f"Attempt {attempt_count} FAILED. Retrying...")
            
        # --- NEW: Hang the simulation window open for inspection ---
        if args.debug:
            print("\n[DEBUG MODE] Task complete. Keeping simulation window open...")
            print("Press Ctrl+C in terminal to exit.")
            
            # Stop the collector from trying to grab frames while we wait
            client.update_simulation = original_update 
            
            try:
                while True:
                    client.update_simulation()
            except KeyboardInterrupt:
                print("\nExiting debug mode...")
                import sys
                sys.exit(0)  # Instantly terminates the script so STEP 1 (Rendering) never runs
                
       # --- THE FIX: The Perfect Reset ---
        # 1. Snap the physics bodies back to their exact starting positions
        try:
            skills.release_grasp()
        except Exception:
            pass
        
        current_num_bodies = p.getNumBodies(physicsClientId=client._client)
        
        # Collect IDs into a static list first to prevent Index shifting errors during deletion
        body_ids = [p.getBodyUniqueId(i, physicsClientId=client._client) for i in range(current_num_bodies)]
        for body_id in body_ids:
            if body_id not in pristine_bodies:
                p.removeBody(body_id, physicsClientId=client._client)

        # 2. NOW safely snap the original physics bodies back to their pristine state
        p.restoreState(pristine_state_id, physicsClientId=client._client)
        
        # 2. Fix the PyBullet "Motor Memory" Bug
        # Read the newly restored joint positions and force the motors to stay there
        for i in range(p.getNumJoints(client.robot, physicsClientId=client._client)):
            restored_pos = p.getJointState(client.robot, i, physicsClientId=client._client)[0]
            p.setJointMotorControl2(
                bodyUniqueId=client.robot,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=restored_pos,
                targetVelocity=0,
                physicsClientId=client._client
            )
        
        # 3. Sweep the world and delete any ghost objects spawned during the attempt
        # current_num_bodies = p.getNumBodies(physicsClientId=client._client)
        # for i in range(current_num_bodies):
        #     body_id = p.getBodyUniqueId(i, physicsClientId=client._client)
        #     if body_id not in pristine_bodies:
        #         p.removeBody(body_id, physicsClientId=client._client)
        
        for joint in client.joints:
            joint.set_point = None
            joint.start_time = None
            joint.timeout = None

    # Disconnect safely only after the entire session is completely finished
    
    p.disconnect(physicsClientId=client._client)
    


if __name__ == "__main__":
    main()
