"""
Offline Renderer for iCub pyCub Recordings (Parallel Chunk Rendering)
=====================================================================
Replays saved proprio/object state back into PyBullet and renders all camera
views at full quality using multiprocessing and TinyRenderer.

Run headless:
    xvfb-run -a python render_recording.py --recording ./recordings/t1_20240506_123456
"""

from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import multiprocessing as mp
import json
from pathlib import Path

# ─── AUTO-RELAUNCH UNDER XVFB IF HEADLESS ────────────────────────────────────

def ensure_display() -> None:
    """If no display is available, relaunch this script under xvfb-run."""
    if not os.environ.get("DISPLAY"):
        print("[Renderer] No $DISPLAY found — relaunching under xvfb-run...")
        os.execvp("xvfb-run", ["xvfb-run", "-a", sys.executable] + sys.argv)

ensure_display()  # Must be called before any OpenGL / PyBullet import.


# ─── IMPORTS & PATCHES ───────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pybullet as p
import icub_pybullet.pycub
from icub_pybullet.pycub import pyCub
from core.data_collector import get_fixed_camera_matrices, get_pybullet_camera_matrices

# MONKEYPATCH: We completely bypass the Open3D GUI. We only need PyBullet's internal 
# camera renderer, so opening 20 Open3D windows will just crash the system.
class DummyVisualizer:
    def __init__(self, client):
        self.is_alive = True
        self.eye_windows = {}
    def render(self): 
        pass

icub_pybullet.pycub.Visualizer = DummyVisualizer


# ─── CONFIGURATION ───────────────────────────────────────────────────────────

CONFIG_PATH   = "/home/ritam/thesis/sim_data_collection/configs/t2.yaml"
CAMERA_KEYS   = ["r_eye", "l_eye", "fixed_camera"]
RENDER_WIDTH  = 320
RENDER_HEIGHT = 240
OBJECT_NAMES  = ["bottle", "computer"]   
CORES         = 10         

LIGHT_DIRECTION = [1, 1, 1]
LIGHT_COLOR     = [1.0, 1.0, 1.0]
LIGHT_DISTANCE  = 2.0


# ─── MULTIPROCESSING LOCK ────────────────────────────────────────────────────

# This lock ensures that 20 processes don't try to write/overwrite the exact 
# same .urdf and .obj files simultaneously during pyCub initialization.
init_lock = None

def worker_init(lock):
    global init_lock
    init_lock = lock


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def build_eye_link_ids(client) -> dict:
    ids = {}
    for link in client.links:
        if "l_eye" in link.name:
            ids["l_eye"] = link.robot_link_id
        elif "r_eye" in link.name:
            ids["r_eye"] = link.robot_link_id
    return ids


def render_frame(client_id: int, robot_id: int, eye_link_ids: dict,
                 fixed_view: tuple, fixed_proj: tuple,
                 cam_key: str) -> np.ndarray | None:
    """Renders a single camera view and returns an (H, W, 3) uint8 array."""
    if cam_key == "fixed_camera":
        view_mat, proj_mat = fixed_view, fixed_proj
    elif cam_key in eye_link_ids:
        view_mat, proj_mat = get_pybullet_camera_matrices(
            client_id, robot_id, eye_link_ids[cam_key])
    else:
        return None

    try:
        # CHANGED: Using ER_TINY_RENDERER to avoid OpenGL/Xvfb deadlocks
        _, _, rgba, _, _ = p.getCameraImage(
            width=RENDER_WIDTH,
            height=RENDER_HEIGHT,
            viewMatrix=view_mat,
            projectionMatrix=proj_mat,
            renderer=p.ER_TINY_RENDERER, 
            shadow=1,
            lightDirection=LIGHT_DIRECTION,
            lightColor=LIGHT_COLOR,
            lightDistance=LIGHT_DISTANCE,
            physicsClientId=client_id
        )
        return np.reshape(rgba, (RENDER_HEIGHT, RENDER_WIDTH, 4))[:, :, :3].astype(np.uint8)
    except Exception as e:
        return None


# ─── WORKER FUNCTION ─────────────────────────────────────────────────────────

def render_chunk(core_id: int, chunk_start: int, chunk_end: int, 
                 proprio: np.ndarray, obj_poses: np.ndarray, 
                 config_path: str, object_names: list, cam_info: dict) -> tuple:
    
    global init_lock

    # Safely boot pyCub one at a time to prevent File I/O race conditions
    with init_lock:
        client = pyCub(config=config_path)
        
    robot_id  = client.robot
    client_id = client._client

    all_joint_ids = [j.robot_joint_id for j in client.joints]
    eye_link_ids  = build_eye_link_ids(client)
    object_ids    = [client.get_object_id(name) for name in object_names]
    
    # for obj_name, oid in zip(object_names, object_ids):
    #     if obj_name == "computer":
    #         # Link 0 is the screenLink. We force its visual orientation to 0
    #         # strictly inside PyBullet's memory, bypassing the URDF entirely.
    #         p.changeVisualShape(oid, 0, localVisualFrameOrientation=[0, 0, 0, 1], physicsClientId=client_id)
                

    # Replace get_fixed_camera_matrices() with this:
    fixed_view, fixed_proj = get_fixed_camera_matrices(
        cam_pos=cam_info["fixed_camera_pos"],
        cam_target=cam_info["fixed_camera_target"],
        cam_up=cam_info["fixed_camera_up"]
    )
    chunk_buffer = {k: [] for k in CAMERA_KEYS}
    total_chunk_frames = chunk_end - chunk_start

    for frame_i in range(chunk_start, chunk_end):
        for joint_id, pos in zip(all_joint_ids, proprio[frame_i]):
            p.resetJointState(robot_id, joint_id, float(pos), physicsClientId=client_id)

        for i_obj, oid in enumerate(object_ids):
            base = i_obj * 7
            obj_pos = obj_poses[frame_i, base:base + 3]
            obj_orn = obj_poses[frame_i, base + 3:base + 7]
            
            # # --- THE FIX: Visual Axis Correction for the Computer ---
            # if object_names[i_obj] == "computer":
            #     # If it's rotated 90 degrees the wrong way, we apply an inverse rotation
            #     # just for the camera. Change -1.5708 to 1.5708 if it rotates the wrong way.
            #     inv_rot = p.getQuaternionFromEuler([0, 0, 1.5708]) 
            #     _, obj_orn = p.multiplyTransforms([0, 0, 0], obj_orn.tolist(), [0, 0, 0], inv_rot)
            # else:
            #     obj_orn = obj_orn.tolist()

            p.resetBasePositionAndOrientation(
                oid, obj_pos.tolist(), obj_orn, physicsClientId=client_id)

        # --- THE FIX: Delete or comment out the physics step ---
        # p.stepSimulation(physicsClientId=client_id)

        for cam_key in CAMERA_KEYS:
            img = render_frame(client_id, robot_id, eye_link_ids, fixed_view, fixed_proj, cam_key)
            if img is not None:
                chunk_buffer[cam_key].append(img)
                
        # --- VISUAL PROGRESS BAR ---
        frames_processed = frame_i - chunk_start + 1
        if frames_processed % 5 == 0 or frames_processed == total_chunk_frames:  
            percent = frames_processed / total_chunk_frames
            bar_length = 20
            filled = int(bar_length * percent)
            bar = '#' * filled + '-' * (bar_length - filled)
            print(f"[Core {core_id:>2}] {bar} ({int(percent * 100):>3}%)")
    
    try:
        p.resetSimulation(physicsClientId=client_id)
    except Exception:
        pass 
        
    del client

    # 2. Save the heavy data to the hard drive
    temp_filename = f"temp_chunk_{core_id}_{chunk_start}.npz"
    np.savez_compressed(temp_filename, **chunk_buffer)
    
    # 3. ONLY return the start index and the file name string!
    # DO NOT return chunk_buffer here.
    return chunk_start, temp_filename

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main(recording_dir: str) -> None:# --- THE SKIP FIX ---
    # Check if all expected camera videos already exist in this folder
    already_rendered = all(
        os.path.exists(os.path.join(recording_dir, f"{cam}_video.npz")) 
        for cam in CAMERA_KEYS
    )
    
    if already_rendered:
        print(f"[Renderer] Skipping {recording_dir} — already rendered!")
        return
    # --------------------
    required = ["timestamps.npy", "proprio.npy", "obj_poses.npy"]
    for fname in required:
        path = os.path.join(recording_dir, fname)
        if not os.path.exists(path):
            print(f"[Renderer] Missing required file: {path}")
            sys.exit(1)

    timestamps = np.load(os.path.join(recording_dir, "timestamps.npy"))
    proprio    = np.load(os.path.join(recording_dir, "proprio.npy"))
    obj_poses  = np.load(os.path.join(recording_dir, "obj_poses.npy"))
    n_frames   = len(timestamps)

    print(f"[Renderer] Recording: {recording_dir}")
    print(f"[Renderer] Frames   : {n_frames}")
    print(f"[Renderer] Cameras  : {CAMERA_KEYS}")
    print(f"[Renderer] Using {CORES} CPU cores for parallel rendering with TinyRenderer...")
    
    meta_path = os.path.join(recording_dir, "metadata.json")
    with open(meta_path, "r") as f:
        meta_data = json.load(f)
        
    config_path = meta_data["config_path"]
    object_names = meta_data["object_names"]

    # --- ADD THIS: Load the unique camera pose for this episode ---
    cam_info_path = os.path.join(recording_dir, "camera_info.json")
    with open(cam_info_path, "r") as f:
        cam_info = json.load(f)

    chunk_size = int(np.ceil(n_frames / CORES))
    chunks = [(i, min(i + chunk_size, n_frames)) for i in range(0, n_frames, chunk_size)]
    
    # --- UPDATE THIS: Pass cam_info to the worker_args ---
    worker_args = [(idx, start, end, proprio, obj_poses, config_path, object_names, cam_info) 
                   for idx, (start, end) in enumerate(chunks)]

    # Create lock to prevent file I/O crashes during initialization
    ctx = mp.get_context('spawn')
    manager = ctx.Manager()
    lock = manager.Lock()  # <--- Use ctx.Lock() instead of mp.Lock()
    
    with ctx.Pool(processes=CORES, initializer=worker_init, initargs=(lock,)) as pool:
        results = pool.starmap(render_chunk, worker_args)
    
    print("\n[Renderer] Reassembling chunks...", flush=True)
    results.sort(key=lambda x: x[0])
    
    print("[Renderer] Saving videos...")
    
    # Process one camera completely from start to finish to prevent RAM inflation
    for cam_key in CAMERA_KEYS:
        cam_frames = []
        
        # Load only this specific camera's data from all chunks
        for _, temp_filename in results:
            chunk_data = np.load(temp_filename)
            cam_frames.extend(chunk_data[cam_key])
            
        if not cam_frames:
            continue
            
        # Stack, compress, and save
        video_block = np.stack(cam_frames)
        save_path = os.path.join(recording_dir, f"{cam_key}_video.npz")
        np.savez_compressed(save_path, video=video_block)
        print(f"  -> {cam_key}: {len(cam_frames)} frames saved  {video_block.shape}", flush=True)
        
        # --- THE FIX ---
        # Explicitly delete these massive arrays from RAM before loading the next camera!
        del cam_frames
        del video_block

    # Now that all cameras are saved, we can safely delete the temporary chunk files
    for _, temp_filename in results:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    print(f"[Renderer] Done. Videos written to: {recording_dir}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline renderer for pyCub recordings.")
    parser.add_argument(
        "--recording", "-r",
        required=True,
        help="Path to the recording directory"
    )
    args = parser.parse_args()
    main(args.recording)