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
                 cam_key: str, width: int, height: int,
                 light_dir: list, light_color: list, light_dist: float) -> np.ndarray | None:
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
            width=width,
            height=height,
            viewMatrix=view_mat,
            projectionMatrix=proj_mat,
            renderer=p.ER_TINY_RENDERER, 
            shadow=1,
            lightDirection=light_dir,
            lightColor=light_color,
            lightDistance=light_dist,
            physicsClientId=client_id
        )
        return np.reshape(rgba, (height, width, 4))[:, :, :3].astype(np.uint8)
    except Exception as e:
        return None


# ─── WORKER FUNCTION ─────────────────────────────────────────────────────────

def render_chunk(core_id: int, chunk_start: int, chunk_end: int, 
                 proprio: np.ndarray, obj_poses: np.ndarray, 
                 config_path: str, object_names: list, cam_info: dict,
                 camera_keys: list, width: int, height: int,
                 light_dir: list, light_color: list, light_dist: float,
                 spill_log: dict) -> tuple:
    
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
    chunk_buffer = {k: [] for k in camera_keys}
    total_chunk_frames = chunk_end - chunk_start
    
    # --- THE FIX: Spawn Spill Particles for this specific Chunk ---
    spill_particles = {}
    if spill_log:
        for name, data in spill_log.items():
            # Only spawn it if it hasn't been deleted yet BEFORE this chunk starts
            if data["deleted_step"] == -1 or data["deleted_step"] > chunk_start:
                vis = p.createVisualShape(
                    p.GEOM_CYLINDER, radius=data["radius"], length=data["thickness"], 
                    rgbaColor=data["rgba"], physicsClientId=client_id)
                bid = p.createMultiBody(
                    baseMass=0, baseVisualShapeIndex=vis, basePosition=data["pos"], 
                    physicsClientId=client_id)
                spill_particles[name] = {"body_id": bid, "deleted_step": data["deleted_step"]}

    for frame_i in range(chunk_start, chunk_end):
        for joint_id, pos in zip(all_joint_ids, proprio[frame_i]):
            p.resetJointState(robot_id, joint_id, float(pos), physicsClientId=client_id)

        for i_obj, oid in enumerate(object_ids):
            base = i_obj * 7
            obj_pos = obj_poses[frame_i, base:base + 3]
            obj_orn = obj_poses[frame_i, base + 3:base + 7]
    

            p.resetBasePositionAndOrientation(
                oid, obj_pos.tolist(), obj_orn, physicsClientId=client_id)
        if spill_log:
            for name, data in list(spill_particles.items()):
                if data["deleted_step"] != -1 and frame_i >= data["deleted_step"]:
                    p.removeBody(data["body_id"], physicsClientId=client_id)
                    del spill_particles[name] # Stop tracking it

        # --- THE FIX: Delete or comment out the physics step ---
        # p.stepSimulation(physicsClientId=client_id)

        for cam_key in camera_keys:
            # --- THE FIX: Delete particles at the exact frame they were cleaned ---
            img = render_frame(client_id, robot_id, eye_link_ids, fixed_view, fixed_proj, 
                               cam_key, width, height, light_dir, light_color, light_dist)
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

# ─── ORCHESTRATOR CLASS ────────────────────────────────────────────────────────────────────

class EpisodeRenderer:
    """Manages the parsing, parallelization, and saving of episode renders."""
    
    def __init__(self, recording_dir: Path, cores: int = 6, width: int = 320, height: int = 240,
                 light_dir: list = [1, 1, 1], light_color: list = [1, 1, 1], light_dist: float = 2.0):
        
        self.recording_dir = Path(recording_dir)
        if cores < 1:
            raise ValueError("cores must be at least 1")
        self.cores = cores
        self.width = width
        self.height = height
        
        self.light_dir = light_dir
        self.light_color = light_color
        self.light_dist = light_dist
        
        # Load dynamic metadata
        self.metadata = self._load_json("metadata.json")
        self.cam_info = self._load_json("camera_info.json")
        self.camera_keys = ["r_eye", "l_eye", "fixed_camera"] # Or pull this from metadata if saved

    def _load_json(self, filename: str):
        filepath = self.recording_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Missing required file: {filepath}")
        with open(filepath, "r") as f:
            return json.load(f)

    def run(self):
        print(f"[Renderer] Processing: {self.recording_dir.name}")
        
        timestamps = np.load(self.recording_dir / "timestamps.npy")
        proprio    = np.load(self.recording_dir / "proprio.npy")
        obj_poses  = np.load(self.recording_dir / "obj_poses.npy")
        n_frames   = len(timestamps)
        if n_frames == 0:
            raise ValueError("Cannot render an episode with no frames")

        # Both original and smooth T5 episodes reconstruct the recorded spill.
        is_cleaning_task = self.metadata["task_name"] in {
            "T5Cleaning", "T5SmoothCleaning",
        }
        spill_file = self.recording_dir / "spill_data.json"
        if spill_file.exists():
            spill_log = self._load_json("spill_data.json")
        elif is_cleaning_task:
            raise FileNotFoundError(
                f"Missing T5 spill log: {spill_file}. "
                "Cannot reconstruct the spill without this file."
            )
        else:
            spill_log = None
        
        chunk_size = int(np.ceil(n_frames / self.cores))
        chunks = [(i, min(i + chunk_size, n_frames)) for i in range(0, n_frames, chunk_size)]
        worker_count = min(self.cores, len(chunks))
        print(
            f"[Renderer] Rendering {n_frames} frames in {len(chunks)} chunks "
            f"with {worker_count} parallel workers.", flush=True,
        )
        
        worker_args = [
            (idx, start, end, proprio, obj_poses, self.metadata["config_path"], 
             self.metadata["object_names"], self.cam_info, self.camera_keys, self.width, self.height, 
             self.light_dir, self.light_color, self.light_dist, spill_log) 
            for idx, (start, end) in enumerate(chunks)
        ]

        ctx = mp.get_context('spawn')
        manager = ctx.Manager()
        lock = manager.Lock()
        
        with ctx.Pool(processes=worker_count, initializer=worker_init, initargs=(lock,)) as pool:
            results = pool.starmap(render_chunk, worker_args, chunksize=1)
        
        self._compile_videos(results)

    def _compile_videos(self, results):
        print("\n[Renderer] Reassembling chunks...", flush=True)
        results.sort(key=lambda x: x[0])
        
        for cam_key in self.camera_keys:
            cam_frames = []
            for _, temp_filename in results:
                chunk_data = np.load(temp_filename)
                if cam_key in chunk_data:
                    cam_frames.extend(chunk_data[cam_key])
                
            if cam_frames:
                video_block = np.stack(cam_frames)
                save_path = self.recording_dir / f"{cam_key}_video.npz"
                np.savez_compressed(save_path, video=video_block)
                print(f"  -> {cam_key}: {len(cam_frames)} frames saved")
                
                del cam_frames
                del video_block

        # Cleanup temp files
        for _, temp_filename in results:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline renderer for pyCub recordings.")
    parser.add_argument("--recording", "-r", required=True, type=Path, help="Path to the recording directory")
    parser.add_argument("--cores", "-c", type=int, default=15, help="Number of CPU cores to use")
    parser.add_argument("--width", type=int, default=320, help="Render width")
    parser.add_argument("--height", type=int, default=240, help="Render height")
    args = parser.parse_args()
    
    renderer = EpisodeRenderer(
        recording_dir=args.recording, 
        cores=args.cores,
        width=args.width,
        height=args.height
    )
    renderer.run()
