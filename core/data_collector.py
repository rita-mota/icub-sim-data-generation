"""
Data Collector for iCub pyCub Simulation
=========================================
Uses RAM Buffering to hold images during the simulation to prevent lag.
Uses a background render thread so getCameraImage() never blocks the sim loop.
Saves all data instantly at the end of the run as compressed .npz files.

Key changes vs original:
  - getCameraImage() calls are offloaded to a dedicated worker thread via a
    queue, so the physics loop is never stalled waiting for a render.
  - View/projection matrices are computed on the sim thread (cheap math only)
    and posted to the queue as a render job alongside the already-collected
    state snapshot.
  - Fixed-camera matrices are cached once at init — they never change.
  - Renderer is auto-selected: OpenGL when a display is available, TinyRenderer
    for headless environments, with an override option.
  - stop() drains the render queue before returning so save() always has the
    complete frame set.
"""

from __future__ import annotations

import os
import queue
import threading
import numpy as np
import pybullet as p
import icub_pybullet.visualizer_plus_other_camera
from typing import Optional, List
import cv2
import json

FINGER_JOINT_NAMES = [
    "r_hand_thumb_2_joint", "r_hand_thumb_3_joint",
    "r_hand_index_2_joint", "r_hand_index_3_joint",
    "r_hand_middle_2_joint", "r_hand_middle_3_joint",
    "r_hand_ring_2_joint",   "r_hand_ring_3_joint",
    "r_hand_little_2_joint", "r_hand_little_3_joint",
]

# ─── MATH HELPERS FOR SYNTHETIC CAMERAS ──────────────────────────────────────

def get_pybullet_camera_matrices(client_id: int, robot_id: int, link_id: int,
                                 fov: float = 60, aspect: float = 1.33,
                                 near: float = 0.02, far: float = 10.0):
    """Computes view and projection matrices matching the PyCub visualizer.
    Only matrix math — safe to call on the sim thread."""
    link_state = p.getLinkState(robot_id, link_id, computeForwardKinematics=True,
                                physicsClientId=client_id)
    pos = np.array(link_state[0])
    ori = link_state[1]

    rot_matrix  = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
    forward_vec = rot_matrix.dot(np.array([0, 0, 1]))
    forward_vec = forward_vec / np.linalg.norm(forward_vec)

    safe_pos   = pos + (forward_vec * 0.05)
    target_pos = safe_pos + forward_vec
    up_vec     = np.array([0, 0, 1])

    view_matrix = p.computeViewMatrix(cameraEyePosition=safe_pos,
                                      cameraTargetPosition=target_pos,
                                      cameraUpVector=up_vec)
    proj_matrix = p.computeProjectionMatrixFOV(fov=fov, aspect=aspect,
                                               nearVal=near, farVal=far)
    return view_matrix, proj_matrix

def get_fixed_camera_matrices(fov: float = 60, aspect: float = 1.0,
                              near: float = 0.01, far: float = 10.0,
                              cam_pos: List[float] = [],
                              cam_target: List[float] = [],
                              cam_up: List[float] = []):
    """Computes static matrices for the fixed table camera."""
    # cam_pos    = [-0.240, 0.098, 0.315]
    # cam_target = [-0.35,  0.1,  -0.1]
    # cam_up     = [0, 0, 1]

    view_matrix = p.computeViewMatrix(cameraEyePosition=cam_pos,
                                      cameraTargetPosition=cam_target,
                                      cameraUpVector=cam_up)
    proj_matrix = p.computeProjectionMatrixFOV(fov=fov, aspect=aspect,
                                               nearVal=near, farVal=far)
    return view_matrix, proj_matrix

def add_ground_shadow(img: np.ndarray, shadow_ellipses: list) -> np.ndarray:
    """
    shadow_ellipses: list of (cx, cy, rx, ry) in pixel coords
    Call get_shadow_ellipses() each frame to compute these from physics state.
    """
    overlay = img.copy()
    for (cx, cy, rx, ry) in shadow_ellipses:
        cv2.ellipse(overlay, (int(cx), int(cy)), (int(rx), int(ry)),
                    0, 0, 360, (0, 0, 0), -1)
    return cv2.addWeighted(img, 0.75, overlay, 0.25, 0)

def get_shadow_ellipses(object_ids, view_mat, proj_mat, 
                        img_w, img_h, client_id):
    """Projects each object's base position onto the image plane."""
    V = np.array(view_mat).reshape(4, 4).T
    P = np.array(proj_mat).reshape(4, 4).T
    ellipses = []
    for obj_id in object_ids:
        pos, _ = p.getBasePositionAndOrientation(obj_id, physicsClientId=client_id)
        # Drop to table level for shadow position
        shadow_pos = np.array([pos[0], pos[1], pos[2] - 0.03, 1.0])
        clip = P @ V @ shadow_pos
        ndc  = clip[:2] / clip[3]
        cx   = int((ndc[0] * 0.5 + 0.5) * img_w)
        cy   = int((1 - (ndc[1] * 0.5 + 0.5)) * img_h)
        ellipses.append((cx, cy, 12, 5))  # tune rx/ry to object size
    return ellipses

# ─── MAIN DATA COLLECTOR CLASS ───────────────────────────────────────────────

class DataCollector:
    def __init__(self, client, save_dir="./recordings", hz=5.0,
                 camera_keys=None, renderer=None,
                 render_width=320, render_height=240, object_ids = None):
        """
        Parameters
        ----------
        client       : pyCub simulation client
        save_dir     : where to write recordings
        hz           : capture frequency (relative to 240 Hz sim)
        camera_keys  : list of camera names to capture
        renderer     : p.ER_TINY_RENDERER | p.ER_BULLET_HARDWARE_OPENGL | None
                       None = auto-select based on DISPLAY env var
        render_width : image width  (reduce for extra speed, e.g. 160)
        render_height: image height (reduce for extra speed, e.g. 120)
        """
        self.client       = client
        self.save_dir     = save_dir
        self.hz           = hz
        self.camera_keys  = camera_keys #or ["r_eye", "l_eye", "fixed_camera"]
        self.render_w     = render_width
        self.render_h     = render_height
        self.object_ids   = object_ids or []

        # Auto-select renderer if not forced
        if renderer is not None:
            self._renderer = renderer
        else:
            self._renderer = (p.ER_BULLET_HARDWARE_OPENGL
                              if os.environ.get("DISPLAY")
                              else p.ER_TINY_RENDERER)

        # ── RAM Buffers (written by the render worker thread) ──────────────
        self._timestamps:    List[float]      = []
        self._proprio:       List[np.ndarray] = []
        self._ee_poses:      List[np.ndarray] = []
        self._ee_deltas:     List[np.ndarray] = []
        self._obj_poses:     List[np.ndarray] = []
        self._image_buffer:  dict             = {k: [] for k in self.camera_keys}
        self._buffer_lock    = threading.Lock()  # protects image_buffer appends

        # ── Precomputed PyBullet IDs ───────────────────────────────────────
        self._robot_id:         int        = -1
        self._client_id:        int        = -1
        self._ee_link_id:       int        = -1
        self._all_joint_ids:    List[int]  = []
        self._joint_names:      List[str]  = []
        self._finger_joint_ids: List[int]  = []
        self._finger_lo:        np.ndarray = np.array([])
        self._finger_ranges:    np.ndarray = np.array([])
        self._eye_link_ids:     dict       = {}

        # Cached fixed-camera matrices (never change)
        self._fixed_view_mat = None
        self._fixed_proj_mat = None

        self._prev_ee_pose: Optional[np.ndarray] = None

        # ── Step-based synchronisation ────────────────────────────────────
        self.sim_steps        = 0
        self.capture_frequency = int(240 / hz)  # PyBullet runs at 240 Hz

        # ── Background render thread ──────────────────────────────────────
        # Each job is a dict:
        #   { "sim_time": float,
        #     "state":    (proprio, ee_abs, ee_delta),
        #     "cameras":  { cam_key: (view_mat, proj_mat), ... } }
        # A None sentinel shuts the worker down.
        self._render_queue  = queue.Queue(maxsize=60)
        self._worker_thread: Optional[threading.Thread] = None
        
        self.success = False


    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._precompute_ids()
        renderer_name = ("OpenGL" if self._renderer == p.ER_BULLET_HARDWARE_OPENGL
                         else "TinyRenderer")
        print(f"[DataCollector] Started | {self.hz} Hz | "
              f"{self.render_w}×{self.render_h} | {renderer_name} | "
              f"cameras: {self.camera_keys}")
        self._worker_thread = threading.Thread(
            target=self._render_worker, name="dc-render", daemon=True)
        self._worker_thread.start()

    def step_collection(self) -> None:
        """Call EXACTLY ONCE per client.update_simulation() in sim_node.py."""
        self.sim_steps += 1
        if self.sim_steps % self.capture_frequency != 0:
            return

        sim_time = self.sim_steps / 240.0

        # ── 1. Collect state on the sim thread (fast PyBullet reads) ──────
        state = self._read_state()
        if state is None:
            return  # an error was already printed inside _read_state

        # ── 2. Compute camera matrices on the sim thread (pure math) ──────
        robot_id = self._robot_id
        cameras  = {}
        for cam_key in self.camera_keys:
            if cam_key == "fixed_camera":
                cameras[cam_key] = (self._fixed_view_mat, self._fixed_proj_mat)
            elif cam_key in self._eye_link_ids:
                cameras[cam_key] = get_pybullet_camera_matrices(
                    self._client_id, robot_id, self._eye_link_ids[cam_key])
            # unknown key → skip silently

        # ── 3. Push a render job to the background thread ─────────────────
        job = {
            "sim_time": sim_time,
            "state":    state,
            "cameras":  cameras,
            "object_ids": self.object_ids # whatever objects you have
        }
        try:
            self._render_queue.put_nowait(job)
        except queue.Full:
            # Worker is falling behind — drop this frame rather than stall sim
            print("[DataCollector] WARNING: render queue full, frame dropped.")

    def stop(self) -> None:
        """Signal the worker to finish and wait for it to drain the queue."""
        print(f"[DataCollector] Stopping — waiting for render queue to drain "
              f"({self._render_queue.qsize()} jobs pending)…")
        self._render_queue.put(None)  # poison pill
        if self._worker_thread:
            self._worker_thread.join()
        print(f"[DataCollector] Render thread done. "
              f"{len(self._timestamps)} samples in RAM.")

    def save(self) -> None:
        """Dumps all RAM buffers to disk. Call after stop()."""
        
        if not self.success:
            print("[DataCollector] Run marked as failed — discarding data.")
            return
        os.makedirs(self.save_dir, exist_ok=True)
        if not self._timestamps:
            print("[DataCollector] Nothing to save.")
            return

        print("\n[DataCollector] Dumping RAM to disk…")

        np.save(os.path.join(self.save_dir, "timestamps.npy"), np.array(self._timestamps))
        np.save(os.path.join(self.save_dir, "proprio.npy"),    np.array(self._proprio))
        np.save(os.path.join(self.save_dir, "ee_poses.npy"),   np.array(self._ee_poses))
        np.save(os.path.join(self.save_dir, "ee_deltas.npy"),  np.array(self._ee_deltas))
        np.save(os.path.join(self.save_dir, "obj_poses.npy"), np.array(self._obj_poses))

        if self._joint_names:
            with open(os.path.join(self.save_dir, "joint_names.txt"), "w") as f:
                f.write("\n".join(self._joint_names))
        with open(os.path.join(self.save_dir, "ee_columns.txt"), "w") as f:
            f.write("x,y,z,roll,pitch,yaw,gripper_state")

        for cam_key, frames in self._image_buffer.items():
            if frames:
                video_block = np.stack(frames)  # (N, H, W, 3)
                save_path   = os.path.join(self.save_dir, f"{cam_key}_video.npz")
                np.savez_compressed(save_path, video=video_block)
                print(f"  -> {cam_key}: {len(frames)} frames  {video_block.shape}")

        pro = np.array(self._proprio)
        
        cam_data = {
            "fixed_camera_pos": self.cam_pos,
            "fixed_camera_target": self.cam_target,
            "fixed_camera_up": self.cam_up
        }
        with open(os.path.join(self.save_dir, "camera_info.json"), "w") as f:
            json.dump(cam_data, f)
            
        print(f"[DataCollector] Done → {self.save_dir}  (proprio {pro.shape})")


    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _precompute_ids(self) -> None:
        client = self.client
        self._robot_id      = client.robot
        self._client_id     = client._client
        self._ee_link_id    = client.end_effector.link_id
        self._all_joint_ids = [j.robot_joint_id for j in client.joints]
        self._joint_names   = [j.name           for j in client.joints]

        finger_ids, finger_lo, finger_rng = [], [], []
        for jname in FINGER_JOINT_NAMES:
            for j in client.joints:
                if j.name == jname:
                    rng = j.upper_limit - j.lower_limit
                    if rng > 1e-6:
                        finger_ids.append(j.robot_joint_id)
                        finger_lo.append(j.lower_limit)
                        finger_rng.append(rng)
                    break
        self._finger_joint_ids = finger_ids
        self._finger_lo        = np.array(finger_lo,  dtype=np.float64)
        self._finger_ranges    = np.array(finger_rng, dtype=np.float64)

        for link in client.links:
            if "l_eye" in link.name:
                self._eye_link_ids["l_eye"] = link.robot_link_id
            elif "r_eye" in link.name:
                self._eye_link_ids["r_eye"] = link.robot_link_id

        # Cache fixed camera matrices — they never change
        #self._fixed_view_mat, self._fixed_proj_mat = get_fixed_camera_matrices()
        
        # --- ABSTRACTION LOGIC: Fetch fixed camera pose ---
        cam_pos = getattr(self.client.config.camera, "fixed_camera_pos", [-0.240, 0.098, 0.315])
        cam_target = getattr(self.client.config.camera, "fixed_camera_target", [-0.35, 0.1, -0.1])
        cam_up = getattr(self.client.config.camera, "fixed_camera_up", [0, 0, 1])

        if hasattr(self.client, "visualizer") and hasattr(self.client.visualizer, "eye_windows"):
            if "fixed_camera" in self.client.visualizer.eye_windows:
                fixed_cam = self.client.visualizer.eye_windows["fixed_camera"]
                cam_pos = fixed_cam.eye_pos.tolist()
                cam_target = fixed_cam.eye_target.tolist()
                cam_up = fixed_cam.up_vector.tolist()

        # Save to self so we can dump them to disk later
        self.cam_pos = cam_pos
        self.cam_target = cam_target
        self.cam_up = cam_up

        self._fixed_view_mat, self._fixed_proj_mat = get_fixed_camera_matrices(
            cam_pos=cam_pos, cam_target=cam_target, cam_up=cam_up
        )

    def _read_state(self):
        """Read all joint / EE state on the sim thread. Returns (proprio, ee_abs, ee_delta) or None."""
        robot_id = self._robot_id

        # Proprioception
        try:
            all_states = p.getJointStates(robot_id, self._all_joint_ids,
                                          physicsClientId=self._client_id)
            proprio = np.array([s[0] for s in all_states], dtype=np.float32)
        except Exception as e:
            print(f"[DataCollector] proprio error: {e}")
            return None

        # Gripper scalar
        try:
            if self._finger_joint_ids:
                finger_states = p.getJointStates(robot_id, self._finger_joint_ids,
                                                 physicsClientId=self._client_id)
                finger_pos = np.array([s[0] for s in finger_states], dtype=np.float64)
                grip = float(np.clip(
                    np.mean((finger_pos - self._finger_lo) / self._finger_ranges),
                    0.0, 1.0))
            else:
                grip = 0.0
        except Exception:
            grip = 0.0

        # EE pose
        try:
            ls    = p.getLinkState(robot_id, self._ee_link_id,
                                   computeForwardKinematics=False,
                                   physicsClientId=self._client_id)
            pos   = ls[4]
            euler = p.getEulerFromQuaternion(ls[5])
            ee_abs = np.array([pos[0], pos[1], pos[2],
                               euler[0], euler[1], euler[2], grip],
                              dtype=np.float32)
        except Exception as e:
            print(f"[DataCollector] ee error: {e}")
            return None

        # Delta EE
        if self._prev_ee_pose is None:
            ee_delta = np.zeros(7, dtype=np.float32)
        else:
            ee_delta = ee_abs - self._prev_ee_pose
            ee_delta[3:6] = (ee_delta[3:6] + np.pi) % (2 * np.pi) - np.pi

        self._prev_ee_pose = ee_abs
        
        # Object poses
        obj_poses = []
        for oid in self.object_ids:
            try:
                pos, orn = p.getBasePositionAndOrientation(oid, physicsClientId=self._client_id)
                obj_poses.extend(list(pos) + list(orn))  # 7 values per object
            except Exception:
                obj_poses.extend([0.0] * 7)
        obj_poses_arr = np.array(obj_poses, dtype=np.float32)

        return (proprio, ee_abs, ee_delta, obj_poses_arr)

    def _render_worker(self) -> None:
        """Background thread: drains the render queue and calls getCameraImage()."""
        while True:
            job = self._render_queue.get()

            # Poison pill → flush and exit
            if job is None:
                break

            sim_time              = job["sim_time"]
            proprio, ee_abs, ee_delta, obj_poses_arr = job["state"]
            cameras               = job["cameras"]

            # Render every requested camera
            rendered = {}
            for cam_key, (view_mat, proj_mat) in cameras.items():
                try:
                    _, _, rgba, _, _ = p.getCameraImage(
                        width=self.render_w,
                        height=self.render_h,
                        viewMatrix=view_mat,
                        projectionMatrix=proj_mat,
                        renderer=self._renderer,
                        shadow=0,
                        physicsClientId=self._client_id
                    )
                    img = (np.reshape(rgba, (self.render_h, self.render_w, 4))
                        [:, :, :3].astype(np.uint8))

                    if job.get("object_ids"):
                        ellipses = get_shadow_ellipses(
                            job["object_ids"], view_mat, proj_mat,
                            self.render_w, self.render_h, self._client_id)
                        img = add_ground_shadow(img, ellipses)
                    rendered[cam_key] = img
                    
                except Exception as e:
                    print(f"[DataCollector] render error ({cam_key}): {e}")

            # Commit everything to RAM buffers (lock only for image appends)
            self._timestamps.append(sim_time)
            self._proprio.append(proprio)
            self._ee_poses.append(ee_abs)
            self._ee_deltas.append(ee_delta)
            self._obj_poses.append(obj_poses_arr)

            with self._buffer_lock:
                for cam_key, img in rendered.items():
                    self._image_buffer[cam_key].append(img)