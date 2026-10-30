from tasks.base_task import BaseTask
from icub_pybullet.utils import Pose
import numpy as np
import pybullet as p
import open3d as o3d
import json
import os


class T5Cleaning(BaseTask):
    CONFIG_PATH = "configs/t5.yaml"
    BASE_PROMPT = "Clean the spill with the sponge."
    OBJECT_NAMES = ["sponge"]

    def object_randomizer(self):
        # --- CAMERA RANDOMIZATION ---
        # 1. Check the YAML config toggle (Defaults to False if you forget to add it)
        if getattr(self.client.config.camera, "randomize_camera", False):
            
            # Read baselines
            base_pos = getattr(self.client.config.camera, "fixed_camera_pos", [-0.6742, -0.0194, 0.4362])
            base_target = getattr(self.client.config.camera, "fixed_camera_target", [-0.2265, -0.0107, -0.4580])
            
            # Read ranges from YAML
            pos_noise = getattr(self.client.config.camera, "random_pos_range", 0.05)
            target_noise = getattr(self.client.config.camera, "random_target_range", 0.02)

            # Apply noise
            new_pos = [
                base_pos[0] + np.random.uniform(-pos_noise, pos_noise),
                base_pos[1] + np.random.uniform(-pos_noise, pos_noise),
                base_pos[2] + np.random.uniform(-pos_noise, pos_noise)
            ]
            new_target = [
                base_target[0] + np.random.uniform(-target_noise, target_noise),
                base_target[1] + np.random.uniform(-target_noise, target_noise),
                base_target[2]
            ]

            # Overwrite for this specific episode
            self.client.config.camera.fixed_camera_pos = new_pos
            self.client.config.camera.fixed_camera_target = new_target

            # Update live GUI if it is open
            if hasattr(self.client, "visualizer") and "fixed_camera" in self.client.visualizer.eye_windows:
                cam = self.client.visualizer.eye_windows["fixed_camera"]
                cam.set_camera_pose(new_pos, new_target)
                cam.update_view()
                

    def setup_environment(self):
        self.spill_log = {}
        self.sponge_id   = self.client.get_object_id("sponge")
        self.skills.disable_robot_object_collision(self.sponge_id)
        self.spill_particles = {}
        self._spawn_spill()
        
        self._sponge_table_z = None 
        # In setup_environment, after getting the sponge_id:
        original_print_collision = self.client.print_collision_info

        def safe_print_collision(c=None):
            try:
                original_print_collision(c)
            except UnboundLocalError:
                pass  # pyCub can't resolve the sponge body name — safe to ignore

        self.client.print_collision_info = safe_print_collision

    # ─── SPILL SPAWNING ───────────────────────────────────────────────────────
    def _spawn_spill(self):
        from core.spill import spawn_spill
        self.spill_particles, self.spill_log = spawn_spill(self.client)
        print(f"[T5] Spawned {len(self.spill_particles)} spill particles.")


    # ─── SPILL CLEANING ───────────────────────────────────────────────────────

    def clean_step(self):
        """Remove any spill particle the sponge is currently touching."""
        # Pin sponge to table surface every step — prevents any lifting
        if self._sponge_table_z is not None:
            sponge_pos, sponge_orn = p.getBasePositionAndOrientation(
                self.sponge_id, physicsClientId=self.client._client)
            if sponge_pos[2] > self._sponge_table_z + 0.0005:  # tolerance
                p.resetBasePositionAndOrientation(
                    self.sponge_id,
                    [sponge_pos[0], sponge_pos[1], self._sponge_table_z],
                    sponge_orn,
                    physicsClientId=self.client._client)
        to_remove = []
        for name, data in self.spill_particles.items():
            contacts = p.getContactPoints(
                bodyA=self.sponge_id,
                bodyB=data["body_id"],
                physicsClientId=self.client._client)
            if contacts:
                to_remove.append(name)

        for name in to_remove:
            if hasattr(self, 'collector'):
                self.spill_log[name]["deleted_step"] = len(self.collector._timestamps)
                
            # --- FIX: Only remove from visualizer if it exists ---
            if hasattr(self.client, "visualizer") and self.client.visualizer:
                try:
                    self.client.visualizer.remove_primitive(name)
                except AttributeError:
                    pass
            
            p.removeBody(self.spill_particles[name]["body_id"],
                         physicsClientId=self.client._client)
            del self.spill_particles[name]

        if to_remove:
            print(f"[T5] Cleaned {len(to_remove)} particles. "
                  f"{len(self.spill_particles)} remaining.")

    # ─── EXPERT ROUTINE ───────────────────────────────────────────────────────

    def execute_expert(self):
        """
        Phase 1 — Approach sponge from above with flat open hand
        Phase 2 — Descend onto sponge
        Phase 3 — Constrain sponge to hand (simulates grip/press)
        Phase 4 — Drag sponge across spill in a coverage pattern
        Phase 5 — Lift and release
        """

        # ── Phase 1: Move arm to pre-approach above the sponge ───────────────
        print("[T5] Phase 1: Pre-approach above sponge.")
        self.skills.set_object_to_track(self.sponge_id)
        self.skills.track_target_with_head()

        sponge_pos = list(p.getBasePositionAndOrientation(
            self.sponge_id, physicsClientId=self.client._client)[0])

        self.skills.move_joint("l_shoulder_pitch", -1.6, wait=True)
        self.skills.move_joint("l_shoulder_roll", 2.1, wait=True)
        self.skills.move_joint("l_elbow", 1.8, wait=True)
        
        self.skills.open_hand()

        # Flat hand orientation — palm facing down
        flat_palm_orn = p.getQuaternionFromEuler([np.pi/2, 0, 0])
        

        pre_approach = Pose(
            pos=[sponge_pos[0] + 0.05, sponge_pos[1], sponge_pos[2] + 0.1],
            ori=flat_palm_orn)
        self.client.move_cartesian(pre_approach, velocity=5, timeout=2)
        self._wait_motion()

        self.skills.set_object_to_track(self.sponge_id)
        self.skills.track_target_with_head()

        # ── Phase 2: Descend onto sponge ─────────────────────────────────────
        print("[T5] Phase 2: Descending onto sponge.")

        target_z = sponge_pos[2] + 0.035  # hover just above sponge surface
        curr_pose = self.client.end_effector.get_position()

        while curr_pose.pos[2] > target_z:
            self.skills.move_ee_relative(dz=-0.005, velocity=10, timeout=1)
            self.client.update_simulation()
            self.skills.track_target_with_head()
            curr_pose = self.client.end_effector.get_position()
        
        sponge_pos = p.getBasePositionAndOrientation(
            self.sponge_id, physicsClientId=self.client._client)[0]
        self._sponge_table_z = sponge_pos[2]  # freeze at this height forever

        # Small settling steps
        for _ in range(30):
            self.client.update_simulation()
            self.skills.track_target_with_head()

        # ── Phase 3: Constrain sponge to palm ────────────────────────────────
        if not self.skills.active_constraints:
            # No contact found — try pressing down a little more
            print("[T5] No contact yet, pressing down further.")
            for _ in range(3):
                self.skills.move_ee_relative(dz=-0.003, velocity=2, timeout=1)
                for _ in range(20):
                    self.client.update_simulation()
            self.skills.secure_grasp_all_contacts(self.sponge_id)
            

        print("[T5] Phase 3: Attaching sponge to palm.")
        sponge_pos = p.getBasePositionAndOrientation(
            self.sponge_id, physicsClientId=self.client._client)[0]
        self._sponge_table_z = sponge_pos[2]
        self.skills.attach_sponge_to_palm(self.sponge_id, self._sponge_table_z)
        self.valid_clean = True

        # ── Phase 4: Coverage sweep over the spill ────────────────────────────
        print("[T5] Phase 4: Cleaning sweep.")

        # Build a bounding box over all remaining spill particles
        positions  = np.array([d["pos"] for d in self.spill_particles.values()])
        spill_min  = positions.min(axis=0)
        spill_max  = positions.max(axis=0)
        sweep_z    = target_z  # keep hand at the same height throughout

        # Sweep in horizontal strips (lawnmower pattern) across the spill bbox
        n_strips  = 6
        strip_ys  = np.linspace(spill_min[1] - 0.02, spill_max[1] + 0.02, n_strips)
        x_start   = spill_min[0] - 0.03
        x_end     = spill_max[0] + 0.03

        curr_pose = self.client.end_effector.get_position()
        current_orn = curr_pose.ori

        for strip_i, strip_y in enumerate(strip_ys):
            self.skills.move_joint("l_elbow", 1.8, wait=True)
            self.skills.move_joint("r_hand_thumb_0_joint", 0.0, wait=True)
            self.skills.move_joint_relative("l_shoulder_roll", 0.3, wait=True)  # slight downward pitch for better contact
            self.skills.open_hand()
            # Alternate sweep direction (left→right, right→left)
            if strip_i % 2 == 0:
                sweep_x_from, sweep_x_to = x_start, x_end
            else:
                sweep_x_from, sweep_x_to = x_end, x_start

            # Move to start of this strip
            strip_start = Pose(
                pos=[sweep_x_from, strip_y, sweep_z],
                ori=current_orn)
            self.skills.move_cartesian_smoothly(strip_start, steps=30, timeout=3.0)
            self._wait_motion(steps=15)
            
            all_cleaned_early = False

            # Sweep across the strip in small steps
            n_sweep_steps = 25
            for step_i in range(n_sweep_steps + 1):
                t = step_i / n_sweep_steps
                sweep_x = sweep_x_from + t * (sweep_x_to - sweep_x_from)

                self.client.move_cartesian(
                    Pose([sweep_x, strip_y, sweep_z], current_orn),
                    velocity=10, timeout=1, check_collision=False)

                # Run several sim steps per sweep step so contacts register
                for _ in range(10):
                    self.client.update_simulation()
                    self.skills.track_target_with_head()
                    self.clean_step()
                    
                    if not self.spill_particles:
                        all_cleaned_early = True
                        break
                if all_cleaned_early:
                    break
                
            if all_cleaned_early:
                print(f"[T5] Spill fully cleaned mid-sweep on Strip {strip_i + 1}! Concluding immediately.")
                break
            
            print(f"[T5] Strip {strip_i + 1}/{n_strips} done. "
                  f"{len(self.spill_particles)} particles remaining.")

            if not self.spill_particles:
                print("[T5] Spill fully cleaned early!")
                break
        # ── Phase 4b: Mop up any remaining particles ─────────────────────────
        if self.spill_particles:
            print(f"[T5] Mopping up {len(self.spill_particles)} remaining particles.")
            
            # Sort by X so the path is efficient (no random jumping)
            remaining = sorted(self.spill_particles.values(), key=lambda d: d["pos"][0])
            
            for data in remaining:
                self.skills.move_joint("l_elbow", 1.8, wait=True)
                self.skills.move_joint("r_hand_thumb_0_joint", 0.0, wait=True)
                self.skills.move_joint_relative("l_shoulder_roll", 0.3, wait=True)
                self.skills.open_hand()
                if not self.spill_particles:  # might get cleaned by contact en route
                    break
                
                tx, ty, _ = data["pos"]
                mop_pose = Pose(
                    pos=[tx, ty, sweep_z],
                    ori=current_orn)
                
                self.skills.move_cartesian_smoothly(mop_pose, steps=20, timeout=3.0)
                
                # Hover over the particle for several steps so contact registers
                for _ in range(20):
                    self.client.update_simulation()
                    self.skills.track_target_with_head()
                    self.clean_step()
                
                if not self.spill_particles:
                    print("[T5] All particles cleaned during mop-up!")
                    break
    
            if self.spill_particles:
                print(f"[T5] {len(self.spill_particles)} particles could not be reached.")

        # ── Phase 5: Release and lift ─────────────────────────────────────────
        print("[T5] Phase 5: Releasing sponge and retreating.")
        self.skills.release_grasp()

        for _ in range(12):
            self.skills.move_ee_relative(dz=0.01, velocity=8)
            self.client.update_simulation()
            
        if hasattr(self, 'collector') and getattr(self.collector, 'save_dir', None):
            spill_path = os.path.join(self.collector.save_dir, "spill_data.json")
            
            # This forces the OS to build 'episode_001' if it doesn't exist yet
            os.makedirs(os.path.dirname(spill_path), exist_ok=True)
            
            with open(spill_path, "w") as f:
                json.dump(self.spill_log, f)
            print(f"[T5] Saved spill_data.json with {len(self.spill_log)} particles.")
    # ─── SUCCESS CHECK ────────────────────────────────────────────────────────

    def check_success(self) -> bool:
        if not getattr(self, "valid_clean", False):
            return False
        remaining = len(self.spill_particles)
        if remaining > 2:  # allow a few stragglers, but not too many
            print(f"[T5] {remaining} spill particles remain — not fully cleaned.")
            return False
        return True

    # ─── HELPERS ─────────────────────────────────────────────────────────────

    def _wait_motion(self, steps=0):
        """Wait for motion_done, then run extra settling steps."""
        while not self.client.motion_done(check_collision=False):
            self.client.update_simulation()
            self.skills.track_target_with_head()
        for _ in range(steps):
            self.client.update_simulation()
