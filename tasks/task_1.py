from tasks.base_task import BaseTask
from icub_pybullet.utils import Pose
import numpy as np
import time
import pybullet as p

class T1Stacking(BaseTask):
    CONFIG_PATH = "configs/t1.yaml"
    BASE_PROMPT = "Stack the red cube on top of the blue cube."
    OBJECT_NAMES = ["cube", "cube_"]
    
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
                
        # --- OBJECT RANDOMIZATION ---
        # Found out corners through adjustable cross
        safety_margin = np.array([-0.12,-0.12])
        min_x, min_y = (-0.6, -0.3) - safety_margin
        max_x, max_y = (-0.22, 0.39) + np.array([0, -0.12])
        z_from_yaml = self.client.config.urdfs.positions[3][2]
        
        #----TO VISUALIZE AABB CORNERS----#
        #self.client.visualizer.add_debug_cross("aabb", [min_x, min_y, z_from_yaml])
        #self.client.visualizer.add_debug_cross("aabb2", [max_x, max_y, z_from_yaml])

        # Generate random X and Y within the table surface
        padding = 0.05
        mid_y = (min_y + max_y) / 2
        
        # Zone 1: Left half
        rand_x_1 = np.random.uniform(min_x + padding, max_x - padding)
        rand_y_1 = np.random.uniform(min_y + padding, mid_y - padding)
        
        # Zone 2: Right half
        rand_x_2 = np.random.uniform(min_x + padding, max_x - padding)
        rand_y_2 = np.random.uniform(mid_y + 0.05 + padding, max_y - padding - 0.12)  # Additional margin for the second cube\
        

        print(f"Cube Variation 1-> X: {rand_x_1:.3f}, Y: {rand_y_1:.3f}, Z: {z_from_yaml}")
        self.client.resetBasePositionAndOrientation(
            self.target_cube_id, 
            [rand_x_1, rand_y_1, z_from_yaml], 
            [0, 0, 0, 1]
        )
        print(f"Cube Variation 2-> X: {rand_x_2:.3f}, Y: {rand_y_2:.3f}, Z: {z_from_yaml}")
        self.client.resetBasePositionAndOrientation(
            self.cube_id,
            [rand_x_2, rand_y_2, z_from_yaml], 
            [0, 0, 0, 1]
        )
        
    def setup_environment(self):
        # Your randomization logic here
        self.cube_id = self.client.get_object_id("cube_")
        self.target_cube_id = self.client.get_object_id("cube")
        self.closed_fingers = 0
        
        # --- THE RESET FIX 1: Melt the Super Glue ---
        # If constraints exist from a previous episode, delete them!
        # Look for the glue on the persistent CLIENT, not the temporary self!
        if hasattr(self.client, 'nail_cid'):
            try:
                p.removeConstraint(self.client.nail_cid, physicsClientId=self.client._client)
            except: pass
            del self.client.nail_cid
            
        if hasattr(self.client, 'weld_cid'):
            try:
                p.removeConstraint(self.client.weld_cid, physicsClientId=self.client._client)
            except: pass
            del self.client.weld_cid
            
        # --- THE RESET FIX 2: Turn off "Ghost Mode" ---
        for link_i in range(-1, p.getNumJoints(self.client.robot, physicsClientId=self.client._client)):
            p.setCollisionFilterPair(self.client.robot, self.cube_id, link_i, -1, 1, physicsClientId=self.client._client)
            p.setCollisionFilterPair(self.client.robot, self.target_cube_id, link_i, -1, 1, physicsClientId=self.client._client)

    def execute_expert(self):
        # 1. Setup Dynamics & Pre-grasp
        self.skills.set_dynamics(self.cube_id, mass=0.2)
        self.skills.set_dynamics(self.target_cube_id, mass=0.7)
        self.skills.set_object_to_track(self.cube_id)
        
        self.skills.move_joint("l_shoulder_pitch", -1.6, wait=True)

        cube_pos = self.client.getBasePositionAndOrientation(self.cube_id)[0]
        pre_grasp_pose = Pose(pos=[cube_pos[0] + 0.13, cube_pos[1] + 0.08, cube_pos[2] + 0.15], 
                            ori=p.getQuaternionFromEuler([1.5, 0, 0]))
        
        self.skills.track_target_with_head()
        self.client.move_cartesian(pre_grasp_pose)
        self.skills.move_joint("r_hand_thumb_0_joint", 1.209, wait=True)

        # 2. Descend
        self.skills.rotate_ee_local(yaw=np.pi/5) # Rotate 36 degrees
        self.skills.move_ee_relative(dz=-0.055)

        # 3. Grasp
        fingers = {"thumb": 0, "index":0, "middle":0, "ring":0}
        for f in fingers:
            fingers[f] = self.skills.flex_finger_to_contact(self.cube_id, f)
        self.skills.secure_grasp(self.cube_id)

        # The Failsafe Check
        if not fingers["thumb"] or sum([fingers["index"], fingers["middle"], fingers["ring"]]) < 2:
            self.client.logger.error("Failed to grab the cube with sufficient fingers.")
            self.valid_grasp = False
            
            # --- THE FAILED GRIP FIX ---
            # Completely detach and unfold the fingers BEFORE aborting
            self.skills.release_grasp()
            self.skills.open_hand()
            
            # Force the physics engine to process the opening hand 
            # so the physical meshes clear the cube before the teleport reset!
            for _ in range(40):
                self.client.update_simulation()
            # ---------------------------
            
            return  # EARLY EXIT: Now safely stops the script

        # If it passes, lock the constraints and mark as valid
        self.valid_grasp = True

        # 4. Lift & Align
        for _ in range(11):
            self.skills.move_ee_relative(dz=0.01, velocity=10)
            self.client.update_simulation()
            
        target_pos = self.client.getBasePositionAndOrientation(self.target_cube_id)[0]
        self.skills.align_ee_xy_to_target(target_pos)
        
        # 5. Drop & Retreat
        self.skills.level_object_to_table(self.cube_id, target_pos, extra = 0.06)

        # Let the arm settle before dropping
        for _ in range(60):
            self.client.update_simulation()
                    
        # STEP A: Turn the robot into a ghost against both cubes
        self.skills.disable_robot_object_collision(self.cube_id)
        self.skills.disable_robot_object_collision(self.target_cube_id)

        # --- NEW: NAIL THE BOTTOM CUBE TO THE TABLE ---

        target_real_pos, target_real_orn = p.getBasePositionAndOrientation(self.target_cube_id, physicsClientId=self.client._client)
        self.client.nail_cid = p.createConstraint( 
            parentBodyUniqueId=self.target_cube_id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1, # -1 refers to the World (the table)
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=target_real_pos,
            parentFrameOrientation=[0, 0, 0, 1],
            childFrameOrientation=target_real_orn,
            physicsClientId=self.client._client
        )
        print("[T1] Bottom cube is now nailed to the table.")
        
        # STEP B: Release grip and open hand
        self.skills.release_grasp()
        self.skills.open_hand()
        
        # STEP C: Let it fall naturally (NO SNAPPING)
        for _ in range(200):
            linear_vel, _ = p.getBaseVelocity(self.cube_id, physicsClientId=self.client._client)
            current_pos, current_orn = p.getBasePositionAndOrientation(self.cube_id, physicsClientId=self.client._client)
            
            # Only force it to stay flat, do NOT teleport X and Y
            p.resetBasePositionAndOrientation(
                self.cube_id,
                current_pos,  # Keep its exact current natural position
                [0, 0, 0, 1], # Force flat orientation so it doesn't land on a corner
                physicsClientId=self.client._client
            )
            
            # Kill sideways drift and tumbling, allow natural Z falling
            p.resetBaseVelocity(self.cube_id, [0, 0, linear_vel[2]], [0, 0, 0], physicsClientId=self.client._client)
            self.client.update_simulation()


        # --- STEP D: CONDITIONAL SUPER GLUE ---
        
        #  Get the top cube's exact center coordinates
        cube_pos, cube_orn = p.getBasePositionAndOrientation(self.cube_id, physicsClientId=self.client._client)
        
        # Get the bottom cube's boundaries (AABB)
        aabb_min, aabb_max = p.getAABB(self.target_cube_id, physicsClientId=self.client._client)
        side = max(aabb_max[0] - aabb_min[0], aabb_max[1] - aabb_min[1])  
        cube_center = (cube_pos[0] + side/2, cube_pos[1] + side/2, cube_pos[2])  # Just to be explicit about the format
        
        # Check if the top cube's center X and Y falls strictly inside the bottom cube's boundaries
        is_center_supported = (aabb_min[0] <= cube_center[0] <= aabb_max[0]) and (aabb_min[1] <= cube_center[1] <= aabb_max[1] )
        is_on_top = aabb_max[2] + 0.01 > cube_center[2] > aabb_max[2]  # Ensure it's above the bottom cube

        if is_on_top and is_center_supported:
            # Target cube's exact position
            target_real_pos, target_real_orn = p.getBasePositionAndOrientation(self.target_cube_id, physicsClientId=self.client._client)

            # --- THE FIX: FORCE PERFECT FLUSH Z ---
            # 1. Calculate the exact physical height of the cube from its bounding box
            cube_height = aabb_max[2] - aabb_min[2]
            
            # 2. The perfect flush Z is exactly one cube-height above the bottom cube's center
            flush_z = target_real_pos[2] + cube_height
            
            # 3. Snap the top cube down to the flush Z, keeping its natural X/Y drop location
            p.resetBasePositionAndOrientation(
                self.cube_id,
                [cube_pos[0], cube_pos[1], flush_z],
                [0, 0, 0, 1],  # Keep it perfectly flat
                physicsClientId=self.client._client
            )
            
            # 4. Re-read the new, perfectly flush coordinates for the constraint math
            cube_pos, cube_orn = p.getBasePositionAndOrientation(self.cube_id, physicsClientId=self.client._client)
            # ----------------------------------------

            # Calculate the exact 3D spatial difference between them
            inv_target_pos, inv_target_orn = p.invertTransform(target_real_pos, target_real_orn)
            rel_pos, rel_orn = p.multiplyTransforms(inv_target_pos, inv_target_orn, cube_pos, cube_orn)

            # Create the unbreakable fixed joint
            self.client.weld_cid = p.createConstraint(
                parentBodyUniqueId=self.target_cube_id,
                parentLinkIndex=-1,
                childBodyUniqueId=self.cube_id,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=rel_pos,
                childFramePosition=[0, 0, 0],
                parentFrameOrientation=rel_orn,
                childFrameOrientation=[0, 0, 0, 1],
                physicsClientId=self.client._client
            )
            print("[T1] Perfect drop! Top cube is FLUSH and Welded!")
        else:
            print("[T1] Bad drop! Top cube's center missed. No weld applied!")

        # STEP E: Retreat the arm
        for _ in range(25):
            self.skills.move_ee_relative(dz=0.01, velocity=10)
            self.client.update_simulation()


    def check_success(self) -> bool:
        # 1. Did the grasp fail earlier in the episode?
        if not getattr(self, 'valid_grasp', False):
            return False
            
        # 2. Did the cube fall off the stack?
        cube_z = self.client.getBasePositionAndOrientation(self.cube_id)[0][2]
        target_z = self.client.getBasePositionAndOrientation(self.target_cube_id)[0][2]
        
        if cube_z < target_z + 0.03:
            self.client.logger.error("Cube fell on the table.")
            return False
        return True