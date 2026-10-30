from tasks.base_task import BaseTask
from icub_pybullet.utils import Pose
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

class T3Pushing(BaseTask):
    CONFIG_PATH = "configs/t3.yaml"
    BASE_PROMPT = "Push the green box to the centre of the table."
    OBJECT_NAMES = ["box", "other_box"]  # <--- NEW
    
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
        aabb_min_green_box, aabb_max_green_box = p.getAABB(self.green_box_id)
        rand_x_1 = np.random.uniform(aabb_min_green_box[0], aabb_max_green_box[0]-0.1)
        rand_y_1 = np.random.uniform(aabb_min_green_box[1], aabb_max_green_box[1])
        
        self.client.resetBasePositionAndOrientation(
            self.green_box_id, 
            [rand_x_1, rand_y_1, aabb_min_green_box[2]], 
            [0, 0, 1, 1]
        )
                
        rand_selector = np.random.randint(0, 2)      
        red_box_og = self.client.getBasePositionAndOrientation(self.red_box_id)[0]
        
        if rand_selector == 0:
            safety_margin = 0.04
            rand_x_2 = np.random.uniform(-0.59 + safety_margin, -0.19 - safety_margin)
            self.client.resetBasePositionAndOrientation(
                self.red_box_id, 
                [rand_x_2, red_box_og[1], red_box_og[2]], 
                [0, 0, 1, 1]
            )
        else:
            safety_margin = 0.05
            rand_y_2 = np.random.uniform(-0.001 - safety_margin, -0.39 + safety_margin)
            self.client.resetBasePositionAndOrientation(
                self.red_box_id, 
                [red_box_og[0], rand_y_2, red_box_og[2]], 
                [0, 0, 1, 1]
            )  
    
    def setup_environment(self):
        # Fetch object IDs based on the names defined in t3.yaml
        self.red_box_id = self.client.get_object_id("box")         # Red box
        self.green_box_id = self.client.get_object_id("other_box") # Green box
        
        # Define table center and save it as a class variable for check_success
        min_x, min_y = (-0.6, -0.4)
        max_x, max_y = (-0.22, 0.339)
        self.target_final_xy = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])
        
    
    def execute_expert(self):
        # =======================================================
        # PHASE 1: Preparation & Hand Posture
        # =======================================================
        self.skills.set_object_to_track(self.green_box_id)
        self.skills.set_dynamics(self.green_box_id, mass=0.5, lat_fric=0.5) 
        
        self.skills.move_joint("l_shoulder_pitch", -1.6, wait=True)
        self.skills.open_hand()
        
        # =======================================================
        # PHASE 2 & 3: Math & Alignment (Delegated to Skills)
        # =======================================================
        face_normal, face_name, push_orn, box_pos, box_xy = self.skills.get_push_alignment(
            self.green_box_id, self.target_final_xy
        )
        print(f"Aligning to push the {face_name} face of the box.")

        # =======================================================
        # PHASE 4: Define Waypoints
        # =======================================================
        offset_dist = 0.08  
        start_xy = box_xy + (face_normal * offset_dist)
        
        push_z = box_pos[2] + 0.04  
        pre_push_z = push_z + 0.1   
        
        # Shift 0.07 to compensate difference between EEF frame and palm x's
        eef_offset_x = 0.07 
        
        pre_push_pose = Pose([start_xy[0] + eef_offset_x, start_xy[1], pre_push_z], push_orn) 
        start_pose = Pose([start_xy[0] + eef_offset_x, start_xy[1], push_z], push_orn)
        end_pose = Pose([self.target_final_xy[0] + eef_offset_x, self.target_final_xy[1], push_z], push_orn)

        # =======================================================
        # PHASE 5: Execute Movement
        # =======================================================
        self.client.move_cartesian(pre_push_pose, timeout=3.0)
        self.client.update_simulation()
        
        self.skills.move_cartesian_smoothly(start_pose, steps=20)
        self.skills.move_cartesian_smoothly(end_pose, steps=50, timeout=2.0)
        self.skills.move_cartesian_smoothly(pre_push_pose, steps=20)

    def check_success(self) -> bool:
        # Get the final box position
        box_pos = self.client.getBasePositionAndOrientation(self.green_box_id)[0]
        box_xy = np.array(box_pos[:2])
        
        # Calculate distance to the center
        distance = np.linalg.norm(box_xy - self.target_final_xy)
        
        # 5cm tolerance for success
        if distance > 0.06:  
            self.client.logger.error(f"Box is {distance:.3f}m away from the center.")
            return False
            
        return True