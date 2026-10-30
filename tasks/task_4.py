from tasks.base_task import BaseTask
from icub_pybullet.utils import Pose
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

class T4Opening(BaseTask):
    CONFIG_PATH = "configs/t4.yaml"
    BASE_PROMPT = "Open the box."
    OBJECT_NAMES = ["box_top", "box_base"]  # <--- NEW

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
        min_x_box, max_x_box = -0.35, -0.3
        min_y_box, max_y_box = 0.13, 0.15
        
        rand_x = np.random.uniform(min_x_box, max_x_box)
        rand_y = np.random.uniform(min_y_box, max_y_box)
        
        # rand_x = max_x_box
        # rand_y = min_y_box
        
        box_bottom_pos = self.client.getBasePositionAndOrientation(self.box_bottom_id)[0]
        box_top_ori = self.client.getBasePositionAndOrientation(self.box_top_id)[1]
        z_bottom = box_bottom_pos[2]
        
        box_top_pos = self.client.getBasePositionAndOrientation(self.box_top_id)[0]
        z_top = box_top_pos[2]
        
        print(f"Box Variation X: {rand_x:.3f}, Y: {rand_y:.3f}")
        self.client.resetBasePositionAndOrientation(
            self.box_bottom_id,
            [rand_x, rand_y, z_bottom],
            [0, 0, 0, 1]
        )
        
        self.client.resetBasePositionAndOrientation(
            self.box_top_id,
            [rand_x, rand_y, z_top],  
            box_top_ori
        )
    
    def setup_environment(self):
        # Your randomization logic here
        self.box_top_id = self.client.get_object_id("box_top")
        self.box_bottom_id = self.client.get_object_id("box_base")
        self.target_final_pos = np.array([0,0,0])
    

    def execute_expert(self):
        # 1. Setup Dynamics & Pre-grasp
        self.skills.set_dynamics(self.box_top_id, mass=0.2)
        self.skills.set_object_to_track(self.box_top_id)
        
        self.skills.move_joint("l_shoulder_pitch", -1.6, wait=True)

        box_top_pos = self.client.getBasePositionAndOrientation(self.box_top_id)[0]
        pre_grasp_pose = Pose(pos=[box_top_pos[0] + 0.13, box_top_pos[1] + 0.08, box_top_pos[2] + 0.3], 
                            ori=p.getQuaternionFromEuler([1.5, 0, 0]))

        self.skills.track_target_with_head()
        self.client.move_cartesian(pre_grasp_pose, timeout=5.0)
        #self.skills.move_joint("r_hand_thumb_0_joint", 1.209, wait=True)

        # 2. Descend
        pose = Pose(pos = [-0.31255515336990356, 0.1912525188922882, 0.1273146212100983],
                    ori = [0.7538662552833557, -0.24759431183338165, 0.2670873999595642, 0.5468518137931824])
        print("Descending to grasp pose...")
        self.client.move_cartesian(pose, timeout = 5.0)
        
        
        self.client.move_fingers()
        print("Closing fingers to grasp the box...")
        while not self.client.motion_done(check_collision=False):
            self.client.update_simulation()
        print("Fingers closed, securing grasp...")
        self.skills.secure_grasp_all_contacts(self.box_top_id)
            
        self.skills.move_ee_relative(dz=0.17)
        base_pos = self.client.getBasePositionAndOrientation(self.box_bottom_id)[0]
        self.target_final_pos = [base_pos[0], base_pos[1] - 0.2, base_pos[2]]
        self.skills.align_ee_xy_to_target(self.target_final_pos)
        self.skills.move_ee_relative(dz=-0.1)
        
        self.skills.release_grasp()
        self.skills.open_hand()
        
        while not self.client.motion_done(check_collision=False):
                self.client.update_simulation()
                self.skills.track_target_with_head()
        


    def check_success(self) -> bool:
        top_pos = self.client.getBasePositionAndOrientation(self.box_top_id)[0]
        distance = np.linalg.norm(np.array(top_pos) - self.target_final_pos)
        
        if distance < 0.1:
            print(f"Success! Box is within {distance:.3f} meters of the target.")
            return True
        else:              
            print(f"Failure. Box is {distance:.3f} meters away from the target.")
        
        return False