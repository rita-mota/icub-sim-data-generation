from tasks.base_task import BaseTask
from icub_pybullet.utils import Pose
import numpy as np
import pybullet as p

class T2Moving(BaseTask):
    CONFIG_PATH = "configs/t2.yaml"
    BASE_PROMPT = "Move the bottle to the front of the laptop."
    OBJECT_NAMES = ["bottle", "computer"]

    def object_randomizer(self):
        # --- CAMERA RANDOMIZATION ---
        # 1. Check the YAML config toggle (Defaults to False if you forget to add it)
        if getattr(self.client.config.camera, "randomize_camera", False):
            
            # Read baselines
            base_pos = getattr(self.client.config.camera, "fixed_camera_pos", [-0.5731, -0.0218, 0.4800])
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
                
        min_y_comp, max_y_comp = -0.15, 0.021
        rand_y_1 = np.random.uniform(min_y_comp, max_y_comp)
        
        min_x_comp, max_x_comp = -0.45, -0.44
        rand_x_1 = np.random.uniform(min_x_comp, max_x_comp)
        
        computer_pos, computer_ori = self.client.getBasePositionAndOrientation(self.computer_id)[0], self.client.getBasePositionAndOrientation(self.computer_id)[1]
        z_from_yaml = computer_pos[2]
        
        print(f"Computer Variation X: {rand_x_1:.3f}, Y: {rand_y_1:.3f}")
        
        self.client.resetBasePositionAndOrientation(
            self.computer_id, 
            [rand_x_1, rand_y_1, z_from_yaml], 
            computer_ori
        )
        
        bottle_pos = self.client.getBasePositionAndOrientation(self.bottle_id)[0]
        min_x_bot, max_x_bot = bottle_pos[0], -0.45
        rand_x_2 = np.random.uniform(min_x_bot, max_x_bot)
        y_bot = computer_pos[1] + 0.25
                
        
        print(f"Bottle Variation X: {rand_x_2:.3f}")
        self.client.resetBasePositionAndOrientation(
            self.bottle_id,
            [rand_x_2, y_bot, z_from_yaml], 
            [0, 0, 0, 1]
        )
        
        
    def setup_environment(self):
        # Your randomization logic here
        self.bottle_id = self.client.get_object_id("bottle")
        self.computer_id = self.client.get_object_id("computer")
        self.target_final_pos = np.array([0,0,0])

    def execute_expert(self):
            # 1. Setup Dynamics & Pre-grasp
            self.skills.set_dynamics(self.bottle_id, mass=0.2)
            self.skills.set_dynamics(self.computer_id, mass=0.2)
            self.skills.set_object_to_track(self.bottle_id)
            
            self.skills.move_joint("l_shoulder_pitch", -1.6, wait=True)
            
            self.skills.track_target_with_head()
            self.skills.move_joint("r_hand_thumb_0_joint", 1.209, wait=True)
            bottle_pos = self.client.getBasePositionAndOrientation(self.bottle_id)[0]
            self.client.move_cartesian(Pose([bottle_pos[0] + 0.03, bottle_pos[1]  + 0.005, bottle_pos[2] + 0.06],  
                                            [-0.02655094675719738, -0.10425235331058502, -0.4081196188926697, 0.9065676331520081]))
            while not self.client.motion_done(check_collision=False):
                self.client.update_simulation()
                self.skills.track_target_with_head()
            
            # Grasp
            fingers = {"thumb": 0, "index": 0, "middle": 0, "ring": 0, "little": 0}
            for f in fingers:
                fingers[f] = self.skills.flex_finger_to_contact(self.bottle_id, f)
            self.skills.secure_grasp(self.bottle_id)
            
            if sum([fingers["index"], fingers["middle"], fingers["ring"], fingers["little"], fingers["thumb"]]) < 5:
                self.client.logger.error("Failed to grab the bottle with sufficient fingers.")
                self.valid_grasp = False
                return  # EARLY EXIT: Stops the script from trying to lift an empty hand

            # If it passes, lock the constraints and mark as valid
            self.valid_grasp = True
            
            # Lift
            for _ in range(9):
                self.skills.move_ee_relative(dz=0.01, velocity=10)
                self.skills.track_target_with_head()
                
            # Move
            target_pos = self.client.getBasePositionAndOrientation(self.computer_id)[0]
            self.skills.pose_adjustment()
            self.target_final_pos = [target_pos[0] + 0.25, target_pos[1] , target_pos[2]]
            self.skills.align_ee_xy_to_target(self.target_final_pos)
            
            # Align and Descend
            self.skills.level_object_to_table(self.bottle_id, target_pos)

            # Drop and Run
            self.skills.release_grasp()
            for _ in range(240):
                self.client.update_simulation()
            self.client.stretch_fingers()
            for _ in range(11):
                self.skills.move_ee_relative(dz=0.01, velocity=20)

    def check_success(self) -> bool:
        if not getattr(self, 'valid_grasp', False):
            return False
        bottle_pos = self.client.getBasePositionAndOrientation(self.bottle_id)[0]
        if np.linalg.norm(np.array(bottle_pos) - np.array(self.target_final_pos)) < 0.01:
            self.client.logger.error("Bottle not in correct position.")
            return False
        return True