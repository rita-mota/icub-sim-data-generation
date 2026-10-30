import time
import numpy as np
import pybullet as p
from icub_pybullet.utils import Pose

from enum import Enum, auto

class GripperState(Enum):
    OPEN = auto()
    CLOSING = auto()
    HOLDING = auto()
    OPENING = auto()
    CLOSED_EMPTY = auto()

class RobotSkills:
    """
    A library of high-level, reusable robot actions.
    Separates low-level PyBullet physics math from the task logic.
    """
    GRIPPER_CONTROLLER_PROFILES = {
        "default": {
            "required_fingers": 3,
            "contact_confirmation_steps": 3,
            "closure": 1.0,
            "closing_duration": 1.5,
            "opening_duration": 1.0,
            "closing_velocity": 3.0,
            "opening_velocity": 5.0,
            "thumb_opposition_target": None,
        },
        # Existing T2 recordings complete each measured hand transition in
        # approximately five 5 Hz actions (one second).
        "t2_legacy": {
            "required_fingers": 3,
            "contact_confirmation_steps": 3,
            "closure": 1.0,
            "closing_duration": 1.0,
            "opening_duration": 1.0,
            "closing_velocity": 1.5,
            "opening_velocity": 3.0,
            "thumb_opposition_target": 1.209,
        },
        # Used by both T2NewSmoothMoving collection and VLA replay.
        "t2_new_smooth": {
            "required_fingers": 3,
            "contact_confirmation_steps": 3,
            "closure": 1.0,
            "closing_duration": 1.5,
            "opening_duration": 1.0,
            "closing_velocity": 1.5,
            "opening_velocity": 3.0,
            "thumb_opposition_target": 1.209,
        },
    }

    def __init__(self, client):
        self.client = client
        self.active_constraints = []
        self.object_tracked = 0
        
        self.gripper_state = None
        self.last_gripper_command = None
        
        self.gripper_hand = "right"  # Default hand for gripper actions
        self.gripper_joint_handles = {}
        self.gripper_open_targets = {}
        self.gripper_closed_targets = {}
        
        self.gripper_candidate_object_ids  =[]
        self.gripper_required_fingers = 3
        self.gripper_contact_confirmation_steps = 3
        self.gripper_open_tolerance = 0.03
        self.gripper_closed_tolerance = 0.05
        
        self.gripper_contact_candidate = None
        self.gripper_contact_steps = 0
        self.gripper_contacting_fingers = set()
        
        self.gripper_constraint_id = None
        self.gripper_constraint_ids = []
        self.gripper_held_object_id = None

        # The task demonstrations close at 60 Hz for 1.5 s and step the
        # 240 Hz simulator four times between successive finger targets.
        self.gripper_control_hz = 60.0
        self.gripper_closing_duration = 1.5
        self.gripper_opening_duration = 1.0
        self.gripper_physics_substeps = 4
        self.gripper_closing_velocity = 3.0
        self.gripper_opening_velocity = 5.0
        self.gripper_profile_name = "default"
        self.gripper_closing_step = 0
        self.gripper_closing_steps = 1
        self.gripper_closing_starts = {}
        self.gripper_opening_step = 0
        self.gripper_opening_steps = 1
        self.gripper_opening_starts = {}
        self.gripper_finger_joint_names = {}
        self.gripper_tip_link_ids = {}
        self.gripper_finger_tip_link_ids = {}
        self.gripper_constraint_by_finger = {}
        self.gripper_frozen_fingers = set()
        
        self.gripper_thumb_opposition_joint_name = None
        self.gripper_thumb_opposition_handle = None
        self.gripper_thumb_opposition_target = None
        
        
    # ---------------------------------------------------------
    # DYNAMICS & ENVIRONMENT
    # ---------------------------------------------------------
    def set_dynamics(self, obj_id, mass=0.2, lat_fric=1.7, roll_fric=0.0, restitution=0.0, spin_fric=0.0, link_id=-1):
        """Quickly adjusts the physics properties of an object."""
        p.changeDynamics(obj_id, link_id, mass=mass, lateralFriction=lat_fric, 
                         rollingFriction=roll_fric, spinningFriction=spin_fric, restitution=restitution)

    # ---------------------------------------------------------
    # HEAD & TRACKING
    # ---------------------------------------------------------
    def set_object_to_track(self, target_id):
        self.object_tracked = target_id
        
    def track_target_with_head(self):
        """Calculates 3D vectors to point the robot's head at an object."""
        if not self.object_tracked:
            print("No object defined to track")
            return
        target_pos = np.array(p.getBasePositionAndOrientation(self.object_tracked)[0])
        head_joint = self.client.get_joint_handle("neck_yaw")
        head_pos = np.array(p.getLinkState(self.client.robot, head_joint.robot_joint_id)[0])

        direction_vector = target_pos - head_pos
        v_x, v_y, v_z = direction_vector[0], direction_vector[1], direction_vector[2]

        yaw_angle = np.clip(-np.arctan2(v_y, -v_x), -0.8726, 0.8726)
        pitch_angle = np.clip(-np.arctan2(-v_z, np.linalg.norm([v_x, v_y])), -0.6981, 0.38)

        self.client.move_position("neck_yaw", yaw_angle, wait=False, velocity=5)
        self.client.move_position("neck_pitch", pitch_angle, wait=False, velocity=5)

    # ---------------------------------------------------------
    # KINEMATICS & MOVEMENT
    # ---------------------------------------------------------
    def move_joint(self, joint_name, position, wait=False, velocity=5.0, timeout=10.0):
        """Moves a specific joint (e.g., shoulder, wrist, thumb)."""
        self.client.move_position(joint_name, position, wait=False, 
                                  check_collision=False, velocity=velocity, timeout=timeout)
        if wait:
            while not self.client.motion_done(check_collision=False):
                self.client.update_simulation()
                self.track_target_with_head()

    def move_ee_relative(self, dx=0.0, dy=0.0, dz=0.0, velocity=5.0, timeout=1.0):
        """Shifts the end-effector by a relative distance while maintaining current orientation."""
        curr_pose = self.client.end_effector.get_position()
        new_pos = [curr_pose.pos[0] + dx, curr_pose.pos[1] + dy, curr_pose.pos[2] + dz]
        self.client.move_cartesian(Pose(new_pos, curr_pose.ori), velocity=velocity, timeout=timeout, check_collision=False)
        if self.object_tracked:
            self.track_target_with_head()       

    def rotate_ee_local(self, roll=0.0, pitch=0.0, yaw=0.0, velocity=5.0):
        """Rotates the end-effector relative to its current orientation (e.g., twisting the wrist)."""
        ee_state = p.getLinkState(self.client.robot, self.client.end_effector.link_id)
        ee_pos, ee_ori = ee_state[0], ee_state[1]
        
        delta_quat = p.getQuaternionFromEuler([roll, pitch, yaw])
        new_ori = p.multiplyTransforms([0, 0, 0], ee_ori, [0, 0, 0], delta_quat)[1]
        
        self.client.move_cartesian(Pose(pos=ee_pos, ori=new_ori), check_collision=False, velocity=velocity)

    def align_ee_xy_to_target(self, target_pos, precision=0.005, step_size=0.02):
        """
        Smoothly glides the end-effector over a target in the XY plane, keeping Z locked.
        Equivalent to Phase 5 of the T1 expert algorithm.
        """
        
        target_xy = np.array([target_pos[0] + 0.055, target_pos[1] + 0.06])
        
        for _ in range(200):
            if self.object_tracked:
                self.track_target_with_head()
            self.client.update_simulation()
            
            curr_pose = self.client.end_effector.get_position()
            curr_xy = np.array([curr_pose.pos[0], curr_pose.pos[1]])
            
            error_xy = target_xy - curr_xy
            distance = np.linalg.norm(error_xy)
            
            print(f"XY Distance to object: {distance:.4f}")
            
            if distance <= precision:
                return True
                
            dir_unit = error_xy / distance
            new_pos = [curr_pose.pos[0] + dir_unit[0] * step_size, 
                       curr_pose.pos[1] + dir_unit[1] * step_size, 
                       curr_pose.pos[2]]  # Z is locked
                       
            self.client.move_cartesian(Pose(new_pos, curr_pose.ori), velocity=5, timeout=1)
        return False

    def level_object_to_table(self, obj_id, target_pos, drop_velocity=0.05, timeout=5.0, extra = 0.0):
        """Pivots the hand around the center of the held object so its base is flat against the table."""
        
        print("Aligning object base with the table plane...")
            
        # 1. Get current states of both the object and the hand
        object_pos, object_orn = p.getBasePositionAndOrientation(obj_id)

        current_pose = self.client.end_effector.get_position()
        hand_pos = current_pose.pos
        hand_orn = current_pose.ori

        # 2. Calculate the Target object Orientation (Flat)
        # We read its current angles, force Roll and Pitch to 0, and keep its Yaw.
        object_euler = p.getEulerFromQuaternion(object_orn)
        flat_object_orn = p.getQuaternionFromEuler([0, 0, object_euler[2]])

        # 3. Find the hand's local position RELATIVE to the object
        # This answers: "Where is the hand from the object's perspective?"
        inv_object_pos, inv_object_orn = p.invertTransform(object_pos, object_orn)
        local_hand_pos, local_hand_orn = p.multiplyTransforms(
            inv_object_pos, inv_object_orn, 
            hand_pos, hand_orn
        )

        # 4. Calculate the new World Pose for the hand
        # We apply the 'flat' orientation to the object, and multiply it by the local hand offset.
        # This pivots the hand perfectly around the center of the object.
        new_hand_pos, new_hand_orn = p.multiplyTransforms(
            object_pos, flat_object_orn, 
            local_hand_pos, local_hand_orn
        )

        # 5. Execute the leveling movement
        # We use a slow velocity so the physics engine can stabilize the object during the twist
        self.client.move_cartesian(Pose(new_hand_pos, new_hand_orn), velocity=0.05, timeout=5.0)

        while not self.client.motion_done(check_collision=False):
            if self.object_tracked:
                self.track_target_with_head()
            self.client.update_simulation()
            time.sleep(0.01)  
        
        print("Object leveled. Ready for descent.")
        
        while p.getBasePositionAndOrientation(obj_id)[0][2] > target_pos[2] + extra:
            current_pos = self.client.end_effector.get_position()
            new_pos = [current_pos.pos[0], current_pos.pos[1], current_pos.pos[2] - 0.005]
            
            # Enforce the leveled orientation while dropping Z
            self.client.move_cartesian(Pose(new_pos, new_hand_orn), timeout=1)
            if self.object_tracked:
                self.track_target_with_head()
            self.client.update_simulation()

    def move_cartesian_smoothly(self, target_pose, steps=50, timeout=2.0):
        """Moves to a Cartesian pose in small, interpolated chunks to prevent IK flipping."""
        current_pose = self.client.end_effector.get_pose()
        
        # Calculate the step size for each axis
        dx = (target_pose.pos[0] - current_pose.pos[0]) / steps
        dy = (target_pose.pos[1] - current_pose.pos[1]) / steps
        dz = (target_pose.pos[2] - current_pose.pos[2]) / steps
        
        # NOTE: For perfect smoothness, you would also use Slerp for orientation, 
        # but applying the target orientation immediately usually works fine 
        # if the positional steps are small.
        
        for i in range(1, steps + 1):
            interp_pos = [
                current_pose.pos[0] + (dx * i),
                current_pose.pos[1] + (dy * i),
                current_pose.pos[2] + (dz * i)
            ]
            
            # Feed the intermediate target to the client
            self.client.move_cartesian(Pose(interp_pos, target_pose.ori), timeout=timeout/steps, check_collision = False)
            self.client.update_simulation()
    
    def pose_adjustment(self, wait=True, velocity=3.0):
        """
        Moves the right arm into a manually verified, natural configuration 
        to bypass awkward IK solver results.
        """
        natural_angles = {
            "r_shoulder_pitch": -0.891,
            "r_shoulder_roll": 1.241,
            "r_shoulder_yaw": 1.054,
            "r_elbow": 1.846,
            "r_wrist_prosup": -0.959,
            "r_wrist_yaw": 0.005
        }
        
        # Send all joint commands simultaneously (wait=False)
        for joint_name, target_angle in natural_angles.items():
            self.client.move_position(
                joint_name, 
                target_angle, 
                wait=False, 
                check_collision=False, 
                velocity=velocity,
                timeout = 2
            )
            self.track_target_with_head()
            
        # Wait for all joints to finish moving
        if wait:
            while not self.client.motion_done(check_collision=False):
                self.client.update_simulation()
    
    def open_hand(self, hand="right"):
            """Fully extends all fingers to create a flat hand posture for either arm."""
            hand_prefix = "r" if hand == "right" else "l"
            fingers = ["thumb", "index", "middle", "ring", "little"]
            for f in fingers:
                self.move_joint(f"{hand_prefix}_hand_{f}_2_joint", 0.0)
                self.move_joint(f"{hand_prefix}_hand_{f}_3_joint", 0.0)
                if f == "thumb":
                    self.move_joint(f"{hand_prefix}_hand_{f}_1_joint", 0.0)
            self.client.update_simulation()
            
    def open_thumb(self, hand="right"):
            """Fully extends the thumb to create a flat hand posture for either arm."""
            hand_prefix = "r" if hand == "right" else "l"
            f = "thumb"
            self.move_joint(f"{hand_prefix}_hand_{f}_2_joint", 0.0)
            self.move_joint(f"{hand_prefix}_hand_{f}_3_joint", 0.0)
            self.move_joint(f"{hand_prefix}_hand_{f}_1_joint", 0.0)
            self.client.update_simulation()

    def get_push_alignment(self, box_id, target_xy):
        """
        Calculates the best face to push based on a target, returning the 
        face normal, face name, hand orientation quaternion, and box state.
        """
        from scipy.spatial.transform import Rotation as R
        
        box_pos, box_orn = p.getBasePositionAndOrientation(box_id)
        box_xy = np.array(box_pos[:2]) + 0.05 # Center adjustment
        
        push_vec = target_xy - box_xy
        push_vec_norm = push_vec / np.linalg.norm(push_vec) 
        
        rot_matrix = np.array(p.getMatrixFromQuaternion(box_orn)).reshape(3, 3)
        local_x = rot_matrix[:2, 0] 
        local_y = rot_matrix[:2, 1] 
        
        face_normals = [
            (local_x, "left"),
            (-local_x, "right"),
            (local_y, "front"),
            (-local_y, "back")
        ]
        
        best_face_normal = None
        best_face_name = ""
        min_dot = float('inf') 
        
        for normal, name in face_normals:
            dot_product = np.dot(normal, push_vec_norm)
            if dot_product < min_dot:
                min_dot = dot_product
                best_face_normal = normal
                best_face_name = name

        R_world_box = R.from_quat(box_orn).as_matrix()
        R_box_hand = np.eye(3,3)
        
        # Map Hand's local frame to the Box's faces
        if best_face_name == "left":
            R_box_hand = np.array([[ 0,  1,  0], [-1,  0,  0], [ 0,  0,  1]])
        elif best_face_name == "front":
            R_box_hand = np.array([[ 1,  0,  0], [ 0,  1,  0], [ 0,  0,  1]])
        elif best_face_name == "right":
            R_box_hand = np.array([[ 0, -1,  0], [ 1,  0,  0], [ 0,  0,  1]])
        elif best_face_name == "back":
            R_box_hand = np.array([[-1,  0,  0], [ 0, -1,  0], [ 0,  0,  1]])

        R_world_hand = R_world_box @ R_box_hand
        push_orn = R.from_matrix(R_world_hand).as_quat()
        
        return best_face_normal, best_face_name, push_orn, box_pos, box_xy
    
    def move_joint_relative(self, joint_name, delta_angle, wait=False, velocity=5.0, timeout=10.0):
        """
        Adds a specific angle (in radians) to the joint's current position.
        Automatically clips the command to safety limits to prevent motor jamming.
        """
        # 1. Look up the joint handle to get limits and the internal PyBullet ID
        joint_handle = self.client.get_joint_handle(joint_name)
        joint_id = joint_handle.robot_joint_id
        
        # 2. Query PyBullet for the joint's exact current physical angle
        current_position = p.getJointState(
            self.client.robot, 
            joint_id, 
            physicsClientId=self.client._client
        )[0]
        
        # 3. Calculate target and clip it strictly within the URDF boundaries
        target_position = current_position + delta_angle
        target_position = np.clip(
            target_position, 
            joint_handle.lower_limit, 
            joint_handle.upper_limit
        )
        
        # 4. Command the absolute position using your existing movement logic
        self.move_joint(
            joint_name, 
            target_position, 
            wait=wait, 
            velocity=velocity, 
            timeout=timeout
        )
        
    # ---------------------------------------------------------
    # GRASPING & CONTACT LOGIC
    # ---------------------------------------------------------
    def flex_finger_to_contact(self, obj_id, finger_name, closure=1.0, hand="right"):
        """Flexes joints 2 and 3 of a specific finger until contact is detected."""
        hand_prefix = "r" if hand == "right" else "l"
        moving_joints = [f"{hand_prefix}_hand_{finger_name}_3_joint", f"{hand_prefix}_hand_{finger_name}_2_joint"]
        tip_handle = self.client.get_joint_handle(f"{hand_prefix}_hand_{finger_name}_3_joint")
        
        locked = False
        for _ in range(500):
            contacts = p.getContactPoints(bodyA=self.client.robot, bodyB=obj_id)
            contacting_links = [c[3] for c in contacts]

            if not locked and tip_handle.robot_joint_id in contacting_links:
                locked = True
                break

            if not locked:
                # if finger_name == "thumb":
                #     moving_joints = moving_joints[::-1]
                for j_name in moving_joints:
                    handle = self.client.get_joint_handle(j_name)
                    target = handle.lower_limit + closure * (handle.upper_limit - handle.lower_limit)
                    self.client.move_position(j_name, target, wait=False, velocity=3, timeout=2)
            
            self.client.update_simulation()
            time.sleep(0.01)
        return locked

    def close_finger(self, finger_name, hand = "right", timeout = 5):
        hand = "r" if hand == "right" else "l"
        joints = [f"{hand}_hand_{finger_name}_2_joint", f"{hand}_hand_{finger_name}_3_joint"]
        
        # Move all the joints    
        for joint in joints:
            joint_handle = self.client.get_joint_handle(joint)
            self.client.move_position(joint, joint_handle.upper_limit, wait=False,
                                        check_collision=False, timeout=timeout, velocity=5)
    
    def secure_grasp(self, obj_id, hand="right"):
        """Creates fixed constraints between touching fingertips and the object."""
        hand_prefix = "r" if hand == "right" else "l"
        fingertip_joints = [f"{hand_prefix}_hand_{f}_3_joint" for f in ["thumb","index", "middle", "ring", "little"]]
        contacts = p.getContactPoints(bodyA=self.client.robot, bodyB=obj_id)
        
        created_constraints = []

        for joint_name in fingertip_joints:
            handle = self.client.get_joint_handle(joint_name)
            joint_id = handle.robot_joint_id
            
            for contact in contacts:
                if contact[3] == joint_id:
                    obj_pos, obj_orn = p.getBasePositionAndOrientation(obj_id)
                    link_pos, link_orn = p.getLinkState(self.client.robot, joint_id)[:2]
                    
                    inv_link_pos, inv_link_orn = p.invertTransform(link_pos, link_orn)
                    local_obj_pos, local_obj_orn = p.multiplyTransforms(inv_link_pos, inv_link_orn, obj_pos, obj_orn)

                    c_id = p.createConstraint(
                        parentBodyUniqueId=self.client.robot, parentLinkIndex=joint_id, childBodyUniqueId=obj_id,
                        childLinkIndex=-1, jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                        parentFramePosition=local_obj_pos, childFramePosition=[0, 0, 0], parentFrameOrientation=local_obj_orn
                    )
                    self.active_constraints.append(c_id)
                    created_constraints.append(c_id)
                    break 

        return created_constraints
    
    def secure_grasp_all_contacts(self, obj_id, hand="right"):
        """Creates fixed constraints between ALL touching hand links and the object."""
        hand_prefix = "r" if hand == "right" else "l"
        hand_string = f"{hand_prefix}_hand_"
        
        # 1. Identify all link IDs that belong to this specific hand
        hand_link_ids = set()
        for joint in self.client.joints:
            if hand_string in joint.name:
                hand_link_ids.add(joint.robot_joint_id)
                
        # 2. Get every single contact point between the robot and the object
        contacts = p.getContactPoints(bodyA=self.client.robot, bodyB=obj_id)
        
        # Keep track of links we have already constrained. 
        # (PyBullet often returns 3 or 4 contact points for a single finger pad. 
        # We only need one constraint per physical link to lock it.)
        constrained_links = set()
        
        for contact in contacts:
            link_id = contact[3] # Index 3 is the link index on bodyA (the robot)
            
            # 3. If the contacting link is part of the hand AND hasn't been locked yet
            if link_id in hand_link_ids and link_id not in constrained_links:
                
                # Calculate relative transform to freeze it exactly in place
                obj_pos, obj_orn = p.getBasePositionAndOrientation(obj_id)
                link_pos, link_orn = p.getLinkState(self.client.robot, link_id)[:2]
                
                inv_link_pos, inv_link_orn = p.invertTransform(link_pos, link_orn)
                local_obj_pos, local_obj_orn = p.multiplyTransforms(inv_link_pos, inv_link_orn, obj_pos, obj_orn)

                # Create the fixed constraint
                c_id = p.createConstraint(
                    parentBodyUniqueId=self.client.robot, 
                    parentLinkIndex=link_id, 
                    childBodyUniqueId=obj_id,
                    childLinkIndex=-1, 
                    jointType=p.JOINT_FIXED, 
                    jointAxis=[0, 0, 0],
                    parentFramePosition=local_obj_pos, 
                    childFramePosition=[0, 0, 0], 
                    parentFrameOrientation=local_obj_orn
                )
                
                self.active_constraints.append(c_id)
                constrained_links.add(link_id)
                
        print(f"Secured grasp using {len(constrained_links)} contact points.")
                 
    def release_grasp(self):
        """Removes all active finger constraints."""
        for c_id in self.active_constraints:
            p.removeConstraint(c_id)
            print(f"Releasing contraint: {c_id}")
        self.active_constraints.clear()
    
    def attach_sponge_to_palm(self, obj_id, table_z):
        """
        Directly constrains the sponge to the wrist link based on current
        transforms — no contact detection needed.
        """
        palm_link = self.client.get_joint_handle("r_wrist_yaw")
        palm_id   = palm_link.robot_joint_id

        palm_pos, palm_orn = p.getLinkState(
            self.client.robot, palm_id,
            physicsClientId=self.client._client)[:2]
        obj_pos, obj_orn = p.getBasePositionAndOrientation(
            obj_id, physicsClientId=self.client._client)

        # Compute sponge position in palm's local frame
        inv_palm_pos, inv_palm_orn = p.invertTransform(palm_pos, palm_orn)
        local_obj_pos, local_obj_orn = p.multiplyTransforms(
            inv_palm_pos, inv_palm_orn, obj_pos, obj_orn)

        c_id = p.createConstraint(
            parentBodyUniqueId=self.client.robot,
            parentLinkIndex=palm_id,
            childBodyUniqueId=obj_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=local_obj_pos,
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=local_obj_orn,
            physicsClientId=self.client._client
        )
        p.changeConstraint(c_id, maxForce=200,
                        physicsClientId=self.client._client)
        self.active_constraints.append(c_id)
        print(f"[Skills] Sponge directly attached to palm (local_pos={local_obj_pos})")
    
    def disable_robot_object_collision(self, obj_id):
        """Disables collision between every robot link and the given object."""
        num_joints = p.getNumJoints(
            self.client.robot, physicsClientId=self.client._client)
        
        # Base link (-1)
        p.setCollisionFilterPair(
            self.client.robot, obj_id, -1, -1,
            enableCollision=0,
            physicsClientId=self.client._client)
        
        # Every joint link
        for i in range(num_joints):
            p.setCollisionFilterPair(
                self.client.robot, obj_id, i, -1,
                enableCollision=0,
                physicsClientId=self.client._client)
        
        print(f"[Skills] Disabled collision between robot and object {obj_id} "
            f"({num_joints + 1} link pairs).")
        
    # ---------------------------------------------------------
    # GRIPPER CONTROLLER/ STATE MACHINE
    # ---------------------------------------------------------
    
    def initialize_gripper_controller(
        self,
        candidate_object_ids,
        hand="right",
        required_fingers=None,
        contact_confirmation_steps=None,
        open_tolerance=0.03,
        closed_tolerance=0.05,
        closure=None,
        thumb_opposition_target=None,
        profile="default",
        closing_duration=None,
        opening_duration=None,
        initial_command=None,
    ):
        """
        Initialize the non-blocking gripper controller.

        candidate_object_ids:
            Objects that the hand is allowed to grasp. The table and robot
            should not be included.

        closure:
            Fraction of each joint range used as the closing target.

        profile:
            Named calibration shared by collection and replay. Explicit
            keyword arguments override the selected profile.
        """
        
        if profile not in self.GRIPPER_CONTROLLER_PROFILES:
            available = ", ".join(self.GRIPPER_CONTROLLER_PROFILES)
            raise ValueError(
                f"Unknown gripper profile {profile!r}; choose from {available}."
            )

        settings = self.GRIPPER_CONTROLLER_PROFILES[profile]
        required_fingers = (
            settings["required_fingers"]
            if required_fingers is None else required_fingers
        )
        contact_confirmation_steps = (
            settings["contact_confirmation_steps"]
            if contact_confirmation_steps is None
            else contact_confirmation_steps
        )
        closure = settings["closure"] if closure is None else closure
        closing_duration = (
            settings["closing_duration"]
            if closing_duration is None else closing_duration
        )
        opening_duration = (
            settings["opening_duration"]
            if opening_duration is None else opening_duration
        )
        if thumb_opposition_target is None:
            thumb_opposition_target = settings["thumb_opposition_target"]

        if hand not in ["right", "left"]:
            raise ValueError(f"Unsupported hand: {hand}.")
        
        if not 0.0 <= closure <= 1.0:
            raise ValueError(f"Closure must be between 0 and 1, got {closure}.")
        if required_fingers < 1:
            raise ValueError("required_fingers must be at least 1.")
        if contact_confirmation_steps < 1:
            raise ValueError("contact_confirmation_steps must be at least 1.")
        if closing_duration <= 0.0 or opening_duration <= 0.0:
            raise ValueError("Gripper trajectory durations must be positive.")
        
        self.gripper_hand = hand
        self.gripper_profile_name = profile
        self.gripper_required_fingers = required_fingers
        self.gripper_contact_confirmation_steps = contact_confirmation_steps
        self.gripper_open_tolerance = open_tolerance
        self.gripper_closed_tolerance = closed_tolerance
        self.gripper_closing_duration = float(closing_duration)
        self.gripper_opening_duration = float(opening_duration)
        self.gripper_closing_velocity = float(
            settings["closing_velocity"]
        )
        self.gripper_opening_velocity = float(
            settings["opening_velocity"]
        )
        
        self.gripper_candidate_object_ids = [
            int(obj_id) 
            for obj_id in candidate_object_ids
            if obj_id is not None
        ]
        
        self.gripper_joint_handles.clear()
        self.gripper_open_targets.clear()
        self.gripper_closed_targets.clear()
        self.gripper_finger_joint_names.clear()
        self.gripper_tip_link_ids.clear()
        self.gripper_finger_tip_link_ids.clear()
        
        hand_prefix = "r" if hand == "right" else "l"
        fingers = ["thumb", "index", "middle", "ring", "little"]
        
        for finger in fingers:
            self.gripper_finger_joint_names[finger] = []
            for joint_num in (2,3):
                joint_name = (
                    f"{hand_prefix}_hand_{finger}_{joint_num}_joint"
                )
                handle = self.client.get_joint_handle(joint_name)
                
                lower = float(handle.lower_limit)
                upper = float(handle.upper_limit)
                
                open_target = float(np.clip(0.0, lower, upper))
                closed_target = lower + closure * (upper - lower)
                
                self.gripper_joint_handles[joint_name] = handle
                self.gripper_open_targets[joint_name] = open_target
                self.gripper_closed_targets[joint_name] = closed_target
                self.gripper_finger_joint_names[finger].append(joint_name)

                if joint_num == 3:
                    self.gripper_tip_link_ids[handle.robot_joint_id] = finger
                    self.gripper_finger_tip_link_ids[finger] = (
                        handle.robot_joint_id
                    )
        
        self.gripper_constraint_id = None
        self.gripper_constraint_ids = []
        self.gripper_constraint_by_finger = {}
        self.gripper_held_object_id = None
        self.gripper_contact_candidate = None
        self.gripper_contact_steps = 0
        self.gripper_contacting_fingers = set()
        self.gripper_frozen_fingers = set()
        self.gripper_closing_starts = {}
        self.gripper_closing_step = 0
        self.gripper_closing_steps = max(
            1,
            int(round(
                self.gripper_closing_duration * self.gripper_control_hz
            )),
        )
        self.gripper_opening_starts = {}
        self.gripper_opening_step = 0
        self.gripper_opening_steps = max(
            1,
            int(round(
                self.gripper_opening_duration * self.gripper_control_hz
            )),
        )
        
        self.gripper_thumb_opposition_joint_name = None
        self.gripper_thumb_opposition_handle = None
        self.gripper_thumb_opposition_target = None

        if thumb_opposition_target is not None:
            hand_prefix = "r" if hand == "right" else "l"

            joint_name = (
                f"{hand_prefix}_hand_thumb_0_joint"
            )
            handle = self.client.get_joint_handle(joint_name)

            self.gripper_thumb_opposition_joint_name = joint_name
            self.gripper_thumb_opposition_handle = handle
            self.gripper_thumb_opposition_target = float(
                np.clip(
                    thumb_opposition_target,
                    handle.lower_limit,
                    handle.upper_limit,
                )
            )
        if self.gripper_thumb_opposition_target is not None:
            self.client.move_position(
                self.gripper_thumb_opposition_joint_name,
                self.gripper_thumb_opposition_target,
                wait=False,
                check_collision=False,
                velocity=3.0,
                timeout=5.0,
            )
            
        if initial_command is not None:
            initial_command = self._decode_gripper_command(initial_command)
            self.gripper_state = (
                GripperState.OPEN
                if initial_command == 1
                else GripperState.CLOSED_EMPTY
            )
            self.last_gripper_command = initial_command
        elif self._is_open():
            self.gripper_state = GripperState.OPEN
            self.last_gripper_command = 1
        else:
            self.gripper_state = GripperState.CLOSED_EMPTY
            self.last_gripper_command = 0
        
        return self.gripper_state
    
    def _decode_gripper_command(self, value):
        """
        Decodes a gripper command into a target state.
        1 = Open
        0 = Close
        """
        value = float(np.clip(value, 0.0, 1.0))
        
        if value >= 0.7:
            return 1
        elif value <= 0.3:
            return 0
        
        # Hysteresis region: retain the preceding command.
        if self.last_gripper_command is not None:
            return self.last_gripper_command

        return 1
    
    def execute_gripper_pre_action(self, command):
        if self.gripper_state is None:
            raise RuntimeError("Gripper controller not initialized. Call initialize_gripper_controller() first.")
        
        command = self._decode_gripper_command(command)
        state_before = self.gripper_state
        
        if command == 0: #Close
            if self.gripper_state in {
                GripperState.OPEN, 
                GripperState.OPENING
            }:
                self._start_closing()
        
        # CLOSED_EMPTY + CLOSE is a no-op.
        # CLOSING + CLOSE is a no-op.
        # HOLDING + CLOSE is a no-op.
        
        else: #Open
            if self.gripper_state in {
                GripperState.CLOSED_EMPTY, 
                GripperState.HOLDING, 
                GripperState.CLOSING
            }:
                self._start_opening()
        # OPEN + OPEN is a no-op.
        # OPENING + OPEN is a no-op.

        self.last_gripper_command = command
        
        return {
            "command": command,
            "state_before": state_before.name,
            "state_after": self.gripper_state.name,
        }
        
    def execute_gripper_post_action(self, command):
        command = self._decode_gripper_command(command)
        state_before = self.gripper_state
        current_contacting_fingers = set()
        
        if self.gripper_state == GripperState.CLOSING:
            object_id, contacting_fingers = self._find_grasp_candidate()
            current_contacting_fingers = contacting_fingers

            if object_id is not None and (
                self.gripper_contact_candidate is None
                or object_id == self.gripper_contact_candidate
            ):
                self.gripper_contact_candidate = object_id
                for finger in contacting_fingers:
                    if finger not in self.gripper_frozen_fingers:
                        self._hold_gripper_finger(finger)
                        self.gripper_frozen_fingers.add(finger)
                        self._attach_grasp_finger(object_id, finger)

                self.gripper_contacting_fingers.update(
                    contacting_fingers
                )

            stable_grasp_contact = (
                object_id is not None
                and object_id == self.gripper_contact_candidate
                and len(self.gripper_constraint_by_finger)
                    >= self.gripper_required_fingers
            )
            if stable_grasp_contact:
                self.gripper_contact_steps += 1
            else:
                self.gripper_contact_steps = 0

            # In the task scripts the hand stays still throughout the blocking
            # closing loop, so constraints can be created at the end. During
            # non-blocking replay the arm may start lifting on the next action;
            # attach as soon as a multi-finger contact has been confirmed.
            if (
                stable_grasp_contact
                and self.gripper_contact_steps
                    >= self.gripper_contact_confirmation_steps
            ):
                if self._attach_grasp_object(object_id):
                    self.gripper_state = GripperState.HOLDING

            if self.gripper_state == GripperState.CLOSING:
                self._advance_gripper_closing()

                closing_finished = (
                    self.gripper_closing_step >= self.gripper_closing_steps
                    or len(self.gripper_frozen_fingers)
                        == len(self.gripper_finger_joint_names)
                )

                if closing_finished and not stable_grasp_contact:
                    self._release_grasp_object()
                    self.gripper_state = GripperState.CLOSED_EMPTY
        
        elif self.gripper_state == GripperState.OPENING:
            self._advance_gripper_opening()
            if self._is_open():
                self.gripper_state = GripperState.OPEN
        
        return {
            "command": command,
            "state_before": state_before.name,
            "state_after": self.gripper_state.name,
            "profile": self.gripper_profile_name,
            "closing_progress": (
                self.gripper_closing_step / self.gripper_closing_steps
            ),
            "opening_progress": (
                self.gripper_opening_step / self.gripper_opening_steps
            ),
            "contact_candidate": self.gripper_contact_candidate,
            "contacting_fingers": sorted(
                self.gripper_contacting_fingers
            ),
            "current_contacting_fingers": sorted(
                current_contacting_fingers
            ),
            "stable_contact_steps": self.gripper_contact_steps,
            "held_object_id": self.gripper_held_object_id,
            "constraint_id": self.gripper_constraint_id,
            "constraint_ids": list(self.gripper_constraint_ids),
        }
        
    def _start_closing(self):
        """Start a non-blocking, minimum-jerk finger-closing trajectory."""
        if self.gripper_thumb_opposition_target is not None:
            self.client.move_position(
                self.gripper_thumb_opposition_joint_name,
                self.gripper_thumb_opposition_target,
                wait=False,
                check_collision=False,
                velocity=3.0,
                timeout=5.0,
            )
        self.gripper_contact_candidate = None
        self.gripper_contact_steps = 0
        self.gripper_contacting_fingers = set()
        self.gripper_frozen_fingers = set()
        self.gripper_closing_step = 0
        self.gripper_closing_starts = {
            joint_name: float(p.getJointState(
                self.client.robot,
                handle.robot_joint_id,
                physicsClientId=self.client._client,
            )[0])
            for joint_name, handle in self.gripper_joint_handles.items()
        }

        self.gripper_state = GripperState.CLOSING

        # Give the Cartesian move a very small initial finger target. Further
        # targets are issued one controller tick at a time in post_action().
        self._advance_gripper_closing()

    @staticmethod
    def _minimum_jerk(progress):
        progress = float(np.clip(progress, 0.0, 1.0))
        return (
            10.0 * progress ** 3
            - 15.0 * progress ** 4
            + 6.0 * progress ** 5
        )

    def _advance_gripper_closing(self):
        if self.gripper_closing_step >= self.gripper_closing_steps:
            return

        self.gripper_closing_step += 1
        progress = (
            self.gripper_closing_step / self.gripper_closing_steps
        )
        blend = self._minimum_jerk(progress)

        for finger, joint_names in self.gripper_finger_joint_names.items():
            if finger in self.gripper_frozen_fingers:
                continue

            for joint_name in joint_names:
                start = self.gripper_closing_starts[joint_name]
                target = self.gripper_closed_targets[joint_name]
                interpolated_target = start + blend * (target - start)
                self.client.move_position(
                    joint_name,
                    interpolated_target,
                    wait=False,
                    check_collision=False,
                    velocity=self.gripper_closing_velocity,
                    timeout=1.0,
                )

    def _hold_gripper_finger(self, finger):
        """Freeze one finger at its measured contact configuration."""
        for joint_name in self.gripper_finger_joint_names[finger]:
            handle = self.gripper_joint_handles[joint_name]
            current_position = float(p.getJointState(
                self.client.robot,
                handle.robot_joint_id,
                physicsClientId=self.client._client,
            )[0])
            self.client.move_position(
                joint_name,
                current_position,
                wait=False,
                check_collision=False,
                velocity=1.0,
                timeout=1.0,
            )
    
    def _start_opening(self):
        """Release the object and start smooth non-blocking opening."""
        self._release_grasp_object()

        self.gripper_contact_candidate = None
        self.gripper_contact_steps = 0
        self.gripper_contacting_fingers = set()
        self.gripper_frozen_fingers = set()
        self.gripper_opening_step = 0
        self.gripper_opening_starts = {
            joint_name: float(p.getJointState(
                self.client.robot,
                handle.robot_joint_id,
                physicsClientId=self.client._client,
            )[0])
            for joint_name, handle in self.gripper_joint_handles.items()
        }
        self.gripper_state = GripperState.OPENING
        self._advance_gripper_opening()

    def _advance_gripper_opening(self):
        if self.gripper_opening_step >= self.gripper_opening_steps:
            return

        self.gripper_opening_step += 1
        progress = (
            self.gripper_opening_step / self.gripper_opening_steps
        )
        blend = self._minimum_jerk(progress)

        for joint_name, target in self.gripper_open_targets.items():
            start = self.gripper_opening_starts[joint_name]
            interpolated_target = start + blend * (target - start)
            self.client.move_position(
                joint_name,
                interpolated_target,
                wait=False,
                check_collision=False,
                velocity=self.gripper_opening_velocity,
                timeout=1.0,
            )
        
    def _gripper_joint_error(self, targets):
        errors = []

        for joint_name, target in targets.items():
            handle = self.gripper_joint_handles[joint_name]

            position = p.getJointState(
                self.client.robot,
                handle.robot_joint_id,
                physicsClientId=self.client._client,
            )[0]

            errors.append(abs(float(position) - target))

        return max(errors) if errors else float("inf")
    
    def _is_open(self):
        return (
            self._gripper_joint_error(self.gripper_open_targets)
            <= self.gripper_open_tolerance
        )
        
    def _is_closed(self):
        return (
            self._gripper_joint_error(self.gripper_closed_targets)
            <= self.gripper_closed_tolerance
        )
            
    def _find_grasp_candidate(self):
        """
        Return the object with the greatest number of contacting fingers.

        Returns:
            (object_id, set_of_finger_names)
        """
        best_object = None
        best_fingers = set()

        for object_id in self.gripper_candidate_object_ids:
            contacts = p.getContactPoints(
                bodyA=self.client.robot,
                bodyB=object_id,
                physicsClientId=self.client._client,
            )

            contacting_fingers = {
                self.gripper_tip_link_ids[contact[3]]
                for contact in contacts
                if contact[3] in self.gripper_tip_link_ids
            }

            if len(contacting_fingers) > len(best_fingers):
                best_object = object_id
                best_fingers = contacting_fingers

        return best_object, best_fingers
    
    def _attach_grasp_object(self, object_id):
        if object_id != self.gripper_contact_candidate:
            return False
        if (
            len(self.gripper_constraint_ids)
            < self.gripper_required_fingers
        ):
            return False

        # Retain the singular field for existing diagnostics/callers.
        self.gripper_constraint_id = self.gripper_constraint_ids[0]
        self.gripper_held_object_id = object_id

        return True

    def _attach_grasp_finger(self, object_id, finger):
        """Anchor one fingertip at the instant it first touches the object."""
        if finger in self.gripper_constraint_by_finger:
            return True

        if (
            self.gripper_contact_candidate is not None
            and object_id != self.gripper_contact_candidate
        ):
            return False

        link_id = self.gripper_finger_tip_link_ids[finger]
        link_pos, link_orn = p.getLinkState(
            self.client.robot,
            link_id,
            physicsClientId=self.client._client,
        )[:2]
        object_pos, object_orn = p.getBasePositionAndOrientation(
            object_id,
            physicsClientId=self.client._client,
        )
        inverse_link_pos, inverse_link_orn = p.invertTransform(
            link_pos,
            link_orn,
        )
        local_object_pos, local_object_orn = p.multiplyTransforms(
            inverse_link_pos,
            inverse_link_orn,
            object_pos,
            object_orn,
        )

        constraint_id = p.createConstraint(
            parentBodyUniqueId=self.client.robot,
            parentLinkIndex=link_id,
            childBodyUniqueId=object_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0.0, 0.0, 0.0],
            parentFramePosition=local_object_pos,
            parentFrameOrientation=local_object_orn,
            childFramePosition=[0.0, 0.0, 0.0],
            childFrameOrientation=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client._client,
        )

        self.gripper_constraint_by_finger[finger] = constraint_id
        self.gripper_constraint_ids.append(constraint_id)
        self.active_constraints.append(constraint_id)
        return True
    
    def _release_grasp_object(self):
        if not self.gripper_constraint_ids:
            self.gripper_constraint_id = None
            self.gripper_held_object_id = None
            return False

        constraint_ids = list(self.gripper_constraint_ids)
        for constraint_id in constraint_ids:
            p.removeConstraint(
                constraint_id,
                physicsClientId=self.client._client,
            )
            if constraint_id in self.active_constraints:
                self.active_constraints.remove(constraint_id)

        self.gripper_constraint_ids = []
        self.gripper_constraint_by_finger = {}
        self.gripper_constraint_id = None
        self.gripper_held_object_id = None

        return True
            
            
      
                
            
    
