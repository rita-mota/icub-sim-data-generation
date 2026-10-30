from tasks.base_task import BaseTask
from icub_pybullet.utils import Pose
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R
import open3d as o3d

class T5Cleaning(BaseTask):
    CONFIG_PATH = "configs/t5.yaml"
    BASE_PROMPT = "Clean the spill"

    def object_randomizer(self):
        return super().object_randomizer()
    
    def setup_environment(self):
        import open3d as o3d

        center_x, center_y = -0.35, -0.15
        z = -0.1210110063463354
        n_particles = 40  # more particles = fuller spill

        self.spill_particles = {}

        # Blob shape: rejection sample from an ellipse with some noise
        rng = np.random.default_rng(42)  # fixed seed = same spill every run
        placed = 0
        attempts = 0

        while placed < n_particles and attempts < 2000:
            attempts += 1
            # Sample within an ellipse (wider in x than y for a natural spill shape)
            angle  = rng.uniform(0, 2 * np.pi)
            radius = rng.beta(2, 1.5)  # beta biases toward center, thins at edges
            x = center_x + radius * 0.12 * np.cos(angle)  # 12cm half-width
            y = center_y + radius * 0.07 * np.sin(angle)  # 7cm half-height
            
            # Add per-particle jitter so edges look ragged
            x += rng.normal(0, 0.008)
            y += rng.normal(0, 0.008)

            # Vary radius: bigger in center, smaller at edges
            particle_r = rng.uniform(0.012, 0.025) * (1.0 - radius * 0.4)
            particle_r = max(0.008, particle_r)

            name = f"spill_{placed}"

            # PyBullet body for contact detection
            col_shape = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=particle_r, height=0.003,
                physicsClientId=self.client._client)
            body_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=-1,
                basePosition=[x, y, z],
                physicsClientId=self.client._client
            )

            # Open3D mesh — flat disc, slightly randomised height so they layer nicely
            thickness = rng.uniform(0.001, 0.004)
            mesh = o3d.geometry.TriangleMesh.create_cylinder(
                radius=particle_r, height=thickness, resolution=20)
            T = np.eye(4)
            T[:3, 3] = [x, y, z]
            mesh.transform(T)

            # Slight colour variation: deep blue core, lighter/greener at edges
            blue  = rng.uniform(0.7, 1.0)
            green = rng.uniform(0.3, 0.55) * (1.0 - radius * 0.3)
            self.client.visualizer.add_primitive(name, mesh,
                                                color=[0.0, green, blue])

            self.spill_particles[name] = {"body_id": body_id, "pos": (x, y, z)}
            placed += 1

        print(f"[T5] Spawned {len(self.spill_particles)} spill particles.")
                
    def execute_expert(self):
        return

    def check_success(self) -> bool:
        # bottle_pos = self.client.getBasePositionAndOrientation(self.bottle_id)[0]
        # if np.linalg.norm(np.array(bottle_pos) - np.array(self.target_final_pos)) < 0.01:
        #     self.client.logger.error("Bottle not in correct position.")
        #     return False
        return False