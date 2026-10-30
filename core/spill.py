"""Spill geometry shared by data collection and inference cameras."""

import numpy as np
import pybullet as p


def sample_spill(seed=None):
    """Sample the same clustered puddles used by the T5 expert."""
    specifications = {}
    # 1. Define your strict bounding box
    x_min, x_max = -0.441, -0.22
    y_min, y_max = -0.011, 0.119
    z = -0.1210110063463354 
    
    n_particles = 20
    
    # Collection remains random by default; evaluation can supply a fixed seed.
    rng = np.random.default_rng(seed) 

    # 3. Create 1 to 3 random "puddle centers" inside the bounding box
    num_puddles = rng.integers(1, 4)
    puddles = []
    for _ in range(num_puddles):
        # Keep puddle centers slightly inside the box so they don't hug the edges purely
        cx = rng.uniform(x_min + 0.02, x_max - 0.02)
        cy = rng.uniform(y_min + 0.02, y_max - 0.02)
        # Random spread for this specific puddle
        sx = rng.uniform(0.02, 0.05)
        sy = rng.uniform(0.02, 0.05)
        puddles.append((cx, cy, sx, sy))

    placed = 0
    attempts = 0
    while placed < n_particles and attempts < 2000:
        attempts += 1

        # 4. Pick a random puddle and scatter a particle around it
        cx, cy, sx, sy = rng.choice(puddles)
        x = rng.normal(cx, sx)
        y = rng.normal(cy, sy)

        # 5. REJECTION SAMPLING: If the particle falls outside your square, throw it away and try again
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            continue

        # Randomize particle size
        particle_r = rng.uniform(0.008, 0.025)
        thickness  = rng.uniform(0.001, 0.004)
        name       = f"spill_{placed}"

        blue  = rng.uniform(0.7, 1.0)
        green = rng.uniform(0.3, 0.55)
        rgba = [0.0, green, blue, 1.0]

        specifications[name] = {
            "pos": [float(x), float(y), z],
            "radius": float(particle_r),
            "thickness": float(thickness),
            "rgba": [float(value) for value in rgba],
            "deleted_step": -1,
        }
        placed += 1
    if placed != n_particles:
        raise RuntimeError("Could not generate all 20 spill particles within collection bounds")
    return specifications


def spawn_spill(client, specifications=None):
    """Create PyBullet visuals/collisions and matching optional GUI meshes."""
    specifications = sample_spill() if specifications is None else specifications
    particles = {}
    for name, spec in specifications.items():
        visual = p.createVisualShape(
            p.GEOM_CYLINDER, radius=spec["radius"], length=spec["thickness"],
            rgbaColor=spec["rgba"], physicsClientId=client._client)
        collision = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=spec["radius"], height=spec["thickness"],
            physicsClientId=client._client)
        body = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual, basePosition=spec["pos"],
            physicsClientId=client._client)
        if getattr(client, "visualizer", None):
            import open3d as o3d
            try:
                mesh = o3d.geometry.TriangleMesh.create_cylinder(
                    radius=spec["radius"], height=spec["thickness"], resolution=20)
                mesh.translate(spec["pos"])
                client.visualizer.add_primitive(name, mesh, color=spec["rgba"][:3])
            except AttributeError:
                pass
        particles[name] = {"body_id": body, "pos": tuple(spec["pos"])}
    return particles, specifications
