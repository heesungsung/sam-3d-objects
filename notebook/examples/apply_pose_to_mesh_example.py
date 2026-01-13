"""
Example: How to apply pose estimation to mesh in a simulator

This script demonstrates how to use the pose estimation values and mesh file
to place the object in the correct position, rotation, and scale in a 3D simulator.
"""

import numpy as np
import json
import trimesh


def apply_pose_to_mesh(mesh_path: str, pose_json_path: str, output_path: str = None):
    """
    Apply pose transformation to mesh and optionally save the transformed mesh.
    
    Args:
        mesh_path: Path to PLY mesh file
        pose_json_path: Path to pose JSON file (with translation, rotation, scale)
        output_path: Optional path to save transformed mesh
    
    Returns:
        transformed_vertices: (N, 3) numpy array of transformed vertex positions
        mesh: Trimesh object with transformed vertices
    """
    # 1. Load mesh
    mesh = trimesh.load(mesh_path)
    vertices = np.array(mesh.vertices)  # (N, 3)
    
    # 2. Load pose information
    with open(pose_json_path, 'r') as f:
        pose = json.load(f)
    
    translation = np.array(pose['translation'])  # [x, y, z]
    rotation_quat = np.array(pose['rotation'])   # [w, x, y, z] (PyTorch3D convention)
    scale = np.array(pose['scale'])              # [sx, sy, sz]
    
    # 3. Apply transformation
    # Order: Scale -> Rotate -> Translate
    
    # 3.1 Scale
    vertices_scaled = vertices * scale  # Element-wise multiplication: (N, 3) * (3,)
    
    # 3.2 Rotate (quaternion to rotation matrix)
    # Convert quaternion [w, x, y, z] to rotation matrix
    w, x, y, z = rotation_quat
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]
    ])
    
    # Apply rotation: vertices @ R.T (because we want to rotate points, not the coordinate frame)
    vertices_rotated = vertices_scaled @ R.T
    
    # 3.3 Translate
    vertices_transformed = vertices_rotated + translation
    
    # 4. Update mesh
    mesh.vertices = vertices_transformed
    
    # 5. Optionally save transformed mesh
    if output_path:
        mesh.export(output_path)
        print(f"Transformed mesh saved to: {output_path}")
    
    return vertices_transformed, mesh


def quaternion_to_matrix(quat):
    """
    Convert quaternion [w, x, y, z] to 3x3 rotation matrix.
    
    Args:
        quat: [w, x, y, z] quaternion
        
    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = quat
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]
    ])


def convert_to_simulator_coordinates(vertices_camera_frame):
    """
    Convert vertices from camera coordinate frame to simulator coordinate frame.
    
    Note: This depends on your simulator's coordinate system!
    Common conversions:
    
    Camera frame (SAM3D): 
        - Origin: camera position [0, 0, 0]
        - X: right, Y: up, Z: viewing direction is -Z
    
    Unity/Unreal (typical):
        - X: right, Y: up, Z: forward
    
    PyBullet/Gazebo (typical):
        - X: forward, Y: left, Z: up
    
    This example assumes a common conversion (you may need to adjust):
    """
    # Example: If simulator uses X: right, Y: forward, Z: up
    # (common in some simulators)
    converted = vertices_camera_frame.copy()
    converted[:, [1, 2]] = converted[:, [2, 1]]  # Swap Y and Z
    converted[:, 2] = -converted[:, 2]  # Negate Z (camera looks along -Z)
    return converted


# Example usage:
if __name__ == "__main__":
    # Example paths (adjust to your actual paths)
    mesh_path = "mesh.ply"
    pose_json_path = "pose.json"
    output_path = "mesh_transformed.ply"
    
    # Apply pose to mesh
    vertices_transformed, mesh = apply_pose_to_mesh(
        mesh_path, 
        pose_json_path, 
        output_path
    )
    
    print(f"Original mesh: {mesh_path}")
    print(f"Pose file: {pose_json_path}")
    print(f"Transformed mesh vertices shape: {vertices_transformed.shape}")
    print(f"\nTransformation applied in camera coordinate frame:")
    print(f"  - Scale: Applied first")
    print(f"  - Rotation: Applied second (quaternion rotation)")
    print(f"  - Translation: Applied last")
    print(f"\nNote: Vertices are now in camera coordinate frame.")
    print(f"      You may need to convert to your simulator's coordinate frame.")


