"""
Reconstruct 3D points from gs.ply and reproject to 2D.
This is for validating pose.json (translation/rotation/scale).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple, Optional

import h5py
import numpy as np
import trimesh
from PIL import Image
import matplotlib.pyplot as plt


def load_points_any(mesh_path: Path) -> np.ndarray:
    loaded = trimesh.load(mesh_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 0:
            raise ValueError(f"No geometry in scene: {mesh_path}")
        loaded = loaded.to_geometry()

    if isinstance(loaded, trimesh.Trimesh):
        return np.asarray(loaded.vertices)
    if isinstance(loaded, trimesh.PointCloud):
        return np.asarray(loaded.vertices)

    raise TypeError(f"Unsupported mesh type: {type(loaded)}")


def quaternion_to_matrix(quat: np.ndarray, order: str) -> np.ndarray:
    if order == "wxyz":
        w, x, y, z = quat
    elif order == "xyzw":
        x, y, z, w = quat
    else:
        raise ValueError(f"Unknown quaternion order: {order}")

    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def apply_pose(points: np.ndarray, pose: dict, quat_order: str) -> np.ndarray:
    translation = np.array(pose["translation"], dtype=np.float64)
    rotation_quat = np.array(pose["rotation"], dtype=np.float64)
    scale = np.array(pose["scale"], dtype=np.float64)

    # Flip X/Y to correct left-right and up-down inversion
    axis_flip = np.diag([-1.0, -1.0, 1.0])

    points_scaled = points * scale
    R = quaternion_to_matrix(rotation_quat, quat_order)
    points_rotated = points_scaled @ (axis_flip @ R).T
    points_transformed = points_rotated + translation
    return points_transformed


def load_intrinsics(h5_path: Path, key: str) -> Tuple[float, float, float, float]:
    with h5py.File(h5_path, "r") as f:
        if key not in f:
            available = list(f.keys())
            raise KeyError(f"{key} not found in {h5_path}. keys={available}")
        K = np.array(f[key], dtype=np.float64)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    return fx, fy, cx, cy


def infer_k_key_from_image(image_path: Path, calib_path: Path) -> str:
    """
    Infer intrinsic key like 'NP1_rgb_K' from image filename.
    Example: NP1_0.jpg -> NP1_rgb_K
    """
    stem = image_path.stem
    prefix = stem.split("_")[0]
    candidate = f"{prefix}_rgb_K"
    with h5py.File(calib_path, "r") as f:
        if candidate in f:
            return candidate
        # Fallback: pick any *_rgb_K that matches prefix partially
        for key in f.keys():
            if key.endswith("_rgb_K") and prefix in key:
                return key
    raise KeyError(f"Could not infer K key from image {image_path.name}")


def project_points(
    points_camera: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    image_size: Tuple[int, int],
    camera_looks_negative_z: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = image_size

    x = points_camera[:, 0]
    y = points_camera[:, 1]
    z = points_camera[:, 2]
    if camera_looks_negative_z:
        z = -z

    valid = z > 1e-6
    x = x[valid]
    y = y[valid]
    z = z[valid]

    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    uv = np.stack([u, v], axis=1)

    in_bounds = (
        (uv[:, 0] >= 0) & (uv[:, 0] < width) &
        (uv[:, 1] >= 0) & (uv[:, 1] < height)
    )
    return uv[in_bounds], z[in_bounds]


def plot_overlay(image_path: Path, uv: np.ndarray, depth: np.ndarray) -> None:
    img = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(7, 7))
    plt.imshow(img)
    if uv.shape[0] > 0:
        plt.scatter(uv[:, 0], uv[:, 1], s=1, c=depth, cmap="turbo", alpha=0.6)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reproject gs.ply using pose.json and intrinsics.")
    parser.add_argument("--data-root", help="Root directory containing NP1_0.jpg and calibration.h5")
    parser.add_argument("--gs-ply", help="Path to gs.ply (or mesh/pcd)")
    parser.add_argument("--pose-json", help="Path to pose.json")
    parser.add_argument("--image", help="Input RGB image path")
    parser.add_argument("--calib-h5", help="Calibration H5 path")
    parser.add_argument("--K-key", help="Intrinsic dataset key in H5 (auto if omitted)")
    parser.add_argument("--quat-order", default="wxyz", choices=["wxyz", "xyzw"])
    parser.add_argument("--max-points", type=int, default=200000)
    z_group = parser.add_mutually_exclusive_group()
    z_group.add_argument("--camera-looks-negative-z", action="store_true")
    z_group.add_argument("--camera-looks-positive-z", action="store_true")
    parser.add_argument("--no-auto-flip-z", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.data_root:
        root = Path(args.data_root)
        image_path = Path(args.image) if args.image else root / "NP1_0.jpg"
        calib_path = Path(args.calib_h5) if args.calib_h5 else root / "calibration.h5"
        gs_path = Path(args.gs_ply) if args.gs_ply else root / "red_bowl" / "0_gs.ply"
        pose_path = Path(args.pose_json) if args.pose_json else root / "red_bowl" / "0_pose.json"
    else:
        if not all([args.gs_ply, args.pose_json, args.image, args.calib_h5]):
            raise ValueError("Provide --data-root or all of --gs-ply --pose-json --image --calib-h5.")
        gs_path = Path(args.gs_ply)
        pose_path = Path(args.pose_json)
        image_path = Path(args.image)
        calib_path = Path(args.calib_h5)

    points = load_points_any(gs_path)
    if points.shape[0] > args.max_points:
        idx = np.random.choice(points.shape[0], args.max_points, replace=False)
        points = points[idx]

    with open(pose_path, "r") as f:
        pose = json.load(f)

    points_camera = apply_pose(points, pose, args.quat_order)
    k_key = args.K_key
    if not k_key:
        k_key = infer_k_key_from_image(image_path, calib_path)
    fx, fy, cx, cy = load_intrinsics(calib_path, k_key)

    img = Image.open(image_path)
    image_size = (img.height, img.width)

    looks_negative = True
    if args.camera_looks_positive_z:
        looks_negative = False
    elif args.camera_looks_negative_z:
        looks_negative = True
    auto_flip = not args.no_auto_flip_z

    uv, depth = project_points(points_camera, fx, fy, cx, cy, image_size, looks_negative)
    if uv.shape[0] == 0 and auto_flip:
        uv, depth = project_points(points_camera, fx, fy, cx, cy, image_size, not looks_negative)

    plot_overlay(image_path, uv, depth)


if __name__ == "__main__":
    main()
