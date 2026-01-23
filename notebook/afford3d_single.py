import os
import sys
import argparse
import re
import torch
import numpy as np
import pandas as pd
from typing import Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GEAL_PATH = os.path.join(PROJECT_ROOT, "third_party", "geal")

if GEAL_PATH not in sys.path:
    sys.path.insert(0, GEAL_PATH)

import open3d as o3d
from utils.utils import seed_torch, read_yaml
from dataset.data_utils import normalize_point_cloud, VIEWPOINTS, AFFORDANCES, CLASSES
from model.branch_3d import Branch3D


def load_ply_file(ply_path: str) -> np.ndarray:
    """
    Parse pcd from SAM3D gs.ply file.
    
    Args:
        ply_path: Path to .ply file
    
    Returns:
        points: (N, 3) numpy array of point coordinates
    """
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")
    
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    
    if len(points) == 0:
        raise ValueError(f"PLY file contains no points: {ply_path}")
    
    print(f"Loaded {len(points)} points from {ply_path}")

    return points


def preprocess_point_cloud(points: np.ndarray, n_points: int = 2048) -> tuple[torch.Tensor, np.ndarray]:
    """
    Preprocessing of pcd: Normalization and Farthest Point Sampling (FPS).
    
    Args:
        points: (N, 3) raw point cloud
        n_points: Target number of points (default: 2048)
    
    Returns:
        point_input: (1, 3, n_points) normalized and sampled point cloud tensor (ready for model)
    """
    # Normalization
    normalized_points, centroid, scale = normalize_point_cloud(points)
    normalized_points_tensor = torch.from_numpy(normalized_points).float()  # (N, 3)
    
    # print(f"Normalized point cloud:")
    # print(f"  Centroid: {centroid}")
    # print(f"  Scale: {scale:.6f}")
    # print(f"  Original points: {len(points)}")
    
    # Farthest Point Sampling (FPS)
    if len(normalized_points) > n_points:
        from model.pointnet2_utils import farthest_point_sample, index_points
        
        # Add batch dimension and sample indices
        points_tensor = normalized_points_tensor.unsqueeze(0)  # (1, N, 3)
        fps_indices = farthest_point_sample(points_tensor, n_points)  # (1, n_points)
        
        # Extract sampled points
        sampled_points_tensor = index_points(points_tensor, fps_indices)  # (1, n_points, 3)
        sampled_points = sampled_points_tensor[0]  # (n_points, 3)
        print(f"  Sampled to {n_points} points using FPS")

    elif len(normalized_points) < n_points:
        # Padding
        n_pad = n_points - len(normalized_points)
        last_point = normalized_points_tensor[-1:].repeat(n_pad, 1)  # (n_pad, 3)
        sampled_points = torch.cat([normalized_points_tensor, last_point], dim=0)  # (n_points, 3)
        print(f"  Padded from {len(normalized_points)} to {n_points} points")

    else:
        sampled_points = normalized_points_tensor
        print(f"  Already has {n_points} points, no sampling needed")
    
    # Convert to model input format
    point_input = sampled_points.T.unsqueeze(0)  # (1, 3, n_points)
    
    return point_input


def load_question_table(data_root: str) -> pd.DataFrame:
    """
    Load Affordance-Question.csv table.
    
    Args:
        data_root: Root directory containing Affordance-Question.csv
    
    Returns:
        questions: DataFrame with question mappings
    """
    csv_path = os.path.join(data_root, "Affordance-Question.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Question table not found: {csv_path}\n"
            f"Please ensure Affordance-Question.csv exists in {data_root}"
        )
    questions = pd.read_csv(csv_path)

    return questions


def _generate_default_question(object_class: str, affordance: str) -> str:
    """
    Generate a default question template for object-affordance pairs not in CSV.
    
    Args:
        object_class: Object class name
        affordance: Affordance type
    
    Returns:
        Default question string
    """
    templates = {
        "grasp": f"if you want to grab the {object_class}, at which points will your palm position be ?",
        "wrap_grasp": f"if you wrap-grasp the {object_class}, at which points on the {object_class} wall will your palm touch ?",
        "contain": f"if you package things into the {object_class}, at which points in this {object_class} would you put?",
        "pour": f"suppose there is water in the {object_class}, and you want to pour the water out of the {object_class}. from which points on the {object_class} will the water flow out?",
        "open": f"if you want to open the {object_class}, from which points of the {object_class} would you open it ?",
        "close": f"if you want to close the {object_class}, at which points would you close it ?",
        "lift": f"if you want to lift the {object_class}, at which points would you lift it ?",
        "move": f"if you want to move the {object_class}, at which points would you move it ?",
        "push": f"if you want to push the {object_class}, at which points would you push it ?",
        "pull": f"if you want to pull the {object_class}, at which points would you pull it ?",
        "press": f"if you want to press the {object_class}, at which points would you press it ?",
        "sit": f"if you were to sit on the {object_class}, at which points on the {object_class} would you sit ?",
        "lay": f"if you were to lie on the {object_class}, which points would you lie on the {object_class} ?",
        "support": f"if you put something on the {object_class}, at which points on the {object_class} would you put it ?",
        "display": f"if you want to display something on the {object_class}, at which points would you display it ?",
        "wear": f"if you want to wear the {object_class}, at which points would you wear it ?",
        "listen": f"if you want to listen with the {object_class}, at which points would you use it ?",
        "cut": f"if you want to cut something with the {object_class}, at which points would you cut it ?",
        "stab": f"if you want to stab something with the {object_class}, at which points would you stab it ?",
    }
    
    # Return specific template if available, otherwise use generic template
    if affordance in templates:
        return templates[affordance]
    else:
        # Generic fallback template
        return f"if you want to {affordance} the {object_class}, at which points would you {affordance} it ?"


def generate_questions(
    object_class: str,
    affordance: str,
    questions_df: pd.DataFrame,
    split: str = "test"
) -> list:
    """
    Generate viewpoint-conditioned questions.
    
    Args:
        object_class: Object class name (e.g., "bottle")
        affordance: Affordance type (e.g., "contain")
        questions_df: DataFrame from Affordance-Question.csv
        split: "train" or "test" (test uses Question0, train uses random)
    
    Returns:
        questions: List of 12 tuples, each containing one viewpoint-conditioned question
                   Format: [("question1",), ("question2",), ..., ("question12",)]
                   (For inference, batch_size = 1)
    """
    # Sample question
    qid = f"Question{np.random.randint(1, 15)}" if split == "train" else "Question0"
    
    row = questions_df.loc[
        (questions_df["Object"] == object_class.capitalize()) & 
        (questions_df["Affordance"] == affordance),
        [qid]
    ]
    
    if row.empty:
        row = questions_df.loc[
            (questions_df["Object"] == object_class.lower()) & 
            (questions_df["Affordance"] == affordance),
            [qid]
        ]
    
    # If not found in CSV, generate default question template
    if row.empty:
        print(f"Warning: No question found for {object_class}-{affordance} in CSV. Using default template.")
        base_question = _generate_default_question(object_class, affordance)
    else:
        base_question = row.iloc[0][qid]
    
    questions = [
        (f"This is a depth map of a {object_class} viewed {vp}. {base_question}",)
        for vp in VIEWPOINTS
    ]
    
    return questions


def run_inference(
    model: Branch3D,
    point_input: torch.Tensor,
    questions: list,
    device: torch.device
) -> np.ndarray:
    """
    Run GEAL inference.
    
    Args:
        model: Loaded Branch3D model
        point_input: (1, 3, N) normalized point cloud tensor
        questions: List of 12 tuples, each containing one viewpoint-conditioned question
                   Format: [("question1",), ("question2",), ..., ("question12",)]
        device: GPU/CPU device
    
    Returns:
        affordance_map: (N,) numpy array of affordance scores
    """
    model.eval()
    
    point_tensor = point_input.to(device)  # (1, 3, N)
    
    with torch.no_grad():
        point = point_tensor.float().to(device)
        pred = model(questions, point)
    affordance_map = pred.cpu().numpy()
    
    return affordance_map


def save_pair_to_ply(
    points: np.ndarray,
    pred_mask: np.ndarray,
    save_dir: str,
    output_filename: str
) -> str:
    """
    Save points and pred_mask to PLY file.
    
    Args:
        points: (N, 3) numpy array of points
        pred_mask: (N,) numpy array of pred_mask (normalized to [0, 1])
        save_dir: Directory to save PLY file
        output_filename: Output filename (e.g., "0_aff_bottle_grasp.ply")
    
    Returns:
        pred_path: Path to saved PLY file
    """
    os.makedirs(save_dir, exist_ok=True)
    back = np.array([190, 190, 190], dtype=np.float64)
    red = np.array([255, 0, 0], dtype=np.float64)

    def colorize(scores):
        return ((red - back) * scores.reshape(-1, 1) + back) / 255.0

    pred_pc = o3d.geometry.PointCloud()
    pred_pc.points = o3d.utility.Vector3dVector(points)
    pred_pc.colors = o3d.utility.Vector3dVector(colorize(pred_mask))

    pred_path = os.path.join(save_dir, output_filename)
    o3d.io.write_point_cloud(pred_path, pred_pc)

    return pred_path


def afford3d_single(
    ply_path: str,
    affordance: str,
    object_class: str = "object",
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    data_root: Optional[str] = None,
    device: str = "cuda",
    seed: Optional[int] = None,
) -> str:
    """
    Run GEAL inference on a single PLY file to generate affordance map.
    
    Args:
        ply_path: Path to input PLY file
        affordance: Affordance type (e.g., "grasp", "contain")
        object_class: Object class name (e.g., "bottle")
        config_path: Path to config file (defaults to evaluation.yaml)
        checkpoint_path: Path to checkpoint (overrides config)
        data_root: Data root directory containing Affordance-Question.csv
        device: Device (cuda/cpu)
        seed: Random seed
    
    Returns:
        output_path: Path to saved affordance PLY file
    """
    # Load config and set defaults
    if config_path is None:
        config_path = os.path.join(GEAL_PATH, "config", "evaluation.yaml")
    cfg = read_yaml(config_path)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    if seed is not None:
        seed_torch(seed)
    elif "seed" in cfg:
        seed_torch(cfg["seed"])
    
    # Set checkpoint path
    if checkpoint_path is None:
        checkpoint_path = cfg.get("ckpt", "ckpt/piad_seen.pt")
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(GEAL_PATH, checkpoint_path)
            checkpoint_path = os.path.normpath(checkpoint_path)
    
    # Set data root
    if data_root is None:
        data_root = cfg.get("data_root", "dataset/piad_dataset")
        if not os.path.isabs(data_root):
            data_root = os.path.join(GEAL_PATH, data_root)
            data_root = os.path.normpath(data_root)
    
    # Validate
    if object_class not in CLASSES:
        print(f"Warning: '{object_class}' not in predefined object classes.")
        print(f"Available: {', '.join(CLASSES)}")
        print("Proceeding anyway (model may still work)...")

    if affordance not in AFFORDANCES:
        print(f"Warning: '{affordance}' not in predefined affordances.")
        print(f"Available: {', '.join(AFFORDANCES)}")
        print("Proceeding anyway (model may still work)...")
    
    # Inference
    print("\nLoading model...")
    model_cfg = cfg["model_3d"]
    model = Branch3D(model_cfg)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    print(f"Loaded checkpoint: {checkpoint_path}")

    print(f"\nLoading PLY file: {ply_path}")
    points = load_ply_file(ply_path)
    
    print("\nPreprocessing point cloud...")
    point_input = preprocess_point_cloud(points)
    
    print(f"\nLoading question table from: {data_root}")
    questions_df = load_question_table(data_root)
    
    print(f"Generating questions for {object_class}-{affordance}...")
    questions = generate_questions(
        object_class,
        affordance,
        questions_df,
        split="test"  # Use Question0 for consistency
    )
    print(f"  Base question: {questions[0][0]}")
    
    print("\nRunning inference...")
    affordance_map = run_inference(model, point_input, questions, device)
    
    # Save results
    # Parse filename (assumes format: {number}_gs.ply)
    input_basename = os.path.splitext(os.path.basename(ply_path))[0]
    input_dir = os.path.dirname(os.path.abspath(ply_path)) or "."
    match = re.match(r'^(\d+)_gs$', input_basename)
    if match:
        number = match.group(1)
    else:
        number = input_basename
        print(f"Warning: Input filename '{input_basename}' doesn't match expected pattern '{{number}}_gs', using '{number}' as prefix")
    
    # Generate output filename: {number}_aff_{object}_{affordance}.ply
    output_filename = f"{number}_aff_{object_class}_{affordance}.ply"
    output_path = os.path.join(input_dir, output_filename)
    
    print(f"\nSaving results...")
    print(f"  Output directory: {input_dir}")
    print(f"  Output filename: {output_filename}")
    
    point_input_np = point_input.squeeze(0).T.cpu().numpy()  # (1, 3, 2048) -> (2048, 3)
    
    if affordance_map.ndim > 1:
        affordance_map = affordance_map.squeeze()
    
    save_pair_to_ply(point_input_np, affordance_map, input_dir, output_filename)
    print(f"  Saved to: {output_path}")
    print("\nInference complete!")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="GEAL Inference on Single PLY File")
    parser.add_argument("--input", type=str, required=True, help="Input PLY file path")
    parser.add_argument("--affordance", type=str, required=True,
                        help=f"Affordance type ({', '.join(AFFORDANCES)})")
    parser.add_argument("--object-class", type=str, default="object",
                        help=f"Object class name (optional, default: 'object')")
    
    # Model config (defaults from evaluation.yaml)
    default_config_path = os.path.join(GEAL_PATH, "config", "evaluation.yaml")
    parser.add_argument("--config", type=str, default=default_config_path,
                        help="Config file path")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path (overrides config)")
    
    # Data root
    parser.add_argument("--data-root", type=str, default=None,
                        help="Data root directory containing Affordance-Question.csv")
    
    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Output PLY file path (default: input_name_affordance.ply)")
    
    # Others
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    args = parser.parse_args()
    
    afford3d_single(
        ply_path=args.input,
        affordance=args.affordance,
        object_class=args.object_class,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()