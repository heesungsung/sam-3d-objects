import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import gc
import json
from typing import Optional, List, Union, Tuple, Dict

# Set spconv algorithm to native early to avoid CUDA kernel errors
if "SPCONV_ALGO" not in os.environ:
    os.environ["SPCONV_ALGO"] = "native"
    print("Setting SPCONV_ALGO=native to avoid CUDA kernel errors")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAM3_PATH = os.path.join(PROJECT_ROOT, "third_party", "sam3")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SAM3_PATH not in sys.path:
    sys.path.insert(0, SAM3_PATH)

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.visualization_utils import plot_mask, plot_results, COLORS
from inference import Inference, load_image, interactive_visualizer
from sam3d_objects.utils.visualization import SceneVisualizer
import open3d as o3d


def setup_device():
    """Setup device and PyTorch settings"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # For Ampere GPUs
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 3 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS."
        )
    
    return device


def load_sam3_model(device, checkpoint_path=None, bpe_path=None):
    """
    Load SAM3 model
    
    Args:
        device: torch device
        checkpoint_path: Path to SAM3 checkpoint (None uses default)
        bpe_path: Path to BPE tokenizer file (None uses default)
    
    Returns:
        SAM3 model and processor
    """
    if bpe_path is None:
        bpe_paths = [
            os.path.join(SAM3_PATH, "checkpoints", "bpe_simple_vocab_16e6.txt.gz"),
        ]
        for path in bpe_paths:
            if os.path.exists(path):
                bpe_path = path
                break
        
        if bpe_path is None:
            raise FileNotFoundError("  BPE file not found.")
    
    checkpoint_dir = os.path.join(SAM3_PATH, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if checkpoint_path is None:
        checkpoint_path = os.path.join(checkpoint_dir, "sam3.pt")
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at {checkpoint_path}")
            print("  Downloading checkpoint from HuggingFace...")
            from huggingface_hub import hf_hub_download
            SAM3_MODEL_ID = "facebook/sam3"
            SAM3_CKPT_NAME = "sam3.pt"
            downloaded_path = hf_hub_download(
                repo_id=SAM3_MODEL_ID,
                filename=SAM3_CKPT_NAME,
                local_dir=checkpoint_dir,
            )
            print(f"  Checkpoint downloaded to: {downloaded_path}")
        else:
            print(f"Using existing checkpoint at: {checkpoint_path}")
    
    print(f"Loading SAM3 model from: {checkpoint_path}")
    model = build_sam3_image_model(bpe_path=bpe_path, checkpoint_path=checkpoint_path, load_from_HF=False)
    processor = Sam3Processor(model, confidence_threshold=0.5)
    
    return model, processor


def generate_mask_with_sam3(
    image: Union[Image.Image, np.ndarray],
    processor: Sam3Processor,
    inference_state,
    text_prompt: Optional[str] = None,
    boxes: Optional[List[List[float]]] = None,
    box_labels: Optional[List[bool]] = None,
    is_vlm: bool = False,
    output_dir: Optional[str] = None,
) -> Tuple[dict, dict]:
    """
    Generate masks using SAM3 with text prompt or bounding boxes
    
    Args:
        image: Input image
        processor: SAM3 processor
        inference_state: SAM3 inference state (from processor.set_image)
        text_prompt: Text prompt
        boxes: List of bounding boxes
        box_labels: List of labels for boxes (True for positive, False for negative)
        is_vlm: If True, indicates this is from VLM (e.g., Gemini) and should select best mask
        output_dir: Optional directory to save tmp/all_masks.png
    
    Returns:
        inference_state: Full inference state from SAM3
    """
    processor.reset_all_prompts(inference_state)
    best_idx = None  # only used when is_vlm=True
    
    # Text prompt
    if text_prompt is not None:
        inference_state = processor.set_text_prompt(state=inference_state, prompt=text_prompt)
        print(f"Using text prompt: '{text_prompt}'")
    
    # Bounding boxes
    if boxes is not None and len(boxes) > 0:
        if isinstance(image, Image.Image):
            img_w, img_h = image.size
        else:
            img_h, img_w = image.shape[:2]
        
        print(f"Using {len(boxes)} bounding box(es)")
        for i, box in enumerate(boxes):
            if len(box) != 4:
                print(f"Warning: Box {i} has invalid format, skipping")
                continue
            
            x, y, w, h = box
            
            # Normalize to [0, 1] and convert to [center_x, center_y, width, height]
            x_norm = x / img_w
            y_norm = y / img_h
            w_norm = w / img_w
            h_norm = h / img_h
            
            center_x = x_norm + w_norm / 2.0
            center_y = y_norm + h_norm / 2.0
            label = box_labels[i] if box_labels is not None and i < len(box_labels) else True
            # print(f"  Box {i+1}: [{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}] -> center=[{center_x:.3f}, {center_y:.3f}], size=[{w_norm:.3f}, {h_norm:.3f}], label={label}")
            
            inference_state = processor.add_geometric_prompt(
                box=[center_x, center_y, w_norm, h_norm],
                label=label,
                state=inference_state
            )
    
    if "masks" in inference_state and inference_state["masks"] is not None:
        num_masks = len(inference_state["scores"])
        print(f"Found {num_masks} mask(s)")
        
        # Save all masks visualization before any selection
        if output_dir is not None:
            if isinstance(image, Image.Image):
                img_pil = image
            else:
                img_pil = Image.fromarray(image.astype(np.uint8))
            
            masks_tensor = inference_state["masks"]
            scores_tensor = inference_state["scores"]
            boxes_tensor = inference_state["boxes"]
            
            masks_list = [masks_tensor[i] for i in range(num_masks)]
            scores_list = [scores_tensor[i] for i in range(num_masks)]
            boxes_list = [boxes_tensor[i] for i in range(num_masks)]
            
            results = {
                "masks": masks_list,
                "scores": scores_list,
                "boxes": boxes_list,
            }
            
        # Select only the best matching mask if using VLM (based on IOU and score)
        if is_vlm and boxes is not None and len(boxes) > 0 and num_masks > 1:
            print(f"  Selecting best mask from {num_masks} candidates based on bounding box overlap...")
            results["ious"] = []
            
            if isinstance(image, Image.Image):
                img_w, img_h = image.size
            else:
                img_h, img_w = image.shape[:2]
            
            box = boxes[0]
            x, y, w, h = box
            target_box = [x, y, x + w, y + h]
            
            # Calculate IoU
            best_idx = 0
            best_iou = 0.0
            
            masks = inference_state["masks"]
            pred_boxes = inference_state["boxes"]
            
            if isinstance(pred_boxes, torch.Tensor):
                pred_boxes_np = pred_boxes.cpu().numpy()
            else:
                pred_boxes_np = pred_boxes
            
            for i in range(num_masks):
                pred_box = pred_boxes_np[i]  # [x1, y1, x2, y2]
                
                x1_inter = max(target_box[0], pred_box[0])
                y1_inter = max(target_box[1], pred_box[1])
                x2_inter = min(target_box[2], pred_box[2])
                y2_inter = min(target_box[3], pred_box[3])
                
                if x2_inter <= x1_inter or y2_inter <= y1_inter:
                    iou = 0.0

                else:
                    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                    target_area = w * h
                    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                    union_area = target_area + pred_area - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0.0
                
                results["ious"].append(iou)
                score = inference_state["scores"][i].item()
                print(f"    Mask {i}: IoU={iou:.3f}, Score={score:.3f}")
                
                # Prefer higher IoU, but if IoU is similar, prefer higher score
                best_score = inference_state["scores"][best_idx].item()
                if iou > best_iou or (abs(iou - best_iou) < 0.05 and score > best_score):
                    best_iou = iou
                    best_idx = i
            
            print(f"  Selected mask {best_idx} with IoU={best_iou:.3f}")
            
            # Filter to keep only the best mask
            inference_state["masks"] = masks[best_idx:best_idx+1]
            inference_state["boxes"] = pred_boxes[best_idx:best_idx+1]
            inference_state["scores"] = inference_state["scores"][best_idx:best_idx+1]
            
            print(f"  Filtered to 1 mask")

            # Save visualization of all masks
            plot_results(img_pil, results)
            fig = plt.gcf()
            fig.set_size_inches(12, 8)
            
            os.makedirs(output_dir, exist_ok=True)
            viz_path = os.path.join(output_dir, "tmp/all_masks.png")
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  All masks visualization saved to: {viz_path}")
        
        return inference_state, best_idx

    else:
        print("Warning: No masks found in output")
        return None


def create_masked_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Create an image with mask applied
    
    Args:
        image: Input image (numpy array, RGB, shape [H, W, 3])
        mask: Binary mask (numpy array, 0 or 1, shape [H, W])
    
    Returns:
        masked_image: Image with mask applied (numpy array, RGB)
    """
    if mask.ndim > 2:
        mask = mask.squeeze()
    if mask.ndim != 2:
        raise ValueError(f"  Mask must be 2D, got shape {mask.shape}")
    
    if mask.shape[:2] != image.shape[:2]:
        from PIL import Image as PILImage
        mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize((image.shape[1], image.shape[0]), PILImage.NEAREST)
        mask = np.array(mask_pil) / 255.0
    
    masked_image = image.copy()
    masked_image[mask == 0] = 0
    return masked_image


def generate_mesh_with_sam3d(
    image: np.ndarray,
    mask: np.ndarray,
    config_path: str,
    seed: Optional[int] = 42,
    output_dir: Optional[str] = None,
    generate_mesh: bool = True,
) -> dict:
    """
    Generate mesh using SAM3D from original image and binary mask
    
    Args:
        image: Original image (numpy array, RGB, shape [H, W, 3])
        mask: Binary mask (numpy array, 0 or 1, shape [H, W])
        config_path: Path to SAM3D pipeline config file
        seed: Random seed
        output_dir: Directory to save outputs
        generate_mesh: If True, generate both mesh and gaussian splatting; if False, only gaussian splatting
    
    Returns:
        output: Dictionary containing mesh and/or gaussian splatting outputs
    """
    print("Loading SAM3D model...")
    inference = Inference(config_path, compile=False)
    
    print("\nGenerating 3D representation with SAM3D...")
    
    # Determine decode formats
    if generate_mesh:
        decode_formats = ["mesh", "gaussian"]  # mesh needs gaussian for texture baking
        print("  Generating mesh and gaussian splatting")
    else:
        decode_formats = ["gaussian"]
        print("  Generating gaussian splatting only")
    
    try:
        output = inference(image, mask, seed=seed, decode_formats=decode_formats)
    except RuntimeError as e:
        error_msg = str(e)
        if "spconv" in error_msg.lower() or "can't find suitable algorithm" in error_msg:
            print("\n  spconv CUDA kernel error occurred.")
            if generate_mesh:
                try:
                    print("  Retrying with gaussian splatting only (mesh skipped)...")
                    output = inference(image, mask, seed=seed, decode_formats=["gaussian"])
                    print("  Successfully generated gaussian splatting (mesh skipped)")
                except Exception as e2:
                    print(f"  Recovery failed: {e2}")
                    raise e
            else:
                print("\n  Gaussian splatting generation failed.")
                raise
        else:
            raise
    
    return output


def visualize_masks_and_select(image: Union[Image.Image, np.ndarray], inference_state: dict, output_dir: Optional[str] = None) -> int:
    """
    Visualize all masks using plot_results style and allow interactive selection by clicking
    
    Args:
        image: Input image (PIL Image or numpy array)
        inference_state: SAM3 inference_state dictionary with "masks", "scores", "boxes"
        output_dir: Optional directory to save visualization
    
    Returns:
        selected_index: Index of selected mask
    """
    if isinstance(image, np.ndarray):
        img_pil = Image.fromarray(image.astype(np.uint8))
    else:
        img_pil = image
    
    if "masks" not in inference_state or inference_state["masks"] is None:
        print("No masks to visualize")
        return 0
    
    # Convert inference_state to plot_results format
    masks_tensor = inference_state["masks"]  # Shape: [N, H, W] or [N, 1, H, W]
    scores_tensor = inference_state["scores"]  # Shape: [N]
    boxes_tensor = inference_state["boxes"]  # Shape: [N, 4]
    
    num_masks = len(scores_tensor)
    masks_list = [masks_tensor[i] for i in range(num_masks)]
    scores_list = [scores_tensor[i] for i in range(num_masks)]
    boxes_list = [boxes_tensor[i] for i in range(num_masks)]
    
    results = {
        "masks": masks_list,
        "scores": scores_list,
        "boxes": boxes_list,
    }
    
    # Plot and select
    plot_results(img_pil, results)
    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    
    if num_masks == 1:
        print("Only one mask found, using it automatically...")
        plt.show()
        return 0
    
    # Add click handler for mask selection
    selected_index = [None]
    
    def on_click(event):
        if event.inaxes is None:
            return
        
        # Get click coordinates
        x, y = event.xdata, event.ydata
        if x is None or y is None or x < 0 or y < 0:
            return
        x, y = int(x), int(y)
        
        # Check which mask contains this point
        img_w, img_h = img_pil.size
        if x >= img_w or y >= img_h:
            return
        
        # Find which mask contains the clicked point
        for i, mask_tensor in enumerate(masks_list):
            mask_np = mask_tensor.squeeze().cpu().numpy() if isinstance(mask_tensor, torch.Tensor) else mask_tensor.squeeze()
            # Resize mask to image size if needed
            if mask_np.shape != (img_h, img_w):
                from PIL import Image as PILImage
                mask_pil = PILImage.fromarray((mask_np * 255).astype(np.uint8))
                mask_pil = mask_pil.resize((img_w, img_h), PILImage.NEAREST)
                mask_np = np.array(mask_pil) / 255.0
            
            if mask_np[y, x] > 0.5:
                selected_index[0] = i
                print(f"  Selected mask {i} (clicked at ({x}, {y}))")
                plt.close(fig)
                return
        
        print(f"  No mask found at clicked position ({x}, {y}). Please click on a mask region.")
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    print(f"\nClick on a mask to select it.")
    plt.show()
    
    return selected_index[0] if selected_index[0] is not None else 0


def process_pose_estimation(output: dict, save_path: Optional[str] = None, save_pose: bool = True) -> Optional[Dict]:
    """
    Pose estimation from SAM3D output.
    
    Args:
        output: Output dictionary from SAM3D inference
        save_path: Path to save pose information as JSON file.
        save_pose: Whether to save pose information as JSON file.
        
    Returns:
        Dictionary containing pose information (translation, rotation, scale) or None if not available
    """
    if "translation" not in output or "rotation" not in output or "scale" not in output:
        return None
    
    translation = output["translation"]
    rotation = output["rotation"]
    scale = output["scale"]
    
    if isinstance(translation, torch.Tensor):
        translation_np = translation.cpu().float().numpy()
        if translation_np.ndim > 1:
            translation = translation_np.squeeze().tolist()
        else:
            translation = translation_np.tolist()
        
    if isinstance(rotation, torch.Tensor):
        rotation_np = rotation.cpu().float().numpy()
        if rotation_np.ndim > 1:
            rotation = rotation_np.squeeze().tolist()
        else:
            rotation = rotation_np.tolist()
        
    if isinstance(scale, torch.Tensor):
        scale_np = scale.cpu().float().numpy()
        if scale_np.ndim > 1:
            scale = scale_np.squeeze().tolist()
        else:
            scale = scale_np.tolist()
    
    if len(translation) != 3 or len(rotation) != 4 or len(scale) != 3:
        print(f"Warning: Unexpected pose dimensions - translation: {len(translation)}, rotation: {len(rotation)}, scale: {len(scale)}")
        return None
    
    pose_info = {
        "translation": translation,
        "rotation": rotation,
        "scale": scale,
        "frame": "camera",
    }
    
    # Print pose information
    print(f"Translation (camera frame): [{translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}]")
    print(f"Rotation (quaternion [w, x, y, z]): [{rotation[0]:.4f}, {rotation[1]:.4f}, {rotation[2]:.4f}, {rotation[3]:.4f}]")
    print(f"Scale: [{scale[0]:.4f}, {scale[1]:.4f}, {scale[2]:.4f}]")
    print("\nFrame Details:")
    print("- Transformation from object's local coordinate to the camera coordinate frame (right-handed system)")
    print("  * Origin: camera position [0, 0, 0]")
    print("  * X-axis: right (positive)")
    print("  * Y-axis: up (positive)")
    print("  * Z-axis: viewing direction is -Z (negative)")
    print("- Quaternion format: [w, x, y, z] (PyTorch3D convention)")
    
    if save_path is not None and save_pose:
        with open(save_path, 'w') as f:
            json.dump(pose_info, f, indent=2)
        print(f"\nPose information saved to: {save_path}")
    
    return pose_info


def recond3d_single(
    image_path: str,
    text_prompt: Optional[str] = None,
    boxes: Optional[List[List[float]]] = None,
    box_labels: Optional[List[bool]] = None,
    sam3d_config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    save_mask: bool = True,
    save_3d: bool = True,
    save_pose: bool = True,
    seed: int = 42,
    checkpoint_path: Optional[str] = None,
    generate_mesh: bool = True,
    is_vlm: bool = False,
):
    """
    Complete pipeline: Image -> SAM3 mask -> SAM3D reconstruction
    
    Args:
        image_path: Path to input image
        text_prompt: Text prompt for SAM3 (e.g., "shoe", "person")
        boxes: List of bounding boxes [[x, y, w, h], ...]
        box_labels: List of labels for boxes (True/False)
        sam3d_config_path: Path to SAM3D config file
        output_dir: Directory to save outputs
        save_mask: Whether to save mask image
        save_3d: Whether to save 3D output files
        save_pose: Whether to save pose information
        seed: Random seed for SAM3D
        checkpoint_path: Path to SAM3 checkpoint (optional)
        generate_mesh: If True, generate both mesh and gaussian splatting; if False, only gaussian splatting
        is_vlm: If True, indicates this is from VLM (e.g., Gemini) and should auto-select best mask
    """
    device = setup_device()
    
    # Load image
    print(f"Loading image from: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    
    if output_dir is None:
        input_dir = os.path.dirname(os.path.abspath(image_path))
        input_basename = os.path.splitext(os.path.basename(image_path))[0]
        
        if text_prompt is not None and text_prompt.strip():
            folder_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in text_prompt.strip())
            folder_name = folder_name.replace(' ', '_')
            output_dir = os.path.join(input_dir, folder_name)
        else:
            output_dir = os.path.join(input_dir, f"{input_basename}_recon3d_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load SAM3 model
    print("\n=== Step 1: Loading SAM3 model ===")
    sam3_model, processor = load_sam3_model(device, checkpoint_path=checkpoint_path)
    inference_state = processor.set_image(image)
    
    # Generate masks
    print("\n=== Step 2: Generating masks ===")
    results, best_idx = generate_mask_with_sam3(
        image=image,
        processor=processor,
        inference_state=inference_state,
        text_prompt=text_prompt,
        boxes=boxes,
        box_labels=box_labels,
        is_vlm=is_vlm,
        output_dir=output_dir if save_mask else None,
    )
    
    if results is None:
        print("  Warning: No masks found in output")
        return
    
    # Select a single mask (human intervention)
    print("\n=== Step 3: Mask selection ===")
    if is_vlm and best_idx is not None:
        print("  Skipping mask selection... Using single mask generated by VLM")
        selected_index = 0  # only single mask generated when using VLM
        
    else:
        selected_index = visualize_masks_and_select(
            image=image,
            inference_state=results,
            output_dir=None
        )
        best_idx = selected_index
        
    selected_mask_tensor = results["masks"][selected_index]
    selected_mask = selected_mask_tensor.squeeze().cpu().numpy() if isinstance(selected_mask_tensor, torch.Tensor) else selected_mask_tensor.squeeze()
    selected_mask = (selected_mask > 0.5).astype(np.uint8)
    print(f"\nUsing mask {best_idx} with {np.sum(selected_mask > 0)} pixels:")
    masked_image = create_masked_image(image_np, selected_mask)
    
    if save_mask:
        mask_path = os.path.join(output_dir, f"tmp/{best_idx}_mask.png")
        Image.fromarray((selected_mask * 255).astype(np.uint8)).save(mask_path)
        print(f"  Selected mask saved to: {mask_path}")
        
        # Save overlay
        overlay = image_np.copy()
        overlay[selected_mask > 0] = overlay[selected_mask > 0] * 0.7 + np.array([255, 0, 0]) * 0.3
        overlay_path = os.path.join(output_dir, f"tmp/{best_idx}_overlay.png")
        Image.fromarray(overlay.astype(np.uint8)).save(overlay_path)
        print(f"  Mask overlay saved to: {overlay_path}")
        
        # Save masked image
        masked_image_path = os.path.join(output_dir, f"tmp/{best_idx}_masked_image.png")
        Image.fromarray(masked_image.astype(np.uint8)).save(masked_image_path)
        print(f"  Masked image saved to: {masked_image_path}")
    
    # Generate mesh with SAM3D
    if sam3d_config_path is not None:
        print("\n=== Step 4: Generating 3D representation with SAM3D ===")
        mesh_output = generate_mesh_with_sam3d(
            image=image_np,  # [H, W, 3] RGB - same shape as masked_image
            mask=selected_mask,  # [H, W] binary mask (0 or 1)
            config_path=sam3d_config_path,
            seed=seed,
            output_dir=output_dir,
            generate_mesh=generate_mesh,
        )

        # Save outputs
        saved_paths = {"gs": None, "mesh": None}
        if save_3d and output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            print(f"\nSaving 3D outputs...")
            print(f"  GS format: PLY (fixed)")
            print(f"  Mesh format: GLB (fixed, SAM3D default)")
            
            # Save gaussian splatting (always PLY)
            if "gs" in mesh_output:
                gs_path = os.path.join(output_dir, f"{selected_index}_gs.ply")
                try:
                    mesh_output["gs"].save_ply(gs_path)
                    print(f"  Gaussian splatting saved to: {gs_path}")
                    saved_paths["gs"] = gs_path
                except Exception as e:
                    print(f"  Failed to save gaussian splatting: {e}")
            
            # Save mesh (always GLB, SAM3D default format)
            if generate_mesh:
                if "glb" in mesh_output:
                    mesh_obj = mesh_output["glb"]
                    if mesh_obj is not None:
                        mesh_path = os.path.join(output_dir, f"{selected_index}_mesh.glb")
                        try:
                            mesh_obj.export(mesh_path)
                            print(f"  Mesh saved to: {mesh_path}")
                            saved_paths["mesh"] = mesh_path
                        except Exception as e:
                            print(f"  Failed to save mesh (glb): {e}")
                
                if saved_paths["mesh"] is None:
                    print(f"  Warning: Mesh not available in output (only gaussian splatting was generated)")
        
        # Extract, print, and save pose information
        print("\n=== Step 5: Pose estimation ===")
        pose_path = os.path.join(output_dir, f"{selected_index}_pose.json") if output_dir is not None else None
        pose_info = process_pose_estimation(mesh_output, save_path=pose_path, save_pose=save_pose)
        
        return selected_mask, saved_paths

    else:
        print("\n  SAM3D config path not provided. Skipping mesh generation.")
        return selected_mask, None
    


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SAM3 Mask Generation and SAM3D Mesh Reconstruction")
    parser.add_argument("--input", type=str, required=False,
                       default=os.path.join(PROJECT_ROOT, "notebook", "images", "kid_box", "image.png"),
                       help="Input image path (.jpg, .png)")
    parser.add_argument("--text-prompt", type=str, default=None,
                       help="Text prompt for SAM3 (e.g., 'shoe', 'person')")
    parser.add_argument("--boxes", type=str, default=None,
                       help="Bounding boxes in format 'x1,y1,w1,h1;x2,y2,w2,h2' (semicolon separated)")
    parser.add_argument("--box-labels", type=str, default=None,
                       help="Box labels in format 'True,False' (comma separated, True=positive, False=negative)")
    parser.add_argument("--sam3d-config", type=str, default=None,
                       help="Path to SAM3D pipeline config file")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: same as input image directory)")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to SAM3 checkpoint (optional)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for SAM3D")
    parser.add_argument("--no-save-mask", action="store_true",
                       help="Do not save mask images")
    parser.add_argument("--no-save-3d", action="store_true",
                       help="Do not save 3D output files")
    parser.add_argument("--gs-only", action="store_true",
                       help="Generate only gaussian splatting (no mesh). Default: generate both mesh and gs")
    
    args = parser.parse_args()
    
    # Parse boxes
    boxes = None
    if args.boxes:
        boxes = []
        for box_str in args.boxes.split(";"):
            box = [float(x) for x in box_str.split(",")]
            if len(box) == 4:
                boxes.append(box)
    
    box_labels = None
    if args.box_labels:
        box_labels = [x.strip().lower() == "true" for x in args.box_labels.split(",")]
    
    print("\n")
    if args.text_prompt is None and boxes is None:
        print("Error: Either --text-prompt or --boxes must be provided")
        return
    
    if args.sam3d_config is None:
        default_config = os.path.join(PROJECT_ROOT, "checkpoints", "hf", "pipeline.yaml")
        if os.path.exists(default_config):
            args.sam3d_config = default_config
            print(f"Using default SAM3D config: {args.sam3d_config}")
        else:
            print("Warning: SAM3D config not found. Mesh generation will be skipped.")
            print(f"  Expected at: {default_config}")
    
    recond3d_single(
        image_path=args.input,
        text_prompt=args.text_prompt,
        boxes=boxes,
        box_labels=box_labels,
        sam3d_config_path=args.sam3d_config,
        output_dir=args.output_dir,
        save_mask=not args.no_save_mask,
        save_3d=not args.no_save_3d,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
        generate_mesh=not args.gs_only,  # If gs_only, generate_mesh=False
    )


if __name__ == "__main__":
    main()

