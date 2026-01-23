"""
Integrated Script: Isaac Lab Simulation + SAM3D Reconstruction
Fix: Added error handling for 0 masks and restored interactive mask selection.
"""

import argparse
import os
import sys
import numpy as np
import torch
import random
from PIL import Image
import trimesh

# --- 1. Isaac Lab App Setup ---
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab + SAM3D Pipeline")
parser.add_argument("--object", type=str, default="cube", 
                    choices=["cube", "cone", "cylinder", "sphere", "table"],
                    help="Type of objects to spawn.")
parser.add_argument("--num_objects", type=int, default=1, help="Number of objects to spawn.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli) 
simulation_app = app_launcher.app

# --- 2. Imports ---
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.spawners.shapes import CuboidCfg, ConeCfg, CylinderCfg, SphereCfg
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# SAM3D Setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAM3_PATH = os.path.join(PROJECT_ROOT, "third_party", "sam3")

if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
if SAM3_PATH not in sys.path: sys.path.insert(0, SAM3_PATH)

os.environ["SPCONV_ALGO"] = "native"

from recon3d_single import (
    setup_device, 
    load_sam3_model, 
    generate_mask_with_sam3, 
    generate_mesh_with_sam3d,
    visualize_masks_and_select 
)

# --- 3. Bridge Class ---
class IsaacSAM3DBridge:
    def __init__(self, text_prompt="cube", output_dir="isaac_output"):
        self.device = setup_device()
        self.text_prompt = text_prompt
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"[Bridge] Loading SAM3 Model (Prompt: {self.text_prompt})...")
        self.sam3_model, self.processor = load_sam3_model(self.device)
        self.sam3d_config = os.path.join(PROJECT_ROOT, "checkpoints", "hf", "pipeline.yaml")

    def process_frame(self, rgb_tensor, frame_idx=0):
        print(f"\n[Bridge] Processing Frame {frame_idx}...")
        
        if rgb_tensor.shape[-1] == 4:
            rgb_tensor = rgb_tensor[..., :3]
        rgb_np = rgb_tensor.cpu().numpy().astype(np.uint8)
        image_pil = Image.fromarray(rgb_np)
        
        image_pil.save(os.path.join(self.output_dir, f"step_{frame_idx}_input.png"))
        
        inference_state = self.processor.set_image(image_pil)
        results = generate_mask_with_sam3(
            image=image_pil,
            processor=self.processor,
            inference_state=inference_state,
            text_prompt=self.text_prompt
        )

        # If no masks found, skip
        if results is None or "scores" not in results or len(results["scores"]) == 0:
            print(f"[Bridge] Warning: No masks found. Skipping.")
            return None

        # Interactive selection window
        print(f"[Bridge] Found {len(results['scores'])} masks. Opening selection window...")
        print(">> Please CLICK on the object you want to reconstruct in the popup window.")
        
        try:
            best_idx = visualize_masks_and_select(
                image=image_pil, 
                inference_state=results, 
                output_dir=self.output_dir
            )
        except Exception as e:
            print(f"[Bridge] Error during selection: {e}. Defaulting to index 0.")
            best_idx = 0

        print(f"[Bridge] Selected mask index: {best_idx}")
        
        mask_binary = (results["masks"][best_idx].squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        
        mesh_output = generate_mesh_with_sam3d(
            image=rgb_np,
            mask=mask_binary,
            config_path=self.sam3d_config,
            seed=42,
            output_dir=self.output_dir,
            generate_mesh=True
        )
        
        if "glb" in mesh_output:
            safe_prompt = self.text_prompt.replace(", ", "_").replace(" ", "")
            save_path = os.path.join(self.output_dir, f"step_{frame_idx}_{safe_prompt}_idx{best_idx}.glb")
            mesh_output["glb"].export(save_path)
            print(f"[Success] Mesh Saved: {save_path}")
            return save_path
        
        return None

# --- 4. Multi-Object Spawning Logic ---
def spawn_objects(obj_type, count, device):
    prompts = []
    
    def get_random_material():
        color = (random.random(), random.random(), random.random())
        return sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.5)

    def get_random_quat():
        roll = torch.tensor([random.uniform(0, np.pi)], device=device)
        pitch = torch.tensor([random.uniform(0, np.pi)], device=device)
        yaw = torch.tensor([random.uniform(0, np.pi)], device=device)
        return quat_from_euler_xyz(roll, pitch, yaw)

    for i in range(count):
        prim_path = f"/World/TargetObj_{i}"
        pos_offset = 0.6 if obj_type != "table" else 1.5 
        pos = ( (i // 2) * pos_offset - (pos_offset/2), (i % 2) * pos_offset - (pos_offset/2), 0.5 )
        
        if obj_type == "table":
            usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
            cfg = UsdFileCfg(usd_path=usd_path)
            sim_utils.spawn_from_usd(prim_path, cfg=cfg, translation=(pos[0], pos[1], 0.0))
            prompts.append("table")
        else:
            material = get_random_material()
            quat = get_random_quat()
            
            if obj_type == "cube":
                cfg = CuboidCfg(size=(0.3, 0.3, 0.3), visual_material=material)
                sim_utils.spawn_cuboid(prim_path, cfg=cfg, translation=pos, orientation=quat)
            elif obj_type == "cone":
                cfg = ConeCfg(radius=0.15, height=0.3, visual_material=material)
                sim_utils.spawn_cone(prim_path, cfg=cfg, translation=pos)
            elif obj_type == "cylinder":
                cfg = CylinderCfg(radius=0.15, height=0.3, visual_material=material)
                sim_utils.spawn_cylinder(prim_path, cfg=cfg, translation=pos, orientation=quat)
            elif obj_type == "sphere":
                cfg = SphereCfg(radius=0.15, visual_material=material)
                sim_utils.spawn_sphere(prim_path, cfg=cfg, translation=pos, orientation=quat)
            
            prompts.append(obj_type)
    
    return ", ".join(list(set(prompts)))

# --- 5. Main Loop ---
def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args_cli.device))
    sim_utils.spawn_ground_plane("/World/ground", cfg=GroundPlaneCfg())
    
    target_prompt = spawn_objects(args_cli.object, args_cli.num_objects, sim.device)
    
    camera_cfg = CameraCfg(
        prim_path="/World/CameraSensor",
        update_period=0,
        height=512, width=512,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955),
    )
    camera = Camera(camera_cfg)
    sim.reset()

    cam_dist = 2.5 if args_cli.object == "table" else 1.5
    cam_z = 2.0 if args_cli.object == "table" else 1.2
    
    camera_eye = torch.tensor([[cam_dist, cam_dist, cam_z]], device=sim.device)
    camera_target = torch.tensor([[0.0, 0.0, 0.5 if args_cli.object == "table" else 0.0]], device=sim.device)
    camera.set_world_poses_from_view(camera_eye, camera_target)

    bridge = IsaacSAM3DBridge(text_prompt=target_prompt)
    print(f"[Sim] Spawned {args_cli.num_objects} {args_cli.object}(s). Prompt: {target_prompt}")

    frame_idx = 0
    reconstruction_done = False
    
    while simulation_app.is_running():
        sim.step()
        
        if frame_idx % 10 == 0:
            camera.update(dt=sim.get_physics_dt())

        if not reconstruction_done and frame_idx >= 100:
            rgb_data = camera.data.output["rgb"]
            single_rgb = rgb_data[0] if rgb_data.dim() == 4 else rgb_data
            
            print("\n[Sim] Inference Triggered! Check the popup window to select mask...")
            mesh_path = bridge.process_frame(single_rgb, frame_idx=frame_idx)
            reconstruction_done = True
            print("[Sim] Inference Done. Keeping simulation running...")
            if mesh_path and os.path.exists(mesh_path):
                print(f"[Sim] Visualizing reconstructed mesh: {mesh_path}")
                mesh = trimesh.load(mesh_path)
                mesh.show()
            else:
                print("[Sim] Mesh path is invalid or generation failed. Skipping visualization.")
            
        
        frame_idx += 1

    simulation_app.close()

if __name__ == "__main__":
    main()
    