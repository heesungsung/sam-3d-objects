#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from inference import interactive_visualizer

try:
    import open3d as o3d
    from PIL import Image
    import OpenEXR, Imath
    from concurrent.futures import ThreadPoolExecutor
    RENDERING_AVAILABLE = True
except ImportError as e:
    RENDERING_AVAILABLE = False
    RENDERING_ERROR = str(e)


def is_gaussian_splatting_ply(file_path: str) -> bool:
    """
    Check if PLY file is Gaussian Splatting format by reading header.

    Args:
        file_path: Path to PLY file
    
    Returns:
        True if Gaussian Splatting format, False if regular pointcloud
    """
    try:
        with open(file_path, 'rb') as f:
            header_bytes = f.read(8192)
            header = header_bytes.decode('ascii', errors='ignore')
            
            gs_properties = ['opacity', 'f_dc_0', 'scale_0', 'rot_0']
            has_gs_properties = any(prop in header for prop in gs_properties)
            
            return has_gs_properties

    except Exception:
        return False


def mesh_error_handler(file_ext: str):
    print(f"\nYou can manually open the mesh file with a 3D viewer:")
    print(f"  - Blender: File > Import > {file_ext.upper()}")
    print(f"  - MeshLab: File > Import Mesh")
    print(f"  - Online: https://3dviewer.net")


# ============================================================
#                Mitsuba Rendering Templates
# ============================================================

XML_HEAD_IMAGE = """<scene version="0.6.0">
    <integrator type="path"><integer name="maxDepth" value="-1"/></integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        <sampler type="ldsampler"><integer name="sampleCount" value="256"/></sampler>
        <film type="hdrfilm">
            <integer name="width" value="800"/>
            <integer name="height" value="800"/>
            <rfilter type="gaussian"/>
            <boolean name="banner" value="false"/>
        </film>
    </sensor>
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/>
    </bsdf>
"""

XML_HEAD_VIDEO = """
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="{origin_x},{origin_y},{origin_z}" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="{fov}"/>
        <sampler type="ldsampler">
            <integer name="sampleCount" value="{sample_count}"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="{width}"/>
            <integer name="height" value="{height}"/>
            <rfilter type="gaussian"/>
            <boolean name="banner" value="false"/>
        </film>
    </sensor>

    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/>
    </bsdf>
"""

XML_POINT = """    <shape type="sphere">
        <float name="radius" value="{radius}"/>
        <transform name="toWorld">
            <translate x="{x}" y="{y}" z="{z}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{r},{g},{b}"/>
        </bsdf>
    </shape>
"""

XML_TAIL = """    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.5"/>
        </transform>
    </shape>
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area"><rgb name="radiance" value="6,6,6"/></emitter>
    </shape>
</scene>
"""


# ============================================================
#                Rendering Functions
# ============================================================

def convert_exr_to_jpg(exr_path, jpg_path):
    """
    Convert a single Mitsuba EXR output to JPG.

    Args:
        exr_path: Path to EXR file
        jpg_path: Path to JPG file
    """
    file = OpenEXR.InputFile(exr_path)
    pix_type = Imath.PixelType(Imath.PixelType.FLOAT)
    dw = file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    rgb = [np.frombuffer(file.channel(c, pix_type), dtype=np.float32) for c in 'RGB']

    for i in range(3):
        rgb[i] = np.where(rgb[i] <= 0.0031308,
                          (rgb[i] * 12.92) * 255.0,
                          (1.055 * (rgb[i] ** (1.0 / 2.4)) - 0.055) * 255.0)
    
    rgb8 = [Image.frombytes("F", size, c.tobytes()).convert("L") for c in rgb]
    Image.merge("RGB", rgb8).save(jpg_path, "JPEG", quality=95)


def render_image_from_ply(ply_file, output_dir=None):
    """
    Render a single PLY file to JPG image using Mitsuba.
    
    Args:
        ply_file: Path to PLY file
        output_dir: Output directory (default: {input_dir}/vis)

    Returns:
        jpg_path: Path to JPG file
    """
    if not RENDERING_AVAILABLE:
        raise ImportError(f"Rendering dependencies not available: {RENDERING_ERROR}")
    
    ply_file = os.path.abspath(ply_file)
    input_dir = os.path.dirname(ply_file) or "."
    
    if output_dir is None:
        output_dir = os.path.join(input_dir, "vis")
    else:
        output_dir = os.path.abspath(output_dir)
    
    xml_dir = os.path.join(output_dir, "xml_new")
    exr_dir = os.path.join(output_dir, "exr_file")
    jpg_dir = os.path.join(output_dir, "jpg_file")
    
    os.makedirs(xml_dir, exist_ok=True)
    os.makedirs(exr_dir, exist_ok=True)
    os.makedirs(jpg_dir, exist_ok=True)
    
    # Load pointcloud
    pcd = o3d.io.read_point_cloud(ply_file)
    pts, colors = np.asarray(pcd.points), np.asarray(pcd.colors)
    
    # Generate XML
    xml_segments = [XML_HEAD_IMAGE]
    for p, c in zip(pts, colors):
        xml_segments.append(XML_POINT.format(
            radius=0.025,
            x=p[2], y=p[0], z=p[1],
            r=c[0], g=c[1], b=c[2]
        ))
    xml_segments.append(XML_TAIL)
    
    base_name = os.path.splitext(os.path.basename(ply_file))[0]
    xml_path = os.path.join(xml_dir, f"{base_name}.xml")
    exr_path = os.path.join(exr_dir, f"{base_name}.exr")
    jpg_path = os.path.join(jpg_dir, f"{base_name}.jpg")
    
    with open(xml_path, "w") as fxml:
        fxml.write("".join(xml_segments))
    print(f"[XML] Generated: {xml_path}")
    
    # Render with Mitsuba
    print(f"[Render] Rendering {xml_path} → {exr_path}...")
    subprocess.run(["mitsuba", "-o", exr_path, xml_path], check=True)
    
    print(f"[Convert] Converting {exr_path} → {jpg_path}...")
    convert_exr_to_jpg(exr_path, jpg_path)
    print(f"[DONE] Image saved to: {jpg_path}")

    return jpg_path


def render_video_from_ply(
    ply_file,
    output_dir=None,
    frames=200,
    radius=3.5,
    fps=24,
    width=800,
    height=800,
    point_radius=0.025,
    fov=25,
    sample_count=256,
    workers=16
):
    """
    Render a rotating GIF from a PLY point cloud using Mitsuba.
    
    Args:
        ply_file: Path to PLY file
        output_dir: Output directory (default: {input_dir}/vis)
        frames: Number of frames to render
        radius: Rotation radius of the camera
        fps: Frames per second for the GIF
        width, height: Image resolution
        point_radius: Radius of each rendered point (sphere)
        fov: Camera fov
        sample_count: Mitsuba sample count per pixel
        workers: Number of parallel threads for rendering

    Returns:
        gif_path: Path to GIF file
    """
    if not RENDERING_AVAILABLE:
        raise ImportError(f"Rendering dependencies not available: {RENDERING_ERROR}")
    
    ply_file = os.path.abspath(ply_file)
    input_dir = os.path.dirname(ply_file) or "."
    
    if output_dir is None:
        output_dir = os.path.join(input_dir, "vis")
    else:
        output_dir = os.path.abspath(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(ply_file))[0]
    gif_path = os.path.join(output_dir, f"{base_name}.gif")
    temp_dir = os.path.join(output_dir, "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Load pointcloud
    pcd = o3d.io.read_point_cloud(ply_file)
    points, colors = np.asarray(pcd.points), np.asarray(pcd.colors)
    
    # Generate all XML frames
    xml_params = dict(width=width, height=height, fov=fov,
                      sample_count=sample_count, point_radius=point_radius)
    xml_files = []

    for i in range(frames):
        theta = 2 * np.pi * i / frames
        cam_pos = [radius * np.cos(theta), radius * np.sin(theta), 3.0]
        xml_segments = [XML_HEAD_VIDEO.format(
            origin_x=cam_pos[0], origin_y=cam_pos[1], origin_z=cam_pos[2],
            **xml_params
        )]

        for p, c in zip(points, colors):
            xml_segments.append(XML_POINT.format(
                radius=point_radius,
                x=p[2], y=p[0], z=p[1],
                r=c[0], g=c[1], b=c[2]
            ))

        xml_segments.append(XML_TAIL)
        xml_path = os.path.join(temp_dir, f"frame_{i:03d}.xml")

        with open(xml_path, "w") as f:
            f.write("".join(xml_segments))

        xml_files.append(xml_path)
        print(f"[Frame {i+1}/{frames}] Scene saved → {xml_path}")
    

    def render_single_frame(xml_file):
        exr_output = xml_file.replace(".xml", ".exr")
        jpg_output = xml_file.replace(".xml", ".jpg")
        subprocess.run(["mitsuba", "-o", exr_output, xml_file], check=True)
        convert_exr_to_jpg(exr_output, jpg_output)

        return jpg_output
    

    # Parallel rendering
    print(f"[INFO] Start rendering {frames} frames using {workers} workers...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        jpg_files = list(executor.map(render_single_frame, xml_files))
    
    # Assemble GIF
    jpg_files.sort()
    imgs = [Image.open(p) for p in jpg_files]
    duration = 1000 // fps
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=duration, loop=0)
    print(f"[DONE] GIF saved → {gif_path}")
    
    # Clean up intermediate files
    print(f"[Cleanup] Removing intermediate files from {temp_dir}...")
    for xml_file in xml_files:
        exr_file = xml_file.replace(".xml", ".exr")
        jpg_file = xml_file.replace(".xml", ".jpg")
        try:
            if os.path.exists(xml_file):
                os.remove(xml_file)
            if os.path.exists(exr_file):
                os.remove(exr_file)
            if os.path.exists(jpg_file):
                os.remove(jpg_file)
        except Exception as e:
            print(f"  Warning: Could not remove {xml_file}: {e}")
    
    # Remove temp directory
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass
    print(f"[DONE] Cleanup complete. Final GIF: {gif_path}")
    
    return gif_path


def visualize_file(file_path: str):
    """
    Visualize 3D file (PLY for GS, GLB/OBJ/STL for mesh)
    
    Args:
        file_path: Path to 3D file
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    if not os.path.isfile(file_path):
        print(f"Error: Not a file: {file_path}")
        sys.exit(1)
    
    file_path = os.path.abspath(file_path)
    file_ext = os.path.splitext(file_path)[1].lower().lstrip('.')
    
    print(f"\n{'='*60}")
    print(f"Visualizing 3D file: {file_path}")
    print(f"File type: {file_ext.upper()}")
    print(f"{'='*60}\n")
    
    try:
        if file_ext == "ply":
            # Gaussian Splatting - use interactive_visualizer (Gradio Model3D)
            if is_gaussian_splatting_ply(file_path):
                print("Detected: Gaussian Splatting (PLY)")
                print("Using interactive visualizer (Gradio)...")
                print("This will open a web browser. Close the browser window to stop.\n")
                interactive_visualizer(file_path)

            # PLY file: Regular point cloud - use open3d or trimesh viewer
            else:
                print("Detected: Point Cloud (PLY)")
                print("Loading point cloud...")
                
                try:
                    import open3d as o3d
                    import numpy as np
                    
                    pcd = o3d.io.read_point_cloud(file_path)
                    print(f"  Points: {len(pcd.points)}")
                    print(f"  Has colors: {len(pcd.colors) > 0}")
                    print(f"  Has normals: {len(pcd.normals) > 0}")
                    
                    points_array = np.asarray(pcd.points)
                    if len(points_array) == 0:
                        print("  No points to visualize")
                        return
                    
                    bbox_size = np.max(points_array, axis=0) - np.min(points_array, axis=0)
                    avg_bbox_size = np.mean(bbox_size)
                    sphere_radius = max(min(avg_bbox_size * 0.01, 0.01), 0.0005)
                                        
                    # Sample points
                    max_points_to_visualize = 5000
                    if len(points_array) > max_points_to_visualize:
                        print(f"  Too many points ({len(points_array)}), sampling {max_points_to_visualize} points...")
                        indices = np.random.choice(len(points_array), max_points_to_visualize, replace=False)
                        points_array = points_array[indices]
                        colors_array = np.asarray(pcd.colors)[indices] if len(pcd.colors) > 0 else None

                    else:
                        colors_array = np.asarray(pcd.colors) if len(pcd.colors) > 0 else None
                    
                    sphere_meshes = []
                    for i, point in enumerate(points_array):
                        # Create a new sphere for each point (Open3D doesn't have copy())
                        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=8)
                        sphere.translate(point)

                        if colors_array is not None:
                            color = colors_array[i]
                            sphere.paint_uniform_color(color)
                        sphere_meshes.append(sphere)
                    
                    combined_mesh = sphere_meshes[0]
                    for sphere in sphere_meshes[1:]:
                        combined_mesh += sphere
                    
                    print("Opening Open3D viewer with sphere visualization...")
                    print("  (Close the viewer window to continue)")
                    print("  Controls:")
                    print("    - Mouse drag: Rotate")
                    print("    - Mouse wheel: Zoom")
                    print("    - Ctrl + drag: Pan")
                    
                    o3d.visualization.draw_geometries(
                        [combined_mesh],
                        window_name="Point Cloud Visualization (Spheres)",
                        width=1024,
                        height=768
                    )

                except ImportError:
                    import trimesh
                    pcd = trimesh.load(file_path)

                    if hasattr(pcd, 'vertices'):
                        print(f"  Points: {len(pcd.vertices)}")
                        print("Opening trimesh viewer...")
                        print("  (Close the viewer window to continue)")
                        pcd.show()

                    else:
                        raise ValueError("Could not load point cloud")

        else:
            # Other formats (GLB, OBJ, STL, etc.): Mesh - use trimesh viewer
            print(f"Detected: Mesh ({file_ext.upper()})")
            print("Loading mesh...")
            
            import trimesh
            mesh = trimesh.load(file_path)
            
            if isinstance(mesh, trimesh.Scene):
                # If Scene, use first geometry
                geometries = list(mesh.geometry.values())
                if geometries:
                    mesh = geometries[0]
                    print(f"  Loaded first geometry from scene")

                else:
                    raise ValueError("No geometry found in scene")
            
            print(f"  Vertices: {len(mesh.vertices)}")
            print(f"  Faces: {len(mesh.faces) if hasattr(mesh, 'faces') else 'N/A'}")
            print("Opening trimesh viewer...")
            print("  (Close the viewer window to continue)")
            mesh.show()
            
    except KeyboardInterrupt:
        print("\n\nVisualization stopped by user.")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        mesh_error_handler(file_ext)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="3D file visualizer and renderer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog= """
                Usage:
                python visualization.py --input (file_path).(extension)
                python visualization.py --input (file_path).ply --render-image
                python visualization.py --input (file_path).ply --render-video
                
                Supported formats:
                - PLY: Gaussian Splatting or Pointcloud
                  - Gaussian Splatting: uses Gradio interactive viewer
                  - Pointcloud: uses Open3D or trimesh viewer
                - GLB, OBJ, STL, etc.: Mesh (uses trimesh viewer)
                
                Rendering options (for PLY files):
                - --render-image: Render single image using Mitsuba
                - --render-video: Render rotating GIF using Mitsuba
                """
    )
    
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default=None,
        help="3D input file path (PLY, GLB, OBJ, STL, etc.)"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        dest="input_alt",
        default=None,
        help="3D input file path (PLY, GLB, OBJ, STL, etc.)"
    )
    
    # Rendering options
    parser.add_argument(
        "--render-image",
        action="store_true",
        help="Render PLY file to JPG image using Mitsuba"
    )
    
    parser.add_argument(
        "--render-video",
        action="store_true",
        help="Render PLY file to rotating GIF using Mitsuba"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for rendered images/videos (default: same as input)"
    )
    
    # Video rendering parameters
    parser.add_argument(
        "--frames",
        type=int,
        default=200,
        help="Number of frames for video rendering (default: 200)"
    )
    
    parser.add_argument(
        "--radius",
        type=float,
        default=3.5,
        help="Camera rotation radius for video (default: 3.5)"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frame rate for video GIF (default: 24)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel rendering threads (default: 16)"
    )
    
    args = parser.parse_args()
    file_path = args.input or args.input_alt
    
    if file_path is None:
        parser.print_help()
        print("\nError: Please provide a 3D file path.")
        sys.exit(1)
    
    if args.render_image or args.render_video:
        if not RENDERING_AVAILABLE:
            print(f"Error: Rendering dependencies not available: {RENDERING_ERROR}")
            print("Please install: open3d, pillow, OpenEXR")
            sys.exit(1)
        
        file_ext = os.path.splitext(file_path)[1].lower().lstrip('.')
        if file_ext != "ply":
            print("Error: Rendering is only supported for PLY files.")
            sys.exit(1)
        
        if args.render_image:
            print(f"\n{'='*60}")
            print(f"Rendering image from: {file_path}")
            print(f"{'='*60}\n")
            render_image_from_ply(file_path, args.output_dir)
        
        if args.render_video:
            print(f"\n{'='*60}")
            print(f"Rendering video from: {file_path}")
            print(f"{'='*60}\n")
            render_video_from_ply(
                file_path,
                output_dir=args.output_dir,
                frames=args.frames,
                radius=args.radius,
                fps=args.fps,
                workers=args.workers
            )
    else:
        visualize_file(file_path)


if __name__ == "__main__":
    main()