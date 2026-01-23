import os
import sys
import json
import textwrap
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

GEAL_PATH = os.path.join(PROJECT_ROOT, "third_party", "geal")
if GEAL_PATH not in sys.path:
    sys.path.insert(0, GEAL_PATH)

from google import genai
from google.genai import types
from recon3d_single import recond3d_single
from dataset.data_utils import CLASSES, AFFORDANCES
from afford3d_single import afford3d_single


def parse_json(json_output: str) -> str:
    """
    Parse JSON output by removing markdown code fencing.
    Based on gemini-robotics-er.ipynb.
    """
    lines = json_output.splitlines()

    for i, line in enumerate(lines):
        if line == "```json":
            # Remove everything before "```json"
            json_output = "\n".join(lines[i + 1 :])
            # Remove everything after the closing "```"
            json_output = json_output.split("```")[0]
            break  # Exit the loop once "```json" is found

    return json_output


def plot_bounding_boxes(img: Image.Image, bounding_boxes_json: str) -> Image.Image:
    """
    Plot bounding boxes on an image.
    Based on gemini-robotics-er.ipynb, simplified for single object.
    
    Args:
        img: PIL Image object
        bounding_boxes_json: JSON string with bounding boxes in format:
            [{"box_2d": [ymin, xmin, ymax, xmax], "label": "object_name"}]
            normalized to 0-1000
    
    Returns:
        PIL Image with bounding boxes drawn
    """
    bounding_boxes_str = parse_json(bounding_boxes_json)
    
    try:
        bounding_boxes = json.loads(bounding_boxes_str)

    except json.JSONDecodeError as e:
        print(f"Error parsing bounding boxes JSON: {e}")
        return img
    
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    width, height = img_copy.size
    colors = ["red", "green", "blue", "yellow", "orange", "pink", "purple"]
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)

    except:
        font = ImageFont.load_default()
    
    # Draw bbox
    for i, bbox_data in enumerate(bounding_boxes):
        if "box_2d" not in bbox_data:
            continue
        
        box_2d = bbox_data["box_2d"]
        label = bbox_data.get("label", f"Object {i+1}")
        
        # Unnormalize
        ymin = int(box_2d[0] / 1000 * height)
        xmin = int(box_2d[1] / 1000 * width)
        ymax = int(box_2d[2] / 1000 * height)
        xmax = int(box_2d[3] / 1000 * width)
        
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        
        color = colors[i % len(colors)]
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=4)
        
        if label:
            bbox_text = draw.textbbox((0, 0), label, font=font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
            
            draw.rectangle(
                [xmin, ymin - text_height - 4, xmin + text_width + 4, ymin],
                fill=color
            )
            draw.text((xmin + 2, ymin - text_height - 2), label, fill="white", font=font)
    
    return img_copy


def convert_bbox_to_xywh(box_2d: List[int], image_width: int, image_height: int) -> List[float]:
    """
    Convert Gemini bbox format [ymin, xmin, ymax, xmax] (normalized 0-1000) 
    to SAM3 format [x, y, w, h] (absolute pixel coordinates).
    
    Args:
        box_2d: [ymin, xmin, ymax, xmax] normalized to 0-1000
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        [x, y, w, h] in absolute pixel coordinates
    """
    if len(box_2d) != 4:
        raise ValueError(f"box_2d must have 4 elements [ymin, xmin, ymax, xmax], got {len(box_2d)}")
    
    ymin, xmin, ymax, xmax = box_2d
    
    # Unnormalize
    ymin_abs = int(ymin / 1000.0 * image_height)
    xmin_abs = int(xmin / 1000.0 * image_width)
    ymax_abs = int(ymax / 1000.0 * image_height)
    xmax_abs = int(xmax / 1000.0 * image_width)
    
    if xmin_abs > xmax_abs:
        xmin_abs, xmax_abs = xmax_abs, xmin_abs
    if ymin_abs > ymax_abs:
        ymin_abs, ymax_abs = ymax_abs, ymin_abs
    
    x = float(xmin_abs)
    y = float(ymin_abs)
    w = float(xmax_abs - xmin_abs)
    h = float(ymax_abs - ymin_abs)
    
    return [x, y, w, h]


def find_object_with_gemini(image_path: str, user_prompt: str) -> Tuple[List[dict], str]:
    """
    Use Gemini to find a single object in image based on user prompt.
    Based on gemini-robotics-er.ipynb Cell 47, but modified to find only one object.
    
    Args:
        image_path: Path to input image
        user_prompt: User prompt (e.g., "Grasp the purple bottle on the table")
        
    Returns:
        Tuple of (bboxes list, output image path)
    """
    print(f"\n=== Step 0: Finding objects with Gemini ===")
    print(f"User prompt: '{user_prompt}'")
    
    classes_str = ", ".join([f'"{c}"' for c in CLASSES])
    affordances_str = ", ".join([f'"{a}"' for a in AFFORDANCES])
    
    gemini_prompt = textwrap.dedent(f"""\
        You are given a user request: "{user_prompt}"
        
        Analyze the user request to identify the specific object that the user wants to interact with.
        Consider the main object nouns, descriptive attributes, spatial relationships, and positional qualifiers.
        
        Find the bounding box of the target object in the image. Return only ONE bounding box for the primary object mentioned in the user request.
        
        Return bounding box as a JSON array with label, class, and affordance. Never return masks or code fencing.
        The format should be as follows:
        [{{"box_2d": [ymin, xmin, ymax, xmax], "label": <label for the object>, "class": <object class>, "affordance": <affordance>}}]
        normalized to 0-1000. The values in box_2d must only be integers.
        
        Important for the label:
        - The label should be concise and contain only the key identifying features of the object
        - Include: object type (e.g., "bottle", "cup", "box") and key distinguishing attributes (e.g., "purple", "red", "large")
        - Exclude: spatial relationships (e.g., "on the table", "leftmost"), action verbs (e.g., "grasp", "pick up"), and unnecessary words
        - Examples: "purple bottle", "red cup", "blue box", "large book" (not "the purple bottle on the table" or "grasp the purple bottle")
        
        Important for class:
        - "class" should be the object class name
        - If the object matches one of these predefined classes, use it: {classes_str}
        - If the object doesn't match any predefined class, use an appropriate class name
        - Use standard capitalization: capitalize the first letter of each word (e.g., "Bowl", "Bottle", "Chair", "StorageFurniture")
        
        Important for affordance:
        - "affordance" MUST be one of the following predefined affordances: {affordances_str}
        - Select the affordance that best matches the action described in the user request
        - If the action doesn't match any affordance exactly, choose the closest match
        - Use the exact spelling as shown in the list (all lowercase)
        - If the user request does not contain any affordance-related information, use "grasp" as the default affordance
        
        If the object is not found, return an empty array [].
    """)
    
    client = genai.Client()
    image = Image.open(image_path)
    image_width, image_height = image.size
    print(f"Image size: {image_width}x{image_height}")
    label = None
    
    # Call Gemini
    print("\nCalling Gemini model...")
    image_response = client.models.generate_content(
        model="gemini-robotics-er-1.5-preview",
        contents=[
            types.Part.from_bytes(
                data=open(image_path, 'rb').read(),
                mime_type='image/jpeg' if image_path.lower().endswith(('.jpg', '.jpeg')) else 'image/png',
            ),
            gemini_prompt
        ],
        config=types.GenerateContentConfig(
            temperature=0.1,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
    )
    
    response_text = image_response.text
    print(f"Gemini response: {response_text}")
    
    try:
        json_str = parse_json(response_text)
        bboxes = json.loads(json_str)

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Response text: {response_text}")
        return [], None
    
    if not bboxes:
        print("Warning: No bounding boxes found in Gemini response")
        return [], None
    
    assert len(bboxes) == 1, f"Expected exactly 1 bounding box, but found {len(bboxes)}"
    
    bbox_info = bboxes[0]
    label = bbox_info.get("label", "Object 1")
    class_name = bbox_info.get("class", None)
    affordance = bbox_info.get("affordance", None)
    box_2d = bbox_info.get("box_2d", [])
    
    print(f"Found object:")
    print(f"  label: {label}")
    print(f"  class: {class_name}")
    print(f"  affordance: {affordance}")
    print(f"  box_2d: {box_2d}")
    
    # Draw bboxes on image
    file_name = label.replace(" ", "_")
    output_dir = os.path.join(os.path.dirname(image_path), f"{file_name}")
    
    # Save user_prompt and object info to txt file
    tmp_dir = os.path.join(output_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    info_path = os.path.join(tmp_dir, "vlm_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"User Prompt: {user_prompt}\n\n")
        f.write(f"Response Summary:\n")
        f.write(f"  label: {label}\n")
        f.write(f"  class: {class_name}\n")
        f.write(f"  affordance: {affordance}\n")
        f.write(f"  box_2d: {box_2d}\n")
    print(f"\nSaved VLM info to: {info_path}")

    # Save vlm_output overlay image
    output_image_path = os.path.join(tmp_dir, "vlm_output.png")
    annotated_img = plot_bounding_boxes(image, response_text)
    annotated_img.save(output_image_path)
    print(f"Saved bbox visualization to: {output_image_path}")
    
    return bboxes, output_dir


def process_with_sam3d(image_path: str, bboxes: List[dict], sam3d_config_path: Optional[str] = None, output_dir: Optional[str] = None):
    """
    Process image with SAM3D using bounding boxes from Gemini
    
    Args:
        image_path: Path to input image
        bboxes: List of bbox dicts from Gemini (format: {"box_2d": [ymin, xmin, ymax, xmax], "label": "..."})
        sam3d_config_path: Path to SAM3D config file
        output_dir: Output directory for SAM3D results
    """
    print(f"\nMoving to SAM3D...")
    
    image = Image.open(image_path)
    image_width, image_height = image.size
    
    sam3_boxes = []
    for bbox_info in bboxes:
        if "box_2d" not in bbox_info:
            print(f"  Warning: Invalid bbox format, skipping: {bbox_info}")
            continue
        
        box_2d = bbox_info["box_2d"]  # [ymin, xmin, ymax, xmax] normalized 0-1000

        try:
            x, y, w, h = convert_bbox_to_xywh(box_2d, image_width, image_height)
            sam3_boxes.append([x, y, w, h])
            label = bbox_info.get("label", "unknown")
            print(f"  Converted bbox ({label}): [{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}]")

        except Exception as e:
            print(f"  Warning: Failed to convert bbox {box_2d}: {e}")
            continue
    
    if not sam3_boxes:
        print("Error: No valid bounding boxes to process")
        return
    
    # Process with SAM3D
    print(f"\nProcessing first bounding box (out of {len(sam3_boxes)})...")
    box_labels = [True] * len(sam3_boxes)
    
    _, saved_paths = recond3d_single(
        image_path=image_path,
        boxes=[sam3_boxes[0]],  # Process first box
        box_labels=[box_labels[0]],
        sam3d_config_path=sam3d_config_path,
        output_dir=output_dir,
        save_mask=True,
        save_3d=True,
        save_pose=True,
        generate_mesh=True,
        is_vlm=True,
    )
    
    return saved_paths


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Gemini + SAM3D + GEAL Pipeline")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input image")
    parser.add_argument("--prompt", type=str, required=True,
                       help="User prompt (e.g., 'Grasp the purple bottle on the table')")
    parser.add_argument("--sam3d-config", type=str, default=None,
                       help="Path to SAM3D config file (optional)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (optional)")
    parser.add_argument("--no-afford", action="store_true",
                       help="Skip affordance map generation using GEAL (default: generate affordance)")
    
    args = parser.parse_args()
    
    bboxes, output_dir = find_object_with_gemini(args.input, args.prompt)
    if args.output_dir is not None:
        output_dir = args.output_dir
    
    if not bboxes:
        print("No objects found. Exiting.")
        return
    
    if args.sam3d_config is None:
        default_config = os.path.join(PROJECT_ROOT, "checkpoints", "hf", "pipeline.yaml")
        if os.path.exists(default_config):
            args.sam3d_config = default_config
            print(f"Using default SAM3D config: {args.sam3d_config}")
        else:
            print("Warning: SAM3D config not found. Mesh generation will be skipped.")
            print(f"  Expected at: {default_config}")

    saved_paths = process_with_sam3d(
        image_path=args.input,
        bboxes=bboxes,
        sam3d_config_path=args.sam3d_config,
        output_dir=output_dir,
    )
    
    if not args.no_afford:
        print(f"\n=== Step 6: Generating affordance map ===")
        
        bbox_info = bboxes[0]
        object_class = bbox_info.get("class", None)
        affordance = bbox_info.get("affordance", None)
        
        if object_class is None or affordance is None:
            print(f"Warning: Could not find class or affordance in bbox info")
            print("Skipping affordance generation.")
            return
        
        ply_path = saved_paths.get("gs", None)
        
        if not os.path.exists(ply_path):
            print(f"Warning: PLY file not found at {ply_path}")
            print("Skipping affordance generation.")
            return
        
        print(f"Using PLY file: {ply_path}")
        print(f"Object class: {object_class}")
        print(f"Affordance: {affordance}")
        
        try:
            afford3d_single(
                ply_path=ply_path,
                affordance=affordance,
                object_class=object_class,
            )
        except Exception as e:
            print(f"Error during affordance generation: {e}")
            import traceback
            traceback.print_exc()
    
    
if __name__ == "__main__":
    main()
