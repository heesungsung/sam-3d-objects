#!/usr/bin/env python3
"""
PLY mesh file visualizer

Usage:
    python visualize_mesh.py <ply_file_path>
    python visualize_mesh.py --ply <ply_file_path>
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import interactive_visualizer from inference.py
from inference import interactive_visualizer


def main():
    parser = argparse.ArgumentParser(
        description="PLY mesh file visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage:
  python visualize_mesh.py mesh.ply
  python visualize_mesh.py --ply mesh.ply
  python visualize_mesh.py /path/to/mesh.ply
        """
    )
    
    parser.add_argument(
        "ply",
        type=str,
        nargs="?",
        default=None,
        help="PLY mesh file path"
    )
    
    parser.add_argument(
        "--ply",
        type=str,
        dest="ply_alt",
        default=None,
        help="PLY mesh file path (--ply option)"
    )
    
    args = parser.parse_args()
    
    # Determine file path
    ply_path = args.ply or args.ply_alt
    
    if ply_path is None:
        parser.print_help()
        print("\nError: Please provide the PLY file path.")
        sys.exit(1)
    
    # Check if file exists
    if not os.path.exists(ply_path):
        print(f"Error: File not found: {ply_path}")
        sys.exit(1)
    
    if not os.path.isfile(ply_path):
        print(f"Error: Not a file: {ply_path}")
        sys.exit(1)
    
    # Convert to absolute path
    ply_path = os.path.abspath(ply_path)
    
    print(f"\n{'='*60}")
    print(f"Visualizing mesh: {ply_path}")
    print(f"{'='*60}\n")
    print("Launching interactive visualizer...")
    print("This will open a web browser. Close the browser window to stop.\n")
    
    # Run interactive_visualizer
    try:
        interactive_visualizer(ply_path)
    except KeyboardInterrupt:
        print("\n\nVisualization stopped by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

