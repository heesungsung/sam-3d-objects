"""
H5 reader utility: summarize groups, datasets, and attributes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


def _is_numeric(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.number)


def _preview_dataset(data: np.ndarray, max_items: int) -> str:
    if data.size == 0:
        return "[]"
    flat = data.ravel()
    n = min(max_items, flat.size)
    preview = flat[:n]
    if n < flat.size:
        return f"{preview.tolist()} ... (total {flat.size})"
    return f"{preview.tolist()}"


def _print_attrs(obj: h5py.Dataset | h5py.Group, indent: int) -> None:
    if len(obj.attrs) == 0:
        return
    pad = " " * indent
    print(f"{pad}attrs:")
    for key in obj.attrs:
        val = obj.attrs[key]
        print(f"{pad}  - {key}: {val}")


def _summarize_group(
    group: h5py.Group,
    indent: int = 0,
    max_items: int = 5,
) -> None:
    pad = " " * indent
    for name, obj in group.items():
        if isinstance(obj, h5py.Group):
            print(f"{pad}[group] {name}")
            _print_attrs(obj, indent + 2)
            _summarize_group(obj, indent + 2, max_items=max_items)
        elif isinstance(obj, h5py.Dataset):
            shape = obj.shape
            dtype = obj.dtype
            print(f"{pad}[dataset] {name} shape={shape} dtype={dtype}")
            _print_attrs(obj, indent + 2)
            if _is_numeric(dtype) and obj.size <= 100000:
                data = obj[()]
                print(f"{pad}  preview: {_preview_dataset(np.array(data), max_items)}")
        else:
            print(f"{pad}[unknown] {name} type={type(obj)}")


def summarize_h5(path: Path, max_items: int = 5) -> None:
    print(f"\n=== {path} ===")
    if not path.exists():
        print("file not found")
        return
    with h5py.File(path, "r") as f:
        _print_attrs(f, 0)
        _summarize_group(f, indent=0, max_items=max_items)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize H5 files.")
    parser.add_argument("paths", nargs="+", help="H5 file paths")
    parser.add_argument("--max-items", type=int, default=5)
    args = parser.parse_args(list(argv) if argv is not None else None)

    for p in args.paths:
        summarize_h5(Path(p), max_items=args.max_items)


if __name__ == "__main__":
    main()
