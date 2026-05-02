#!/usr/bin/env python3
"""
Check the 'epoch' value inside every .pth.tar checkpoint in ./saved_checkpoints
and report files whose epoch does NOT match an expected value.

Usage:
    python check_epochs.py <expected_epochs>

Example:
    python check_epochs.py 100
"""

import argparse
import sys
from pathlib import Path

import torch

FOLDER = Path("saved_checkpoints")


def get_epoch(checkpoint_path: Path):
    """Load a .pth.tar file and return its 'epoch' value, or None if missing."""
    # weights_only=False is needed because checkpoints often contain
    # non-tensor objects (optimizer state, epoch ints, etc.)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Most checkpoints are dicts with an 'epoch' key
    if isinstance(ckpt, dict) and "epoch" in ckpt:
        return ckpt["epoch"]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Find .pth.tar files in ./saved_checkpoints whose epoch count != expected value."
    )
    parser.add_argument("expected", type=int, help="Expected number of epochs")
    args = parser.parse_args()

    if not FOLDER.is_dir():
        print(f"Error: {FOLDER} is not a directory", file=sys.stderr)
        sys.exit(1)

    pth_files = sorted(FOLDER.glob("*.pth.tar"))

    if not pth_files:
        print(f"No .pth.tar files found in {FOLDER}")
        return

    mismatches = []
    missing_key = []
    errors = []

    for f in pth_files:
        try:
            epoch = get_epoch(f)
            if epoch is None:
                missing_key.append(f)
            elif epoch != args.expected:
                mismatches.append((f, epoch))
        except Exception as e:
            errors.append((f, str(e)))

    print(f"Checked {len(pth_files)} file(s). Expected epoch = {args.expected}\n")

    if mismatches:
        print(f"Files with mismatched epoch ({len(mismatches)}):")
        for f, epoch in mismatches:
            print(f"  {f}  (epoch={epoch})")
        print()

    if missing_key:
        print(f"Files with no 'epoch' key ({len(missing_key)}):")
        for f in missing_key:
            print(f"  {f}")
        print()

    if errors:
        print(f"Files that failed to load ({len(errors)}):")
        for f, err in errors:
            print(f"  {f}  -> {err}")
        print()

    if not mismatches and not missing_key and not errors:
        print("All files match the expected epoch count.")


if __name__ == "__main__":
    main()