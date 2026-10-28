#!/usr/bin/env python3
"""
Check every .pth.tar checkpoint in ./out/<method>/ subfolders and report,
per method, whether training reached and FINISHED the target epoch
(default: 50), i.e. it was not aborted.

A METHOD counts as "finished" when at least one of its checkpoints
(usually the latest/last one, e.g. checkpoint.pth.tar) shows that the target
epoch was completed. The best-model checkpoint may hold an earlier epoch —
that is expected and does NOT count as aborted. Per method, the highest
epoch found across all its checkpoints decides.

If a method only has a best-model checkpoint, its epoch is the best epoch,
not the last one, so completion can't be proven from the files alone.

Usage:
    python check_training_finished.py                 # out/, target 50
    python check_training_finished.py --target 50 --folder out
    python check_training_finished.py --zero-indexed  # if you saved 'epoch' as the loop index

Exit code: 0 if every method finished, 1 otherwise (handy in scripts/CI).
"""

import argparse
import sys
from pathlib import Path

import torch


def load_epoch(path: Path):
    """Return (epoch_or_None, has_model_weights)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        return None, False

    epoch = ckpt.get("epoch")
    if torch.is_tensor(epoch):
        epoch = epoch.item()
    if epoch is not None:
        epoch = int(epoch)

    has_weights = any(k in ckpt for k in ("state_dict", "model_state_dict", "model"))
    return epoch, has_weights


def main():
    p = argparse.ArgumentParser(description="Check that training reached and finished the target epoch.")
    p.add_argument("--folder", type=Path, default=Path("out"), help="Folder with checkpoints (default: out)")
    p.add_argument("--target", type=int, default=50, help="Epoch that must be finished (default: 50)")
    p.add_argument(
        "--zero-indexed",
        action="store_true",
        help="Use if the saved 'epoch' is the 0-based loop index of the last completed epoch "
             "(then epoch 50 finished => saved value 49). Default assumes the saved value is the "
             "number of completed epochs (e.g. torch.save({'epoch': epoch + 1, ...})).",
    )
    args = p.parse_args()

    if not args.folder.is_dir():
        print(f"Error: {args.folder} is not a directory", file=sys.stderr)
        sys.exit(2)

    # out/<method>/.../*.pth.tar  (also picks up files directly in out/)
    files = sorted(args.folder.rglob("*.pth.tar"))
    if not files:
        print(f"No .pth.tar files found in {args.folder} or its subfolders")
        sys.exit(2)

    # Saved value that means "target epoch is done"
    required = args.target - 1 if args.zero_indexed else args.target

    # method name = first folder below out/
    by_method = {}
    for f in files:
        rel = f.relative_to(args.folder)
        method = rel.parts[0] if len(rel.parts) > 1 else "(top level)"
        by_method.setdefault(method, []).append(f)

    print(f"Checking {len(files)} checkpoint(s) in {len(by_method)} method folder(s) under {args.folder}/ "
          f"— target: epoch {args.target} finished\n")

    summary = []  # (method, finished, max_completed, source_file)

    for method, mfiles in sorted(by_method.items()):
        print(f"=== {method} ===")
        best_epoch, best_file = None, None
        for f in mfiles:
            name = f.relative_to(args.folder / method) if method != "(top level)" else f.name
            try:
                epoch, has_weights = load_epoch(f)
            except Exception as e:  # truncated / corrupt file -> save was interrupted
                print(f"  CORRUPT   {name}  -> {type(e).__name__}: {e}")
                continue

            if epoch is None:
                print(f"  NO EPOCH  {name}  (no 'epoch' key)")
                continue

            completed = epoch + 1 if args.zero_indexed else epoch
            note = "" if has_weights else "  [warning: no model weights found]"
            print(f"  epoch {completed:>4}  {name}{note}")
            if best_epoch is None or epoch > best_epoch:
                best_epoch, best_file = epoch, name

        if best_epoch is None:
            summary.append((method, False, None, None))
        else:
            completed = best_epoch + 1 if args.zero_indexed else best_epoch
            summary.append((method, best_epoch >= required, completed, best_file))
        print()

    print("=== Summary (highest epoch per method) ===")
    width = max(len(m) for m, *_ in summary)
    for method, done, completed, src in summary:
        if completed is None:
            print(f"  FAIL {method:<{width}}  no readable checkpoint with an epoch")
        else:
            status = "OK  " if done else "FAIL"
            print(f"  {status} {method:<{width}}  {completed}/{args.target} epochs  (from {src})")
    print()

    all_done = all(done for _, done, _, _ in summary)
    print(f"All methods ran through epoch {args.target}." if all_done
          else f"Some methods did NOT reach epoch {args.target} — see above.\n"
               "(If a method only stores its best checkpoint, check the training log instead.)")
    sys.exit(0 if all_done else 1)


if __name__ == "__main__":
    main()