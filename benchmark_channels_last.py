#!/usr/bin/env python3
"""Benchmarks NCHW vs channels_last on the real nn-lpf ResNet-18 models.

Place this file in the root of your nn-lpf repository, beside main.py.
Training benchmark only: synthetic 64x64 data; no dataloader or validation.
"""
import argparse
import csv
import gc
import os
import time
from pathlib import Path

# Set this BEFORE importing torch and initializing CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import aa_models


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", nargs="+", default=[
        "none", "blur", "pasa_efficient", "dwt", "dab", "asap_sp"
    ])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--batch-accum", type=int, default=2)
    p.add_argument("--warmup", type=int, default=5, help="optimizer updates before measuring")
    p.add_argument("--steps", type=int, default=20, help="measured optimizer updates")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fused-sgd", action="store_true")
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--gpu", type=int, default=0, help="visible CUDA device index")
    p.add_argument("--filter-size", type=int, default=3)
    p.add_argument("--pasa-group", type=int, default=8)
    p.add_argument("--wavelet-type", default="db2")
    p.add_argument("--output", default="channels_last_results.csv")
    return p.parse_args()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_one(method, layout, a, device):
    # Re-seed both cases so weights/inputs/labels start the same.
    torch.manual_seed(2026)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(2026)

    model = aa_models.resnet18(
        aa_type=method, filter_size=a.filter_size,
        pasa_group=a.pasa_group, wavelet_type=a.wavelet_type,
        num_classes=200,
    ).to(device)
    if layout == "channels_last":
        model = model.to(memory_format=torch.channels_last)

    images = torch.randn(a.batch_size, 3, 64, 64, device=device)
    if layout == "channels_last":
        images = images.contiguous(memory_format=torch.channels_last)
    labels = torch.randint(0, 200, (a.batch_size,), device=device)

    # Initialize input-shape-dependent internal state before compilation,
    # particularly ASAP's lazily generated Fourier window.
    model.eval()
    with torch.no_grad():
        _ = model(images)
    sync(device)
    model.train()

    if a.compile:
        model.compile(mode="default")

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.2, momentum=0.9,
        weight_decay=1e-4, nesterov=True, fused=a.fused_sgd,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp and device.type == "cuda")

    def update():
        optimizer.zero_grad(set_to_none=True)
        samples = 0
        for _ in range(a.batch_accum):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=a.amp,
            ):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss * images.shape[0]).backward()
            samples += images.shape[0]
        scaler.unscale_(optimizer)
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        torch._foreach_div_(grads, samples)
        scaler.step(optimizer)
        scaler.update()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    warmup_started = time.perf_counter()
    for _ in range(a.warmup):
        update()
    sync(device)
    warmup_seconds = time.perf_counter() - warmup_started

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(a.steps):
        update()
    sync(device)
    seconds = time.perf_counter() - start

    peak_gib = (torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                if device.type == "cuda" else None)
    result = dict(method=method, layout=layout, seconds=seconds,
                  steps=a.steps, images_per_second=(a.steps * a.batch_accum *
                  a.batch_size / seconds),
                  ms_per_update=(1000 * seconds / a.steps),
                  peak_gib=peak_gib, warmup_and_compile_s=warmup_seconds)
    del model, optimizer, scaler, images, labels
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    a = parse_args()
    if a.batch_size <= 0 or a.batch_accum <= 0 or a.warmup <= 0 or a.steps <= 0:
        raise SystemExit("batch-size, batch-accum, warmup and steps must be positive")
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU available here. Run this script on your RTX 2080 Ti.")
    device = torch.device(f"cuda:{a.gpu}" if a.device == "cuda" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)  # ASAP currently creates CUDA tensors without a device argument.
    if a.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"torch={torch.__version__} device={device} ", end="", flush=True)
    if device.type == "cuda":
        print(torch.cuda.get_device_name(device), flush=True)
    else:
        print("(CPU: timing is NOT predictive of RTX 2080 Ti)", flush=True)
    print(f"batch={a.batch_size} accum={a.batch_accum} "
          f"AMP={a.amp} compile={a.compile} deterministic={a.deterministic}", flush=True)

    results = []
    for method in a.methods:
        per_method = {}
        for layout in ("contiguous", "channels_last"):
            print(f"\n[{method} / {layout}] warming up and benchmarking...", flush=True)
            try:
                r = run_one(method, layout, a, device)
            except Exception as e:
                print(f"FAILED ({type(e).__name__}): {e}", flush=True)
                gc.collect()
                if a.device == "cuda":
                    torch.cuda.empty_cache()
                continue
            results.append(r)
            per_method[layout] = r
            peak = f"  peak VRAM={r['peak_gib']:.2f} GiB" if r["peak_gib"] is not None else ""
            print(f"  {r['images_per_second']:.1f} images/s | "
                  f"{r['ms_per_update']:.1f} ms/update{peak} | "
                  f"warmup+compile={r['warmup_and_compile_s']:.1f}s", flush=True)
        if len(per_method) == 2:
            ratio = per_method["channels_last"]["images_per_second"] / per_method["contiguous"]["images_per_second"]
            print(f"=> {method}: channels_last = {ratio:.3f}x of contiguous "
                  f"({'FASTER' if ratio > 1 else 'SLOWER'})", flush=True)

    if results:
        path = Path(a.output)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved {len(results)} results to {path.resolve()}")


if __name__ == "__main__":
    main()
