#!/usr/bin/env python3
"""
Standalone evaluation & inference script for KLA Semiconductor Image Restoration (PS01).

Usage:
    python eval.py --input <input_dir> --output <output_dir> [options]

Options:
    --input, -i       Directory containing input degraded .npy files (required)
    --output, -o      Directory where restored .npy files will be saved (required)
    --checkpoint, -c  Path to model weights checkpoint (default: checkpoints/best.pth)
    --batch-size, -b  Batch size for inference on homogeneous inputs (default: 16)
    --no-tta          Disable test-time augmentation (default: 4-way flip TTA enabled)
    --device          Explicit device override ('cuda', 'cpu', etc.)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch

# Ensure local repository modules are importable regardless of current working directory
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src.models.restoration_net import RestorationNet

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KLA Semiconductor Image Restoration — Standalone Evaluation Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to directory containing input degraded .npy files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Path to directory where restored .npy files will be written",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        default=SCRIPT_DIR / "checkpoints" / "best.pth",
        help="Path to model weights checkpoint (.pth file)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=16,
        help="Batch size for batchable inputs with identical spatial dimensions",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        default=False,
        help="Disable 4-way test-time augmentation (TTA)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda', 'cpu', etc.). If not specified, auto-detects CUDA.",
    )
    return parser.parse_args()


def get_device(device_override: str | None = None) -> torch.device:
    if device_override:
        return torch.device(device_override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tta_forward(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Test-time augmentation using the 4-element dihedral flip group:
    Identity, Horizontal Flip, Vertical Flip, and 180-degree Rotation.
    Each transformation is its own inverse.
    """
    transforms = [
        lambda t: t,
        lambda t: torch.flip(t, dims=[3]),
        lambda t: torch.flip(t, dims=[2]),
        lambda t: torch.flip(t, dims=[2, 3]),
    ]
    predictions: List[torch.Tensor] = []
    for transform in transforms:
        augmented_input = transform(x)
        raw_pred = model(augmented_input)
        inverted_pred = transform(raw_pred)
        predictions.append(inverted_pred)
    return torch.stack(predictions).mean(dim=0)


def load_model(checkpoint_path: Path, device: torch.device) -> RestorationNet:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint file not found at: {checkpoint_path.resolve()}"
        )

    model = RestorationNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def discover_input_files(input_dir: Path) -> List[Path]:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir.resolve()}")

    # Find all .npy files in the immediate directory, ignoring hidden files, subdirectories, etc.
    files = sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".npy" and not p.name.startswith(".")]
    )
    return files


def run_warmup(model: torch.nn.Module, device: torch.device, use_tta: bool) -> None:
    """Execute a forward pass with dummy tensor to warm up CUDA/PyTorch runtime."""
    dummy = torch.zeros((1, 1, 128, 128), dtype=torch.float32, device=device)
    with torch.inference_mode():
        if use_tta:
            _ = tta_forward(model, dummy)
        else:
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()


def process_batch(
    model: torch.nn.Module,
    batch_tensors: List[torch.Tensor],
    batch_paths: List[Path],
    output_dir: Path,
    device: torch.device,
    use_tta: bool,
) -> Tuple[int, int, float]:
    """
    Process a batch of valid tensors. If shapes match, processes as a single batch tensor.
    If spatial shapes vary, falls back to processing image by image.

    Returns (success_count, fail_count, model_only_seconds). model_only_seconds
    is wall-clock time spent strictly inside the forward pass(es) for this
    batch, bracketed by torch.cuda.synchronize() so it reflects actual
    execution rather than queued-but-unfinished kernels — kept separate from
    the caller's end-to-end timer (discover/read/H2D/forward/D2H/write) so
    I/O overhead is never hidden behind a flattering GPU-only number.
    """
    successes = 0
    failures = 0
    model_only_seconds = 0.0

    if not batch_tensors:
        return (0, 0, 0.0)

    # Check if all shapes in batch are identical
    first_shape = batch_tensors[0].shape
    same_shape = all(t.shape == first_shape for t in batch_tensors)

    if same_shape:
        try:
            stacked = torch.stack(batch_tensors, dim=0).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            forward_start = time.perf_counter()
            with torch.inference_mode():
                if use_tta:
                    out = tta_forward(model, stacked)
                else:
                    out = model(stacked)
            if device.type == "cuda":
                torch.cuda.synchronize()
            model_only_seconds += time.perf_counter() - forward_start
            out = torch.clamp(out, 0.0, 1.0)

            # Save individual outputs
            for i, p in enumerate(batch_paths):
                try:
                    res_np = out[i, 0].cpu().numpy().astype(np.float32)
                    out_path = output_dir / p.name
                    np.save(out_path, res_np)
                    successes += 1
                except Exception as save_err:
                    logger.error(f"Failed to write output for {p.name}: {save_err}")
                    failures += 1
            return (successes, failures, model_only_seconds)
        except Exception as batch_err:
            logger.warning(
                f"Batch execution failed with error ({batch_err}). Falling back to per-file processing."
            )
            model_only_seconds = 0.0

    # Fallback / heterogeneous shape handling: process item by item
    for tensor, path in zip(batch_tensors, batch_paths):
        try:
            x = tensor.unsqueeze(0).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            forward_start = time.perf_counter()
            with torch.inference_mode():
                if use_tta:
                    out = tta_forward(model, x)
                else:
                    out = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            model_only_seconds += time.perf_counter() - forward_start
            out = torch.clamp(out, 0.0, 1.0)
            res_np = out[0, 0].cpu().numpy().astype(np.float32)
            out_path = output_dir / path.name
            np.save(out_path, res_np)
            successes += 1
        except Exception as item_err:
            logger.error(f"Error restoring {path.name}: {item_err}")
            failures += 1

    return (successes, failures, model_only_seconds)


def main() -> int:
    args = parse_args()

    print("=" * 70)
    print(" KLA SEMICONDUCTOR IMAGE RESTORATION — EVALUATION HARNESS")
    print("=" * 70)

    device = get_device(args.device)
    logger.info(f"Target Device: {device.type.upper()}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"Test-Time Augmentation (TTA): {not args.no_tta}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Input Directory: {args.input.resolve()}")
    logger.info(f"Output Directory: {args.output.resolve()}")
    logger.info(f"Checkpoint: {args.checkpoint.resolve()}")

    # Ensure output directory exists
    args.output.mkdir(parents=True, exist_ok=True)

    # Load model
    try:
        model = load_model(args.checkpoint, device)
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded successfully: {param_count:,} parameters")
    except Exception as e:
        logger.critical(f"Failed to load model from {args.checkpoint}: {e}")
        return 1

    # Discover inputs
    try:
        input_files = discover_input_files(args.input)
    except Exception as e:
        logger.critical(f"Failed to read input directory: {e}")
        return 1

    total_files = len(input_files)
    logger.info(f"Discovered {total_files} .npy file(s) in {args.input}")

    if total_files == 0:
        logger.warning("No .npy files found to process. Exiting cleanly.")
        return 0

    # Warmup
    logger.info("Warming up model...")
    try:
        run_warmup(model, device, not args.no_tta)
    except Exception as e:
        logger.warning(f"Warmup encountered an issue ({e}), continuing to inference.")

    # Synchronize before starting official timing
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    total_successes = 0
    total_failures = 0
    total_model_seconds = 0.0

    current_batch_tensors: List[torch.Tensor] = []
    current_batch_paths: List[Path] = []

    for file_idx, file_path in enumerate(input_files):
        # Per-file loading error handling
        try:
            arr = np.load(file_path)
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"Expected numpy ndarray, got {type(arr).__name__}")
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D array [H, W], got shape {arr.shape}")
            
            tensor = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)  # [1, H, W]
            current_batch_tensors.append(tensor)
            current_batch_paths.append(file_path)
        except Exception as load_err:
            logger.error(f"Skipping unreadable or invalid file '{file_path.name}': {load_err}")
            total_failures += 1

        # When batch is full or at last file, process batch
        if len(current_batch_tensors) >= args.batch_size or (file_idx == total_files - 1 and current_batch_tensors):
            s, f, model_seconds = process_batch(
                model=model,
                batch_tensors=current_batch_tensors,
                batch_paths=current_batch_paths,
                output_dir=args.output,
                device=device,
                use_tta=not args.no_tta,
            )
            total_successes += s
            total_failures += f
            total_model_seconds += model_seconds
            current_batch_tensors = []
            current_batch_paths = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_time = time.perf_counter() - start_time

    # End-to-end: discover -> read -> tensor -> H2D -> forward -> D2H -> write.
    e2e_avg_ms = (elapsed_time / total_successes * 1000.0) if total_successes > 0 else 0.0
    e2e_throughput = (total_successes / elapsed_time) if elapsed_time > 0 else 0.0

    # Model-only: strictly the forward pass(es), summed across process_batch
    # calls and bracketed by torch.cuda.synchronize() in each. Reported
    # separately from end-to-end so I/O overhead is never hidden behind a
    # flattering GPU-only number.
    model_avg_ms = (
        (total_model_seconds / total_successes * 1000.0) if total_successes > 0 else 0.0
    )
    model_throughput = (
        (total_successes / total_model_seconds) if total_model_seconds > 0 else 0.0
    )

    print("=" * 70)
    print(" INFERENCE SUMMARY")
    print("=" * 70)
    logger.info(f"Total Discovered Files: {total_files}")
    logger.info(f"Successfully Restored:  {total_successes}")
    logger.info(f"Failed / Skipped Files: {total_failures}")
    logger.info(f"Total Elapsed Time:     {elapsed_time:.3f} s")
    logger.info(
        f"End-to-End Latency:     {e2e_avg_ms:.2f} ms/image "
        f"({e2e_throughput:.2f} images/sec) "
        "[discover+read+H2D+forward+D2H+write]"
    )
    logger.info(
        f"Model-Only Latency:     {model_avg_ms:.2f} ms/image "
        f"({model_throughput:.2f} images/sec) [forward pass only]"
    )
    print("=" * 70)

    return 0 if total_successes > 0 or total_files == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
