import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.kla_dataset import KLADataset
from src.datasets.split import load_split_ids, subset_by_ids
from src.models.restoration_net import RestorationNet


# ============================================================
# Configuration
# ============================================================

GT_DIR = "data/official/train/train/GT"
NOISY_DIR = "data/official/train/train/NoisyLR"

# Cluster-aware split (scripts/build_split.py) — replaces a plain seeded
# random split after scripts/check_split_leakage.py found near-duplicate
# ground-truth images (adjacent indices, sequential filenames) straddling
# train/val under the old split.
SPLIT_PATH = Path("data/splits/split_v2_leakage_safe.json")

BATCH_SIZE = 4

# IMPORTANT: 100 epochs
NUM_EPOCHS = 100

LEARNING_RATE = 1e-4

# Kept at 0: this machine has ~7.5 GB RAM total and is often already
# under memory pressure, so extra worker subprocesses (each loading its
# own copy of torch) risk swapping and making things much slower, not
# faster. Only raise this if you've confirmed there's headroom.
NUM_WORKERS = 0

CHECKPOINT_DIR = "checkpoints"

SEED = 42

# Stop once validation PSNR hasn't improved for this many epochs, instead
# of always burning through NUM_EPOCHS regardless of convergence.
EARLY_STOPPING_PATIENCE = 8

# Warm-start model weights from an existing checkpoint instead of training
# from scratch. Set to None to disable and train from random init.
#
# Forced to None after the split was corrected for leakage
# (data/splits/split_v2_leakage_safe.json): checkpoints/best.pth was
# trained on the old split and would carry information about images that
# are now held out in validation, defeating the point of the re-split.
# The pre-correction checkpoint is preserved at
# checkpoints/best_pre_resplit_27_6319.pth.
RESUME_CHECKPOINT = None

# Loss weights
L1_WEIGHT = 0.7
MSE_WEIGHT = 0.3

# Learning-rate scheduler
LR_MIN = 1e-6


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# NOTE: PyTorch's default CPU thread count already targets physical
# cores (measured 8 threads on this machine's 8-core/16-thread CPU),
# which benchmarked faster than forcing all 16 logical (SMT) threads —
# convolutions are compute/memory-bound, so hyperthreads mostly add
# contention here rather than throughput. Left at the default.

# ------------------------------------------------------------
# GPU tuning
#
# The settings above (BATCH_SIZE=4, NUM_WORKERS=0, no mixed
# precision) are deliberately conservative for this CPU-only,
# ~7.5GB-RAM machine. A CUDA GPU wants the opposite: a bigger
# batch to fill its parallelism, background workers so the GPU
# is never waiting on disk, pinned memory for faster host->device
# transfer, and mixed precision (real speedup on GPU Tensor
# Cores, unlike the CPU bf16 attempt that backfired here).
#
# BATCH_SIZE=16 below is a conservative default for an unknown
# GPU. If you know your VRAM, raise it and re-check for
# out-of-memory errors: ~8GB -> try 32, ~16GB+ -> try 64+.
# LEARNING_RATE is scaled to match (sqrt rule).
# ------------------------------------------------------------

USE_AMP = False

PIN_MEMORY = False

if device.type == "cuda":

    BATCH_SIZE = 16

    NUM_WORKERS = 4

    PIN_MEMORY = True

    USE_AMP = True

    LEARNING_RATE = 2e-4

    # Input/output sizes are fixed (128x128 -> 256x256), so let
    # cuDNN benchmark and cache the fastest conv algorithms.
    torch.backends.cudnn.benchmark = True


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(prediction, target):
    """
    Calculate PSNR assuming images are in [0, 1].
    """

    mse = torch.mean(
        (prediction - target) ** 2
    )

    mse_value = mse.item()

    if mse_value <= 1e-12:
        return 100.0

    psnr = 10.0 * np.log10(
        1.0 / mse_value
    )

    return psnr


# ============================================================
# Data augmentation
# ============================================================

def augment(noisy, gt):
    """
    Apply identical geometric augmentation to noisy and GT images.
    """

    # Horizontal flip
    if random.random() < 0.5:
        noisy = torch.flip(
            noisy,
            dims=[3]
        )

        gt = torch.flip(
            gt,
            dims=[3]
        )

    # Vertical flip
    if random.random() < 0.5:
        noisy = torch.flip(
            noisy,
            dims=[2]
        )

        gt = torch.flip(
            gt,
            dims=[2]
        )

    # 90-degree rotation
    if random.random() < 0.5:
        k = random.randint(1, 3)

        noisy = torch.rot90(
            noisy,
            k=k,
            dims=[2, 3]
        )

        gt = torch.rot90(
            gt,
            k=k,
            dims=[2, 3]
        )

    return noisy, gt


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("KLA RESTORATION TRAINING")
    print("=" * 70)

    print("Device:", device)

    if device.type == "cpu":
        print("CPU threads:", torch.get_num_threads())

    print("Mixed precision (AMP):", USE_AMP)
    print("Epochs (max):", NUM_EPOCHS)
    print("Early stopping patience:", EARLY_STOPPING_PATIENCE)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = KLADataset(
        gt_dir=GT_DIR,
        noisy_dir=NOISY_DIR,
    )

    print("Total dataset:", len(dataset))

    # --------------------------------------------------------
    # Cluster-aware train / validation split
    # (data/splits/split_v2_leakage_safe.json — see scripts/build_split.py)
    # --------------------------------------------------------

    train_ids, val_ids = load_split_ids(SPLIT_PATH)

    train_dataset = subset_by_ids(dataset, train_ids)
    validation_dataset = subset_by_ids(dataset, val_ids)

    print(
        "Training samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(validation_dataset)
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    print(
        "Training batches:",
        len(train_loader)
    )

    print(
        "Validation batches:",
        len(validation_loader)
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = RestorationNet()

    model = model.to(device)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Model parameters:",
        f"{parameters:,}"
    )

    # --------------------------------------------------------
    # Resume (warm-start weights only; optimizer/scheduler are
    # kept fresh since batch size and LR have changed)
    # --------------------------------------------------------

    resume_validation_psnr = -float("inf")

    if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):

        print()
        print(f"Resuming weights from: {RESUME_CHECKPOINT}")

        resume_checkpoint = torch.load(
            RESUME_CHECKPOINT,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            resume_checkpoint["model_state_dict"]
        )

        resume_validation_psnr = resume_checkpoint.get(
            "validation_psnr",
            -float("inf"),
        )

        print(
            f"Checkpoint epoch: "
            f"{resume_checkpoint.get('epoch', '?')} | "
            f"Validation PSNR: {resume_validation_psnr:.3f} dB"
        )

    elif RESUME_CHECKPOINT:

        print()
        print(
            f"WARNING: RESUME_CHECKPOINT "
            f"'{RESUME_CHECKPOINT}' not found, "
            f"training from scratch."
        )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    l1_loss = nn.L1Loss()

    mse_loss = nn.MSELoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5,
    )

    # --------------------------------------------------------
    # AMP grad scaler (no-op when USE_AMP is False, i.e. on CPU)
    # --------------------------------------------------------

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    # --------------------------------------------------------
    # Learning-rate scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=LR_MIN,
    )

    # --------------------------------------------------------
    # Checkpoint directory
    # --------------------------------------------------------

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True,
    )

    # Seeded from the resumed checkpoint (if any) so a freshly warm-started
    # model can't overwrite best.pth with something worse than it started.
    best_validation_psnr = resume_validation_psnr

    best_validation_loss = float("inf")

    epochs_without_improvement = 0

    # ========================================================
    # Training loop
    # ========================================================

    for epoch in range(NUM_EPOCHS):

        print()
        print("=" * 70)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}"
        )

        print("=" * 70)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Learning rate: {current_lr:.8f}"
        )

        # ====================================================
        # Training
        # ====================================================

        model.train()

        training_loss = 0.0

        training_psnr = 0.0

        for batch_index, batch in enumerate(train_loader):

            noisy = batch["noisy"].to(
                device,
                non_blocking=PIN_MEMORY,
            )

            gt = batch["gt"].to(
                device,
                non_blocking=PIN_MEMORY,
            )

            # ------------------------------------------------
            # Augmentation
            # ------------------------------------------------

            noisy, gt = augment(
                noisy,
                gt
            )

            # ------------------------------------------------
            # Forward
            #
            # NOTE: fp16 autocast here is a real speedup on CUDA
            # Tensor Cores (USE_AMP). A CPU bf16 attempt was also
            # tried and measured dramatically SLOWER on this
            # machine's CPU (AMD Ryzen 5700U has no native BF16
            # support) — autocast is a no-op when USE_AMP is False.
            # ------------------------------------------------

            with torch.amp.autocast(
                "cuda",
                dtype=torch.float16,
                enabled=USE_AMP,
            ):

                prediction = model(noisy)

                # --------------------------------------------
                # Combined loss
                # --------------------------------------------

                l1 = l1_loss(
                    prediction,
                    gt
                )

                mse = mse_loss(
                    prediction,
                    gt
                )

                loss = (
                    L1_WEIGHT * l1
                    +
                    MSE_WEIGHT * mse
                )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            scaler.scale(loss).backward()

            # Unscale before clipping so max_norm is measured in
            # true gradient units, not the AMP-scaled ones.
            scaler.unscale_(optimizer)

            # Prevent unstable gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            scaler.step(optimizer)

            scaler.update()

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            training_loss += loss.item()

            training_psnr += calculate_psnr(
                prediction.detach(),
                gt
            )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (batch_index + 1) % 50 == 0:

                average_loss = (
                    training_loss
                    /
                    (batch_index + 1)
                )

                average_psnr = (
                    training_psnr
                    /
                    (batch_index + 1)
                )

                print(
                    f"Batch "
                    f"{batch_index + 1}/"
                    f"{len(train_loader)} | "
                    f"Loss: {average_loss:.6f} | "
                    f"PSNR: {average_psnr:.3f} dB"
                )

        # ----------------------------------------------------
        # Average training statistics
        # ----------------------------------------------------

        training_loss /= len(train_loader)

        training_psnr /= len(train_loader)

        # ====================================================
        # Validation
        # ====================================================

        model.eval()

        validation_loss = 0.0

        validation_psnr = 0.0

        with torch.no_grad():

            for batch in validation_loader:

                noisy = batch["noisy"].to(
                    device,
                    non_blocking=PIN_MEMORY,
                )

                gt = batch["gt"].to(
                    device,
                    non_blocking=PIN_MEMORY,
                )

                with torch.amp.autocast(
                    "cuda",
                    dtype=torch.float16,
                    enabled=USE_AMP,
                ):

                    prediction = model(noisy)

                    # Validation loss
                    l1 = l1_loss(
                        prediction,
                        gt
                    )

                    mse = mse_loss(
                        prediction,
                        gt
                    )

                    loss = (
                        L1_WEIGHT * l1
                        +
                        MSE_WEIGHT * mse
                    )

                validation_loss += loss.item()

                validation_psnr += calculate_psnr(
                    prediction,
                    gt
                )

        validation_loss /= len(
            validation_loader
        )

        validation_psnr /= len(
            validation_loader
        )

        # ====================================================
        # Scheduler
        # ====================================================

        scheduler.step()

        # ====================================================
        # Epoch results
        # ====================================================

        print()
        print("-" * 70)

        print(
            f"Training Loss   : "
            f"{training_loss:.6f}"
        )

        print(
            f"Validation Loss : "
            f"{validation_loss:.6f}"
        )

        print(
            f"Training PSNR   : "
            f"{training_psnr:.3f} dB"
        )

        print(
            f"Validation PSNR : "
            f"{validation_psnr:.3f} dB"
        )

        print(
            f"Learning Rate   : "
            f"{current_lr:.8f}"
        )

        # ====================================================
        # Save latest
        # ====================================================

        latest_path = os.path.join(
            CHECKPOINT_DIR,
            "latest.pth",
        )

        torch.save(
            {
                "epoch": epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "training_loss":
                    training_loss,

                "validation_loss":
                    validation_loss,

                "training_psnr":
                    training_psnr,

                "validation_psnr":
                    validation_psnr,
            },
            latest_path,
        )

        print(
            f"Saved: {latest_path}"
        )

        # ====================================================
        # Save best model based on PSNR
        # ====================================================

        if validation_psnr > best_validation_psnr:

            best_validation_psnr = validation_psnr

            best_validation_loss = validation_loss

            best_path = os.path.join(
                CHECKPOINT_DIR,
                "best.pth",
            )

            torch.save(
                {
                    "epoch": epoch + 1,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "scheduler_state_dict":
                        scheduler.state_dict(),

                    "training_loss":
                        training_loss,

                    "validation_loss":
                        validation_loss,

                    "training_psnr":
                        training_psnr,

                    "validation_psnr":
                        validation_psnr,
                },
                best_path,
            )

            print()
            print(
                "NEW BEST MODEL!"
            )

            print(
                f"Best validation PSNR: "
                f"{best_validation_psnr:.3f} dB"
            )

            print(
                f"Saved: {best_path}"
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

            print(
                f"No improvement for "
                f"{epochs_without_improvement} epoch(s) "
                f"(patience: {EARLY_STOPPING_PATIENCE})"
            )

        # ====================================================
        # Early stopping
        # ====================================================

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:

            print()
            print(
                f"Stopping early: validation PSNR has not "
                f"improved in {EARLY_STOPPING_PATIENCE} epochs."
            )

            break

    # ========================================================
    # Training complete
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Best validation PSNR:",
        f"{best_validation_psnr:.3f} dB"
    )

    print(
        "Best validation loss:",
        f"{best_validation_loss:.6f}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()