import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.kla_dataset import KLADataset
from src.models.restoration_net import RestorationNet


GT_DIR = "data/official/train/train/GT"
NOISY_DIR = "data/official/train/train/NoisyLR"


def main():

    print("=" * 60)
    print("TESTING ONE TRAINING STEP")
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset = KLADataset(
        gt_dir=GT_DIR,
        noisy_dir=NOISY_DIR,
    )

    print("Dataset size:", len(dataset))

    # ---------------------------------------------------------
    # DataLoader
    # IMPORTANT: num_workers=0 for Windows
    # ---------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    print("Getting first batch...", flush=True)

    batch = next(iter(loader))

    print("First batch loaded!", flush=True)

    noisy = batch["noisy"]
    gt = batch["gt"]

    print("Noisy shape:", noisy.shape)
    print("GT shape:", gt.shape)

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = RestorationNet().to(device)

    print("Model created.", flush=True)

    noisy = noisy.to(device)
    gt = gt.to(device)

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------

    print("Running forward pass...", flush=True)

    output = model(noisy)

    print("Forward pass complete!", flush=True)

    print("Output shape:", output.shape)

    # ---------------------------------------------------------
    # Loss
    # ---------------------------------------------------------

    criterion = nn.L1Loss()

    loss = criterion(
        output,
        gt,
    )

    print(
        "Loss:",
        loss.item(),
        flush=True
    )

    # ---------------------------------------------------------
    # Backward
    # ---------------------------------------------------------

    print(
        "Running backward pass...",
        flush=True
    )

    loss.backward()

    print(
        "Backward pass complete!",
        flush=True
    )

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    optimizer.step()

    print(
        "Optimizer step complete!",
        flush=True
    )

    print("=" * 60)
    print("✅ ONE TRAINING STEP PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()