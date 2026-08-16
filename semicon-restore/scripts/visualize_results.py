import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

# Allow importing src/
sys.path.insert(0, os.path.abspath("."))

from src.datasets.kla_dataset import KLADataset
from src.models.restoration_net import RestorationNet


# ============================================================
# Configuration
# ============================================================

GT_DIR = "data/official/train/train/GT"
NOISY_DIR = "data/official/train/train/NoisyLR"

CHECKPOINT = "checkpoints/best_28_4458.pth"

OUTPUT_DIR = "results/visualizations"

# These are validation samples used in our training split.
# We will use fixed IDs so the comparison is reproducible.
SAMPLE_IDS = [
    "000000",
    "000001",
    "000002",
    "000003",
    "000004",
]


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("KLA RESTORATION VISUALIZATION")
print("=" * 70)

print("Device:", device)


# ============================================================
# Dataset
# ============================================================

dataset = KLADataset(
    gt_dir=GT_DIR,
    noisy_dir=NOISY_DIR,
)

print("Dataset size:", len(dataset))


# ============================================================
# Model
# ============================================================

model = RestorationNet()

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False,
)

# Handle our checkpoint format
if "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("Loaded checkpoint:", CHECKPOINT)

if isinstance(checkpoint, dict):
    if "epoch" in checkpoint:
        print("Checkpoint epoch:", checkpoint["epoch"])

    if "validation_loss" in checkpoint:
        print(
            "Validation loss:",
            checkpoint["validation_loss"]
        )


# ============================================================
# Output directory
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# Helper functions
# ============================================================

def normalize_for_display(image):
    """
    Convert image to a display-safe [0,1] range.

    IMPORTANT:
    This is ONLY for visualization.
    It does not modify the actual model output.
    """
    image = image.astype(np.float32)

    minimum = image.min()
    maximum = image.max()

    if maximum > minimum:
        image = (
            image - minimum
        ) / (
            maximum - minimum
        )

    return image


def get_dataset_index(dataset, sample_id):
    """
    Find dataset index corresponding to an image ID.
    """

    for index in range(len(dataset)):

        if dataset[index]["id"] == sample_id:
            return index

    return None


# ============================================================
# Generate visualizations
# ============================================================

for sample_id in SAMPLE_IDS:

    print()
    print("-" * 70)
    print("Processing:", sample_id)

    index = get_dataset_index(
        dataset,
        sample_id,
    )

    if index is None:
        print(
            "WARNING: ID not found:",
            sample_id
        )
        continue

    sample = dataset[index]

    noisy = sample["noisy"].unsqueeze(0).to(device)
    gt = sample["gt"].numpy().squeeze()

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    with torch.no_grad():

        prediction = model(noisy)

    prediction = (
        prediction
        .cpu()
        .numpy()
        .squeeze()
    )

    noisy_image = (
        noisy
        .cpu()
        .numpy()
        .squeeze()
    )

    # --------------------------------------------------------
    # Bicubic baseline
    # --------------------------------------------------------

    import torch.nn.functional as F

    noisy_tensor = torch.from_numpy(
        noisy_image
    ).float()

    noisy_tensor = noisy_tensor.unsqueeze(0).unsqueeze(0)

    bicubic = F.interpolate(
        noisy_tensor,
        size=(256, 256),
        mode="bicubic",
        align_corners=False,
    )

    bicubic = (
        bicubic
        .numpy()
        .squeeze()
    )

    # --------------------------------------------------------
    # Print ranges
    # --------------------------------------------------------

    print(
        "NoisyLR range:",
        f"{noisy_image.min():.6f}",
        "→",
        f"{noisy_image.max():.6f}"
    )

    print(
        "Bicubic range:",
        f"{bicubic.min():.6f}",
        "→",
        f"{bicubic.max():.6f}"
    )

    print(
        "Prediction range:",
        f"{prediction.min():.6f}",
        "→",
        f"{prediction.max():.6f}"
    )

    print(
        "GT range:",
        f"{gt.min():.6f}",
        "→",
        f"{gt.max():.6f}"
    )

    # --------------------------------------------------------
    # Clip ONLY for visualization
    # --------------------------------------------------------

    noisy_display = normalize_for_display(
        noisy_image
    )

    bicubic_display = normalize_for_display(
        bicubic
    )

    prediction_display = np.clip(
        prediction,
        0.0,
        1.0,
    )

    gt_display = np.clip(
        gt,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4),
    )

    axes[0].imshow(
        noisy_display,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[0].set_title(
        "NoisyLR\n128×128"
    )

    axes[1].imshow(
        bicubic_display,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[1].set_title(
        "Bicubic\n256×256"
    )

    axes[2].imshow(
        prediction_display,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[2].set_title(
        "RestorationNet\n256×256"
    )

    axes[3].imshow(
        gt_display,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[3].set_title(
        "Ground Truth\n256×256"
    )

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"KLA Restoration — Sample {sample_id}",
        fontsize=14,
    )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{sample_id}_comparison.png",
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Saved:",
        output_path
    )


# ============================================================
# Done
# ============================================================

print()
print("=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)

print(
    "Results saved to:",
    OUTPUT_DIR
)