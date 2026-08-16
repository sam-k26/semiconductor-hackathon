from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

from src.datasets.kla_dataset import KLADataset
from src.datasets.split import load_split_ids, subset_by_ids
from src.models.restoration_net import RestorationNet


GT_DIR = "data/official/train/train/GT"
NOISY_DIR = "data/official/train/train/NoisyLR"

# Cluster-aware split (scripts/build_split.py) — replaces a plain seeded
# random split after scripts/check_split_leakage.py found near-duplicate
# ground-truth images (adjacent indices, sequential filenames) straddling
# train/val under the old split.
SPLIT_PATH = Path("data/splits/split_v2_leakage_safe.json")

CHECKPOINT = "checkpoints/best.pth"

BATCH_SIZE = 4

# Test-time augmentation: average predictions over the flip group
# (identity, h-flip, v-flip, 180-rotation). Each of these is its own
# inverse, so we flip, predict, then flip the prediction back the
# same way before averaging. No retraining involved.
USE_TTA = True


def tta_predict(model, x):

    transforms = [
        lambda t: t,
        lambda t: torch.flip(t, dims=[3]),
        lambda t: torch.flip(t, dims=[2]),
        lambda t: torch.flip(t, dims=[2, 3]),
    ]

    predictions = []

    for transform in transforms:

        # Each transform above is self-inverse, so applying it a
        # second time to the prediction undoes it.
        prediction = transform(model(transform(x)))

        predictions.append(prediction)

    return torch.stack(predictions).mean(dim=0)


def main():

    print("=" * 70)
    print("KLA MODEL EVALUATION")
    print("=" * 70)

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

    # ---------------------------------------------------------
    # Cluster-aware validation split (data/splits/split_v2_leakage_safe.json)
    # ---------------------------------------------------------

    _, val_ids = load_split_ids(SPLIT_PATH)
    validation_dataset = subset_by_ids(dataset, val_ids)

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(
        "Validation samples:",
        len(validation_dataset)
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = RestorationNet().to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Loaded:", CHECKPOINT)
    print("TTA:", USE_TTA)

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    psnr_scores = []
    ssim_scores = []

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    with torch.no_grad():

        for batch_index, batch in enumerate(
            validation_loader
        ):

            noisy = batch["noisy"].to(device)
            gt = batch["gt"].to(device)

            if USE_TTA:
                prediction = tta_predict(model, noisy)
            else:
                prediction = model(noisy)

            prediction = torch.clamp(
                prediction,
                0.0,
                1.0
            )

            for i in range(
                prediction.shape[0]
            ):

                pred_np = (
                    prediction[i, 0]
                    .cpu()
                    .numpy()
                )

                gt_np = (
                    gt[i, 0]
                    .cpu()
                    .numpy()
                )

                psnr = peak_signal_noise_ratio(
                    gt_np,
                    pred_np,
                    data_range=1.0
                )

                ssim = structural_similarity(
                    gt_np,
                    pred_np,
                    data_range=1.0
                )

                psnr_scores.append(psnr)
                ssim_scores.append(ssim)

            if (batch_index + 1) % 10 == 0:

                print(
                    f"Batch "
                    f"{batch_index + 1}/"
                    f"{len(validation_loader)}"
                )

    # ---------------------------------------------------------
    # Final results
    # ---------------------------------------------------------

    mean_psnr = np.mean(psnr_scores)
    mean_ssim = np.mean(ssim_scores)

    print()
    print("=" * 70)
    print("FINAL MODEL RESULTS")
    print("=" * 70)

    print(
        f"PSNR : {mean_psnr:.4f} dB"
    )

    print(
        f"SSIM : {mean_ssim:.6f}"
    )

    print("=" * 70)

    print()
    print("BICUBIC BASELINE")
    print("----------------")
    print("PSNR : 22.8530 dB")
    print("SSIM : 0.5361")

    print()
    print("IMPROVEMENT")
    print("-----------")

    print(
        f"PSNR improvement : "
        f"{mean_psnr - 22.8530:+.4f} dB"
    )

    print(
        f"SSIM improvement : "
        f"{mean_ssim - 0.5361:+.6f}"
    )


if __name__ == "__main__":
    main()