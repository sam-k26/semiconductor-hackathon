from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from src.datasets.kla_dataset import KLADataset
from src.datasets.split import load_split_ids, subset_by_ids


GT_DIR = "data/official/train/train/GT"
NOISY_DIR = "data/official/train/train/NoisyLR"

# Must match scripts/evaluate.py exactly — the bicubic baseline is only a
# fair comparison against the model if it's measured on the identical
# validation images.
SPLIT_PATH = Path("data/splits/split_v2_leakage_safe.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():

    print("=" * 70)
    print("KLA BICUBIC BASELINE")
    print("=" * 70)

    print(f"Device: {DEVICE}")

    dataset = KLADataset(
        gt_dir=GT_DIR,
        noisy_dir=NOISY_DIR,
    )

    print(f"Dataset size: {len(dataset)}")

    # Recreate the exact same validation split as scripts/evaluate.py so
    # the two numbers are comparable.
    _, val_ids = load_split_ids(SPLIT_PATH)
    validation_dataset = subset_by_ids(dataset, val_ids)

    print(f"Validation samples: {len(validation_dataset)}")

    psnr_scores = []
    ssim_scores = []

    for i in range(len(validation_dataset)):

        sample = validation_dataset[i]

        noisy = sample["noisy"]
        gt = sample["gt"]

        # ---------------------------------------------------------
        # Add batch dimension
        # [1, 128, 128] -> [1, 1, 128, 128]
        # ---------------------------------------------------------

        noisy = noisy.unsqueeze(0).to(DEVICE)
        gt = gt.unsqueeze(0).to(DEVICE)

        # ---------------------------------------------------------
        # Bicubic upsampling
        # 128x128 -> 256x256
        # ---------------------------------------------------------

        prediction = F.interpolate(
            noisy,
            size=(256, 256),
            mode="bicubic",
            align_corners=False,
        )

        # ---------------------------------------------------------
        # KLA ground truth is [0, 1].
        # Clip prediction to valid image range.
        # ---------------------------------------------------------

        prediction = torch.clamp(
            prediction,
            0.0,
            1.0,
        )

        # ---------------------------------------------------------
        # Convert to numpy
        # ---------------------------------------------------------

        pred_np = prediction[0, 0].detach().cpu().numpy()
        gt_np = gt[0, 0].detach().cpu().numpy()

        # ---------------------------------------------------------
        # PSNR
        # ---------------------------------------------------------

        psnr = peak_signal_noise_ratio(
            gt_np,
            pred_np,
            data_range=1.0,
        )

        # ---------------------------------------------------------
        # SSIM
        # ---------------------------------------------------------

        ssim = structural_similarity(
            gt_np,
            pred_np,
            data_range=1.0,
        )

        psnr_scores.append(psnr)
        ssim_scores.append(ssim)

        # Progress
        if (i + 1) % 100 == 0:
            print(
                f"Processed {i + 1}/{len(validation_dataset)} | "
                f"PSNR: {np.mean(psnr_scores):.4f} | "
                f"SSIM: {np.mean(ssim_scores):.4f}"
            )

    # -------------------------------------------------------------
    # Final results
    # -------------------------------------------------------------

    mean_psnr = np.mean(psnr_scores)
    mean_ssim = np.mean(ssim_scores)

    print()
    print("=" * 70)
    print("FINAL BICUBIC BASELINE")
    print("=" * 70)

    print(f"Images : {len(validation_dataset)}")
    print(f"PSNR   : {mean_psnr:.4f} dB")
    print(f"SSIM   : {mean_ssim:.6f}")

    print("=" * 70)


if __name__ == "__main__":
    main()