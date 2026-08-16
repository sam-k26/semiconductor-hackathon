import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.datasets.kla_dataset import KLADataset


GT_DIR = "data/official/train/train/GT"
NOISY_DIR = "data/official/train/train/NoisyLR"


def main():

    dataset = KLADataset(
        gt_dir=GT_DIR,
        noisy_dir=NOISY_DIR,
    )

    # Pick a few fixed samples
    indices = [0, 1, 2, 3]

    fig, axes = plt.subplots(
        len(indices),
        3,
        figsize=(12, 14)
    )

    for row, index in enumerate(indices):

        sample = dataset[index]

        noisy = sample["noisy"].unsqueeze(0)
        gt = sample["gt"].unsqueeze(0)

        # Bicubic upsampling
        bicubic = F.interpolate(
            noisy,
            size=(256, 256),
            mode="bicubic",
            align_corners=False,
        )

        noisy_img = noisy[0, 0].numpy()
        bicubic_img = bicubic[0, 0].numpy()
        gt_img = gt[0, 0].numpy()

        # NoisyLR
        axes[row, 0].imshow(
            noisy_img,
            cmap="gray"
        )

        axes[row, 0].set_title(
            f"NoisyLR\n{sample['id']} — 128×128"
        )

        # Bicubic
        axes[row, 1].imshow(
            bicubic_img,
            cmap="gray"
        )

        axes[row, 1].set_title(
            "Bicubic ×2"
        )

        # Ground Truth
        axes[row, 2].imshow(
            gt_img,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[row, 2].set_title(
            "Ground Truth — 256×256"
        )

        for col in range(3):
            axes[row, col].axis("off")

    plt.tight_layout()

    output = "reports/dataset_visualization.png"

    plt.savefig(
        output,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    print()
    print(f"Saved visualization to: {output}")


if __name__ == "__main__":
    main()