import torch
from torch.utils.data import DataLoader

from src.datasets.kla_dataset import KLADataset


GT_DIR = "data/official/train/train/GT"
NOISY_DIR = "data/official/train/train/NoisyLR"


def main():

    print("=" * 70)
    print("KLA DATASET TEST")
    print("=" * 70)

    # --------------------------------------------------
    # Create dataset
    # --------------------------------------------------

    dataset = KLADataset(
        gt_dir=GT_DIR,
        noisy_dir=NOISY_DIR,
    )

    print()
    print(f"Dataset size: {len(dataset)}")

    # --------------------------------------------------
    # Test one sample
    # --------------------------------------------------

    sample = dataset[0]

    print()
    print("Single sample")
    print("-" * 70)

    print(f"ID          : {sample['id']}")
    print(f"Noisy shape : {sample['noisy'].shape}")
    print(f"GT shape    : {sample['gt'].shape}")

    print(f"Noisy dtype : {sample['noisy'].dtype}")
    print(f"GT dtype    : {sample['gt'].dtype}")

    print(
        f"Noisy min   : "
        f"{sample['noisy'].min().item():.6f}"
    )

    print(
        f"Noisy max   : "
        f"{sample['noisy'].max().item():.6f}"
    )

    print(
        f"GT min      : "
        f"{sample['gt'].min().item():.6f}"
    )

    print(
        f"GT max      : "
        f"{sample['gt'].max().item():.6f}"
    )

    # --------------------------------------------------
    # Test DataLoader
    # --------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(loader))

    noisy = batch["noisy"]
    gt = batch["gt"]

    print()
    print("First batch")
    print("-" * 70)

    print(
        f"Noisy batch shape : {noisy.shape}"
    )

    print(
        f"GT batch shape    : {gt.shape}"
    )

    print(
        f"IDs               : {batch['id']}"
    )

    print()
    print("=" * 70)
    print("DATASET TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()