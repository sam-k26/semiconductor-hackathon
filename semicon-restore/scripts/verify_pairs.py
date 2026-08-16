from pathlib import Path
import numpy as np


GT_DIR = Path("data/official/train/train/GT")
NOISY_DIR = Path("data/official/train/train/NoisyLR")


def main():

    print("=" * 70)
    print("KLA TRAINING PAIR VERIFICATION")
    print("=" * 70)

    if not GT_DIR.exists():
        print(f"GT directory not found: {GT_DIR.resolve()}")
        return

    if not NOISY_DIR.exists():
        print(f"NoisyLR directory not found: {NOISY_DIR.resolve()}")
        return

    gt_files = {
        p.stem: p
        for p in GT_DIR.glob("*.npy")
    }

    noisy_files = {
        p.stem: p
        for p in NOISY_DIR.glob("*.npy")
    }

    gt_ids = set(gt_files)
    noisy_ids = set(noisy_files)

    common_ids = sorted(gt_ids & noisy_ids)
    missing_gt = sorted(noisy_ids - gt_ids)
    missing_noisy = sorted(gt_ids - noisy_ids)

    print()
    print(f"GT files       : {len(gt_files)}")
    print(f"NoisyLR files  : {len(noisy_files)}")
    print(f"Matched pairs  : {len(common_ids)}")
    print(f"Missing GT     : {len(missing_gt)}")
    print(f"Missing Noisy  : {len(missing_noisy)}")

    print()
    print("-" * 70)
    print("VERIFYING SAMPLE PAIRS")
    print("-" * 70)

    for image_id in common_ids[:10]:

        gt_path = gt_files[image_id]
        noisy_path = noisy_files[image_id]

        gt = np.load(gt_path, mmap_mode="r")
        noisy = np.load(noisy_path, mmap_mode="r")

        print()
        print(f"ID       : {image_id}")
        print(f"GT       : {gt.shape} | {gt.dtype}")
        print(f"NoisyLR  : {noisy.shape} | {noisy.dtype}")

        print(
            f"GT range      : "
            f"{gt.min():.6f} → {gt.max():.6f}"
        )

        print(
            f"NoisyLR range : "
            f"{noisy.min():.6f} → {noisy.max():.6f}"
        )

    print()
    print("=" * 70)

    if len(common_ids) == len(gt_files) == len(noisy_files):

        print("ALL TRAINING FILES ARE PAIRED ✓")

    else:

        print("PAIRING PROBLEM FOUND — CHECK ABOVE")

    print("=" * 70)


if __name__ == "__main__":
    main()