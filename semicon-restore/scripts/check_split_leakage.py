from pathlib import Path

import numpy as np
import torch


GT_DIR = Path("data/official/train/train/GT")

# Must match scripts/evaluate.py exactly — leakage is checked against the
# split actually used to report every metric so far.
SEED = 42
VALIDATION_RATIO = 0.10

THUMBNAIL_SIZE = 16


def thumbnail(image: np.ndarray, size: int = THUMBNAIL_SIZE) -> np.ndarray:
    h, w = image.shape
    block_h, block_w = h // size, w // size
    return image.reshape(size, block_h, size, block_w).mean(axis=(1, 3))


def main() -> None:

    files = sorted(GT_DIR.glob("*.npy"))
    ids = [f.stem for f in files]
    n = len(files)

    print("=" * 70)
    print("SPLIT LEAKAGE CHECK")
    print("=" * 70)
    print(f"GT images: {n}")

    thumbs = np.stack([
        thumbnail(np.load(f).astype(np.float32)).reshape(-1)
        for f in files
    ])

    # Pairwise squared L2 distance via the norm expansion trick, since a
    # direct (n, n, 256) broadcast would blow up memory at n=3200.
    sq_norms = np.sum(thumbs ** 2, axis=1)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * thumbs @ thumbs.T
    np.fill_diagonal(dist_sq, np.inf)
    dist_sq = np.maximum(dist_sq, 0.0)

    nn_idx = dist_sq.argmin(axis=1)
    nn_dist = np.sqrt(dist_sq[np.arange(n), nn_idx])

    print()
    print("Nearest-neighbour distance distribution (16x16 thumbnail L2):")
    for p in [0, 0.1, 0.5, 1, 5, 10, 25, 50, 75, 100]:
        print(f"  p{p:>5}: {np.percentile(nn_dist, p):.6f}")

    # Recreate the exact same split as scripts/evaluate.py.
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(n, generator=generator).tolist()
    validation_size = int(n * VALIDATION_RATIO)
    validation_indices = set(indices[-validation_size:])

    split_of = [
        "val" if i in validation_indices else "train"
        for i in range(n)
    ]

    print()
    print("-" * 70)
    print("CROSS-SPLIT NEAR-DUPLICATE CANDIDATES")
    print("-" * 70)

    for pct in [0.1, 0.5, 1.0, 2.0, 5.0]:

        threshold = np.percentile(nn_dist, pct)

        candidate_mask = nn_dist <= threshold
        candidate_pairs = [
            (i, int(nn_idx[i]))
            for i in np.where(candidate_mask)[0]
        ]

        cross_split = [
            (i, j) for i, j in candidate_pairs
            if split_of[i] != split_of[j]
        ]

        print(
            f"Bottom {pct:>4}% (threshold={threshold:.6f}): "
            f"{len(candidate_pairs)} candidate pairs, "
            f"{len(cross_split)} cross train/val"
        )

        if cross_split and pct in (0.1, 0.5):
            for i, j in cross_split[:10]:
                print(
                    f"    {ids[i]} ({split_of[i]}) <-> "
                    f"{ids[j]} ({split_of[j]}) "
                    f"dist={nn_dist[i]:.6f}"
                )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
