import json
import random
from pathlib import Path

import numpy as np


GT_DIR = Path("data/official/train/train/GT")
SPLIT_DIR = Path("data/splits")

SEED = 42
VALIDATION_RATIO = 0.10
THUMBNAIL_SIZE = 16

# Chosen from the nearest-neighbour distance distribution measured by
# check_split_leakage.py: p1=0.0039, p5=0.020, with a clear spike at 0.0
# for genuine duplicates. 0.01 sits comfortably between the duplicate
# spike and the naturally-similar bulk of the dataset.
DUPLICATE_THRESHOLD = 0.01


def thumbnail(image: np.ndarray, size: int = THUMBNAIL_SIZE) -> np.ndarray:
    h, w = image.shape
    block_h, block_w = h // size, w // size
    return image.reshape(size, block_h, size, block_w).mean(axis=(1, 3))


class UnionFind:

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def main() -> None:

    files = sorted(GT_DIR.glob("*.npy"))
    ids = [f.stem for f in files]
    n = len(files)

    print("=" * 70)
    print("BUILD LEAKAGE-SAFE SPLIT")
    print("=" * 70)
    print(f"GT images: {n}")

    thumbs = np.stack([
        thumbnail(np.load(f).astype(np.float32)).reshape(-1)
        for f in files
    ])

    sq_norms = np.sum(thumbs ** 2, axis=1)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * thumbs @ thumbs.T
    dist_sq = np.maximum(dist_sq, 0.0)

    threshold_sq = DUPLICATE_THRESHOLD ** 2
    edges = np.argwhere(np.triu(dist_sq < threshold_sq, k=1))

    uf = UnionFind(n)
    for i, j in edges:
        uf.union(int(i), int(j))

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        root = uf.find(i)
        clusters.setdefault(root, []).append(i)

    cluster_list = list(clusters.values())
    sizes = sorted((len(c) for c in cluster_list), reverse=True)

    print(f"Duplicate threshold: {DUPLICATE_THRESHOLD}")
    print(f"Edges below threshold: {len(edges)}")
    print(f"Clusters: {len(cluster_list)} (singletons: {sum(1 for s in sizes if s == 1)})")
    print(f"Non-singleton cluster sizes: {[s for s in sizes if s > 1]}")

    for cluster in cluster_list:
        if len(cluster) > 1:
            print(f"  cluster: {[ids[i] for i in cluster]}")

    # Greedy cluster-level assignment targeting VALIDATION_RATIO, whole
    # clusters only — this is what actually fixes the leakage, as opposed
    # to a random per-image split that can split a cluster across sides.
    rng = random.Random(SEED)
    order = list(range(len(cluster_list)))
    rng.shuffle(order)

    target_val = int(n * VALIDATION_RATIO)
    val_indices: list[int] = []
    train_indices: list[int] = []

    for idx in order:
        cluster = cluster_list[idx]
        if len(val_indices) + len(cluster) <= target_val:
            val_indices.extend(cluster)
        else:
            train_indices.extend(cluster)

    val_ids = sorted(ids[i] for i in val_indices)
    train_ids = sorted(ids[i] for i in train_indices)

    print()
    print(f"Train: {len(train_ids)}  Validation: {len(val_ids)}")

    # Sanity check: no edge below threshold should cross the new split.
    val_set = set(val_ids)
    leaks = 0
    for i, j in edges:
        a, b = ids[int(i)], ids[int(j)]
        if (a in val_set) != (b in val_set):
            leaks += 1
    print(f"Cross-split duplicate edges remaining: {leaks} (must be 0)")
    assert leaks == 0, "Cluster assignment failed to eliminate leakage"

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    split_path = SPLIT_DIR / "split_v2_leakage_safe.json"
    split_path.write_text(json.dumps({
        "seed": SEED,
        "validation_ratio": VALIDATION_RATIO,
        "duplicate_threshold": DUPLICATE_THRESHOLD,
        "train_ids": train_ids,
        "val_ids": val_ids,
    }, indent=2))

    print(f"Wrote {split_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
