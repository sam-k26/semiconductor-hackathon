from pathlib import Path
from collections import Counter
import numpy as np


FOLDERS = {
    "GT": Path("data/official/train/train/GT"),
    "NoisyLR": Path("data/official/train/train/NoisyLR"),
    "Test_NoisyLR": Path("data/official/Test_NoisyLR/NoisyLR"),
}


def inspect_folder(name, folder):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    files = sorted(folder.glob("*.npy"))

    print(f"Files: {len(files)}")

    shapes = Counter()
    dtypes = Counter()

    global_min = float("inf")
    global_max = float("-inf")

    for path in files:

        arr = np.load(path, mmap_mode="r")

        shapes[tuple(arr.shape)] += 1
        dtypes[str(arr.dtype)] += 1

        global_min = min(
            global_min,
            float(arr.min())
        )

        global_max = max(
            global_max,
            float(arr.max())
        )

    print("\nShapes:")

    for shape, count in shapes.items():
        print(f"  {shape}: {count}")

    print("\nDtypes:")

    for dtype, count in dtypes.items():
        print(f"  {dtype}: {count}")

    print(f"\nMinimum: {global_min}")
    print(f"Maximum: {global_max}")


def main():

    for name, folder in FOLDERS.items():

        if not folder.exists():

            print(
                f"\nWARNING: {folder} does not exist."
            )

            continue

        inspect_folder(
            name,
            folder
        )


if __name__ == "__main__":
    main()