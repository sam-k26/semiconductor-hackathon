from pathlib import Path
import numpy as np
import pandas as pd


DATASET_ROOT = Path("data/official")


def inspect_npy(path):
    try:
        arr = np.load(path, mmap_mode="r")

        return {
            "file": str(path),
            "shape": str(arr.shape),
            "dtype": str(arr.dtype),
            "ndim": arr.ndim,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }

    except Exception as e:
        return {
            "file": str(path),
            "shape": "ERROR",
            "dtype": "ERROR",
            "ndim": -1,
            "min": 0,
            "max": 0,
            "mean": 0,
            "std": 0,
            "error": str(e),
        }


def main():

    if not DATASET_ROOT.exists():
        print(f"Dataset not found:")
        print(DATASET_ROOT.resolve())
        return

    files = sorted(DATASET_ROOT.rglob("*.npy"))

    print("=" * 70)
    print("KLA NPY DATASET INSPECTION")
    print("=" * 70)

    print(f"\nDataset root:")
    print(DATASET_ROOT.resolve())

    print(f"\nTotal .npy files: {len(files)}\n")

    if not files:
        print("No .npy files found.")
        return

    records = []

    for i, path in enumerate(files):

        print(
            f"[{i + 1}/{len(files)}] {path}"
        )

        records.append(
            inspect_npy(path)
        )

    df = pd.DataFrame(records)

    Path("reports").mkdir(exist_ok=True)

    output = Path(
        "reports/dataset_statistics.csv"
    )

    df.to_csv(output, index=False)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nShapes:")
    print(df["shape"].value_counts())

    print("\nDtypes:")
    print(df["dtype"].value_counts())

    print("\nDimensions:")
    print(df["ndim"].value_counts())

    print("\nGlobal minimum:")
    print(df["min"].min())

    print("\nGlobal maximum:")
    print(df["max"].max())

    print("\nGlobal mean:")
    print(df["mean"].mean())

    print("\nStatistics saved to:")
    print(output.resolve())


if __name__ == "__main__":
    main()