from pathlib import Path
import numpy as np

ROOT = Path("data/official")

files = sorted(ROOT.rglob("*.npy"))

errors = []

for path in files:

    try:
        arr = np.load(path, mmap_mode="r")

    except Exception as e:

        errors.append((path, repr(e)))


print("=" * 80)
print("FAILED NPY FILES")
print("=" * 80)

print(f"\nTotal failed: {len(errors)}\n")

for path, error in errors[:30]:

    print("-" * 80)
    print("FILE:")
    print(path)

    print("\nERROR:")
    print(error)

print("\n" + "=" * 80)