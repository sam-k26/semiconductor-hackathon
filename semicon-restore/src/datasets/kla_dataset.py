from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class KLADataset(Dataset):

    def __init__(self, gt_dir, noisy_dir):

        self.gt_dir = Path(gt_dir)
        self.noisy_dir = Path(noisy_dir)

        if not self.gt_dir.exists():
            raise FileNotFoundError(
                f"GT directory not found: {self.gt_dir}"
            )

        if not self.noisy_dir.exists():
            raise FileNotFoundError(
                f"NoisyLR directory not found: {self.noisy_dir}"
            )

        # Get files from GT directory
        gt_files = {
            path.stem: path
            for path in self.gt_dir.glob("*.npy")
        }

        # Get files from NoisyLR directory
        noisy_files = {
            path.stem: path
            for path in self.noisy_dir.glob("*.npy")
        }

        # Find matching IDs
        common_ids = sorted(
            set(gt_files) & set(noisy_files)
        )

        if len(common_ids) == 0:
            raise RuntimeError(
                "No matching GT/NoisyLR pairs found."
            )

        self.samples = [
            (
                image_id,
                noisy_files[image_id],
                gt_files[image_id],
            )
            for image_id in common_ids
        ]

        print(
            f"Loaded {len(self.samples)} "
            f"GT/NoisyLR pairs."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_id, noisy_path, gt_path = self.samples[index]

        # Load arrays
        noisy = np.load(noisy_path).astype(np.float32)
        gt = np.load(gt_path).astype(np.float32)

        # Verify the 2x super-resolution relationship, not a fixed size —
        # the problem statement covers both 128->256 and 256->512, and the
        # model is fully convolutional with no dependency on a specific
        # spatial size.
        expected_gt_shape = (noisy.shape[0] * 2, noisy.shape[1] * 2)

        if gt.shape != expected_gt_shape:
            raise ValueError(
                f"GT/NoisyLR shape mismatch for {image_id}: "
                f"NoisyLR {noisy.shape}, GT {gt.shape} "
                f"(expected {expected_gt_shape})"
            )

        # Convert NumPy → PyTorch
        noisy = torch.from_numpy(noisy)
        gt = torch.from_numpy(gt)

        # [H,W] → [1,H,W]
        noisy = noisy.unsqueeze(0)
        gt = gt.unsqueeze(0)

        return {
            "noisy": noisy,
            "gt": gt,
            "id": image_id,
        }