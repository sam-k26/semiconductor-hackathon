import os
import zipfile

import numpy as np
import torch

from src.models.restoration_net import RestorationNet


# ============================================================
# Configuration
# ============================================================

TEST_NOISY_DIR = "data/official/Test_NoisyLR/NoisyLR"

OUTPUT_DIR = "results/predictions"

OUTPUT_ZIP = "results/predictions.zip"

CHECKPOINT = "checkpoints/best.pth"

BATCH_SIZE = 16

# Test-time augmentation: average predictions over the flip group
# (identity, h-flip, v-flip, 180-rotation). Each of these is its own
# inverse, so we flip, predict, then flip the prediction back the
# same way before averaging. Measured +0.055 dB PSNR / +0.0013 SSIM
# on the validation set (evaluate.py) — free, no retraining.
USE_TTA = True


def tta_predict(model, x):

    transforms = [
        lambda t: t,
        lambda t: torch.flip(t, dims=[3]),
        lambda t: torch.flip(t, dims=[2]),
        lambda t: torch.flip(t, dims=[2, 3]),
    ]

    predictions = []

    for transform in transforms:

        # Each transform above is self-inverse, so applying it a
        # second time to the prediction undoes it.
        prediction = transform(model(transform(x)))

        predictions.append(prediction)

    return torch.stack(predictions).mean(dim=0)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("KLA TEST PREDICTIONS")
    print("=" * 70)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Leave most cores for a training run that may be in progress
    # in another process; this is inference-only, so it doesn't
    # need much CPU to still finish quickly.
    if device.type == "cpu":
        torch.set_num_threads(2)

    print("Device:", device)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = RestorationNet().to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Loaded:", CHECKPOINT)
    print("TTA:", USE_TTA)

    print(
        "Checkpoint validation PSNR (internal metric):",
        f"{checkpoint.get('validation_psnr', float('nan')):.3f} dB",
    )

    # --------------------------------------------------------
    # Test files
    # --------------------------------------------------------

    test_files = sorted(
        f
        for f in os.listdir(TEST_NOISY_DIR)
        if f.endswith(".npy")
    )

    print("Test images:", len(test_files))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Inference in batches
    # --------------------------------------------------------

    with torch.no_grad():

        for start in range(0, len(test_files), BATCH_SIZE):

            batch_files = test_files[start:start + BATCH_SIZE]

            batch_arrays = [
                np.load(
                    os.path.join(TEST_NOISY_DIR, f)
                ).astype(np.float32)
                for f in batch_files
            ]

            batch_tensor = torch.from_numpy(
                np.stack(batch_arrays)
            ).unsqueeze(1).to(device)

            if USE_TTA:
                prediction = tta_predict(model, batch_tensor)
            else:
                prediction = model(batch_tensor)

            prediction = torch.clamp(
                prediction,
                0.0,
                1.0,
            )

            prediction = prediction.cpu().numpy()

            for i, f in enumerate(batch_files):

                out_path = os.path.join(
                    OUTPUT_DIR,
                    f,
                )

                np.save(
                    out_path,
                    prediction[i, 0].astype(np.float32),
                )

            done = start + len(batch_files)

            if done % 80 == 0 or done == len(test_files):
                print(f"Predicted {done}/{len(test_files)}")

    # --------------------------------------------------------
    # Zip for submission
    # --------------------------------------------------------

    with zipfile.ZipFile(
        OUTPUT_ZIP,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as zf:

        for f in sorted(os.listdir(OUTPUT_DIR)):
            zf.write(
                os.path.join(OUTPUT_DIR, f),
                arcname=f,
            )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print("Predictions dir:", OUTPUT_DIR)
    print("Zipped:", OUTPUT_ZIP)


if __name__ == "__main__":
    main()
