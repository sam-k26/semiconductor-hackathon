import torch

from src.models.restoration_net import RestorationNet


def main():

    print("=" * 60)
    print("TESTING RESTORATION NETWORK")
    print("=" * 60)

    # ---------------------------------------------------------
    # Select device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # ---------------------------------------------------------
    # Create model
    # ---------------------------------------------------------

    model = RestorationNet()

    model = model.to(device)

    print("Model created successfully.")

    # ---------------------------------------------------------
    # Fake KLA input
    #
    # Batch = 2
    # Channel = 1
    # Height = 128
    # Width = 128
    # ---------------------------------------------------------

    x = torch.randn(
        2,
        1,
        128,
        128,
    ).to(device)

    print("Input shape:", x.shape)

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------

    with torch.no_grad():

        y = model(x)

    print("Output shape:", y.shape)

    # ---------------------------------------------------------
    # Check output range
    # ---------------------------------------------------------

    print(
        "Output min:",
        y.min().item()
    )

    print(
        "Output max:",
        y.max().item()
    )

    # ---------------------------------------------------------
    # Parameter count
    # ---------------------------------------------------------

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Parameters:",
        f"{parameters:,}"
    )

    print("=" * 60)

    # ---------------------------------------------------------
    # Expected output
    # ---------------------------------------------------------

    expected = (
        y.shape
        ==
        torch.Size([2, 1, 256, 256])
    )

    if expected:

        print("✅ MODEL TEST PASSED")

    else:

        print("❌ MODEL TEST FAILED")


if __name__ == "__main__":
    main()